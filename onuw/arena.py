from typing import List, Dict, Union
import uuid
import json
import csv
import logging
import random

from .agents.agent import Player
from .environments import Environment, TimeStep, load_environment
from .backends import Human
from .config import ArenaConfig


class TooManyInvalidActions(Exception):
    pass


class Arena:
    """
    Utility class that manages the game environment and players
    """

    def __init__(self, players: List[Player], environment: Environment, global_prompt: str = None):
        # Create a container for the players and environment and reset the game
        self.players = players
        self.environment = environment
        self.global_prompt = global_prompt

        self.current_timestep = environment.reset()
        self.uuid = uuid.uuid4()  # Generate a unique id for the game
        self.invalid_actions_retry = 5

    @property
    def num_players(self):
        return self.environment.num_players

    @property
    def name_to_player(self) -> Dict[str, Player]:
        return {player.name: player for player in self.players}

    def reset(self) -> TimeStep:
        # Reset the environment
        self.current_timestep = self.environment.reset()
        # Reset the players
        for player in self.players:
            player.reset()
        # Reset the uuid
        self.uuid = uuid.uuid4()
        return self.current_timestep

    def step(self) -> TimeStep:
        """
        Take a step in the game: one player takes an action and the environment updates
        """
        player_name = self.environment.get_next_player()
        player = self.name_to_player[player_name]  # get the player object
        observation = self.environment.get_observation(player_name, only_message=False)  # get the observation for the player

        timestep = None
        for i in range(self.invalid_actions_retry):  # try to take an action for a few times
            action = player(observation)  # take an action
            if self.environment.check_action(action, player_name):  # action is valid
                timestep = self.environment.step(player_name, action)  # update the environment
                break
            else:  # action is invalid
                logging.warning(f"{player_name} made an invalid action {action}")
                continue

        if timestep is None:  # if the player made invalid actions for too many times, terminate the game
            warning_msg = f"{player_name} has made invalid actions for {self.invalid_actions_retry} times. Terminating the game."
            logging.warning(warning_msg)
            raise TooManyInvalidActions(warning_msg)

        return timestep

    def next_is_human(self):
        """
        check if the next player is human
        """
        player_name = self.environment.get_next_player()
        player = self.name_to_player[player_name]
        return isinstance(player.backend, Human)

    def run(self, num_steps: int = 1):
        """
        run the game for num_turns
        """
        for i in range(num_steps):
            timestep = self.step()
            if timestep.terminal:
                break

    @classmethod
    def from_config(cls, config: Union[str, ArenaConfig], randomness: bool = False):
        """
        create an arena from a config
        """
        # If config is a path, load the config
        if isinstance(config, str):
            config = ArenaConfig.load(config)

        global_prompt = config.get("global_prompt", None)
        role_pool = config.environment.get("role_pool", None).copy()
        assert len(role_pool) == len(config.players) + 3, "The number of roles in role pool must be 3 more than the number of players."
        if randomness:  # if randomness=True, shuffle role pool
            random.shuffle(role_pool)
        
        # Create the players
        players, roles_assigned = [], {}
        for player_config in config.players:
            # Turning players' names to lowercase
            player_config["name"] = player_config["name"].lower()
            # Add public_prompt to the player config
            if global_prompt is not None:
                player_config["global_prompt"] = global_prompt
            
            # if randomness=True, re-assign roles to players
            if randomness:
                player_config["role"] = role_pool.pop(0)
            else:  # else just remove the assigned roles
                role_pool.remove(player_config["role"])
            roles_assigned[player_config["name"]] = player_config["role"]

            player = Player.from_config(player_config)
            players.append(player)

        # Check that the player names are unique
        player_names = [player.name for player in players]
        assert len(player_names) == len(set(player_names)), "Player names must be unique"

        # Create the environment
        config.environment["player_names"] = player_names  # add the player names to the environment config
        config.environment["roles_assigned"] = roles_assigned
        config.environment["role_pool"] = role_pool  # record remained roles to the environment config
        env = load_environment(config.environment)

        return cls(players, env, global_prompt=global_prompt)

    def to_config(self) -> ArenaConfig:
        """
        convert the arena to a config
        """
        return ArenaConfig(
            players=[player.to_config() for player in self.players],
            environment=self.environment.to_config(),
            global_prompt=self.global_prompt
        )

    def launch_cli(self, max_steps: int = None, interactive: bool = True):
        """
        launch the command line interface
        """
        from onuw.ui.cli import ArenaCLI
        cli = ArenaCLI(self)
        cli.launch(max_steps=max_steps, interactive=interactive)

    def save_config(self, path: str):
        """
        save the config to a file
        """
        config = self.to_config()
        config.save(path)

    def save_history(self, path: str):
        """
        save the history of the game to a file
        Supports csv and json formats.
        """
        messages = self.environment.get_observation()
        message_rows = []

        if path.endswith(".csv"):
            header = ["agent_name", "belief", "strategy", "content", "face", "tone", "guesses", "sp_actions", "thought", "turn", "timestamp", "visible_to", "msg_type"]
            for message in messages:
                message_row = [
                    message.agent_name,
                    message.belief,
                    message.strategy,
                    message.content,
                    message.face,
                    message.tone,
                    message.guesses,
                    message.sp_actions,
                    message.thought,
                    message.turn,
                    str(message.timestamp),
                    message.visible_to,
                    message.msg_type,
                ]
                message_rows.append(message_row)

            with open(path, "w") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(message_rows)
        elif path.endswith(".json"):
            for message in messages:
                message_row = {
                    "agent_name": message.agent_name,
                    "belief": message.belief,
                    "strategy": message.strategy,
                    "content": message.content,
                    "face": message.face,
                    "tone": message.tone,
                    "guesses": message.guesses,
                    "sp_actions": message.sp_actions,
                    "thought": message.thought,
                    "turn": message.turn,
                    "timestamp": str(message.timestamp),
                    "visible_to": message.visible_to,
                    "msg_type": message.msg_type,
                }
                message_rows.append(message_row)
            
            backends = {}
            for player in self.players:
                backend_name = player.backend.type_name
                if backend_name == "openai-chat":
                    if "gpt-3.5" in player.backend.model:
                        backend_name = "chatgpt-3.5"
                    elif "gpt-4" in player.backend.model:
                        backend_name = "chatgpt-4"
                backends[player.name] = backend_name
            history = {
                "messages": message_rows,
                "evaluation": {
                    "roles_assigned": self.environment.roles_assigned,
                    "roles_ground_truth": self.environment.roles_ground_truth,
                    "role_pool": self.environment.role_pool,
                    "player_backends": backends,
                    "voting_result": self.environment._players_votes,
                    "winner": self.environment.winner
                },
            }

            with open(path, "w") as f:
                json.dump(history, f, indent=4)
        else:
            raise ValueError("Invalid file format")
