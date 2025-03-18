import random
from typing import List, Dict, Union

from .base import Environment, TimeStep
from ..memory import Message, MessagePool
from ..agents.agent import SIGNAL_END_OF_CONVERSATION


class Werewolf(Environment):
    type_name = "werewolf"

    def __init__(self, player_names: List[str], roles_assigned: Dict[str, str], role_pool: List[str], max_discuss_round: int, **kwargs):
        super().__init__(player_names=player_names, roles_assigned=roles_assigned, role_pool=role_pool, max_discuss_round=max_discuss_round, **kwargs)
        self.roles_assigned = roles_assigned
        self.role_pool = role_pool
        self.roles_ground_truth = self.roles_assigned.copy()
        
        # The "state" of the environment is maintained by the message pool
        self.message_pool = MessagePool()
        
        # Game states, Night phase
        self._current_turn = 0
        self._next_player_idx = 0
        self._current_phase = "Begin"
        self._current_candidates = None
        
        # Day phase
        self.max_discuss_round = max_discuss_round
        self._discuss_round = 0

        # Voting phase
        self._players_votes = {name: 0 for name in self.player_names}
        self.winner = None

        self._initialized = False
        self.reset()
    
    def get_next_player(self) -> str:
        if self._current_phase != "Finish":
            return self.player_names[self._next_player_idx]
        else:
            return None
    
    def role_to_player(self, role, return_name=False):
        try:
            if return_name:  # Get the name of players corresponding to the role
                players = [player for player, player_role in self.roles_assigned.items() if player_role == role]
            else:  # Get the index of players corresponding to the role
                players = [self.player_names.index(player) for player, player_role in self.roles_assigned.items() if player_role == role]
            return players
        except KeyError:
            print(f"Role {role} not found in the roles dictionary.")
            return []

    def reset(self):
        self.message_pool.reset()
        self.roles_ground_truth = self.roles_assigned.copy()
        
        self._current_turn = 0
        self._next_player_idx = 0
        self._current_phase = "Begin"
        self._current_candidates = None
        
        self._discuss_round = 0
        
        self._players_votes = {name: 0 for name in self.player_names}
        self.winner = None
        
        self._moderator_speak("Welcome to the One Night Ultimate Werewolf game. Now please confirm your roles and close your eyes.")
        self._switch_to_werewolf()  # Start with Werewolves
        
        self._initialized = True
        init_timestep = TimeStep(observation=self.get_observation(self.get_next_player()),
                                 reward=self.get_zero_rewards(),
                                 terminal=False)

        return init_timestep
    
    def get_observation(self, player_name=None, only_message=True) -> Union[List[Message], Dict]:
        """
        get observation for the player
        """
        if player_name is None:
            message_history = self.message_pool.get_all_messages()
        else:
            message_history = self.message_pool.get_visible_messages(player_name, turn=self._current_turn)
        
        if only_message:
            return message_history
        else:
            observation = {
                "current_phase": self._current_phase, 
                "message_history": message_history,
                "current_players": self.player_names
            }
            return observation

    def _moderator_speak(self, text: str, visible_to: Union[str, List[str]] = "all"):
        """
        moderator say something
        """
        message = Message(agent_name="Moderator", content=text, turn=self._current_turn, visible_to=visible_to)
        self.message_pool.append_message(message)
        self._current_turn += 1

    def is_terminal(self) -> bool:
        """
        check if the conversation is over
        """
        # If the last message is the signal, then the conversation is over
        if self.message_pool.last_message.content.startswith(SIGNAL_END_OF_CONVERSATION) or self._current_phase == "Finish":
            return True
        return False
    
    def _switch_to_day(self):
        self._current_candidates = None
        self._next_player_idx = 0
        self._moderator_speak(f"Night phase ends. Everyone, wake up! Now we will start discussion from {self.player_names[self._next_player_idx]}.")
        self._current_phase = "Day"
    
    def _switch_to_insomniac(self):
        self._moderator_speak("Troublemaker, close your eyes. Insomniac, wake up. You can check your final role to see if it has changed.")
        insomniacs = self.role_to_player("Insomniac", return_name=True)
        if len(insomniacs) > 0:
            self._current_phase = "Night->Insomniac"
            for player_name in insomniacs:
                self._moderator_speak(f"Your final role is {self.roles_ground_truth[player_name]}.", visible_to=player_name)
        self._switch_to_day()
    
    def _switch_to_troublemaker(self):
        self._moderator_speak("Robber, close your eyes. Troublemaker, wake up. You may swap roles between two other players.")
        troublemakers = self.role_to_player("Troublemaker")
        if len(troublemakers) > 0:
            self._current_phase = "Night->Troublemaker"
            self._current_candidates = troublemakers
            self._next_player_idx = self._current_candidates.pop(0)
        else:
            self._switch_to_insomniac()
    
    def _switch_to_robber(self):
        self._moderator_speak("Seer, close your eyes. Robber, wake up. You may switch your role with another player, and then view your new role.")
        robbers = self.role_to_player("Robber")
        if len(robbers) > 0:
            self._current_phase = "Night->Robber"
            self._current_candidates = robbers
            self._next_player_idx = self._current_candidates.pop(0)
        else:
            self._switch_to_troublemaker()
    
    def _switch_to_seer(self):
        self._moderator_speak("Werewolves, close your eyes. Seer, wake up. You may check one another player's role or two roles in the role pool.")
        seers = self.role_to_player("Seer")
        if len(seers) > 0:
            self._current_phase = "Night->Seer"
            self._current_candidates = seers
            self._next_player_idx = self._current_candidates.pop(0)
        else:
            self._switch_to_robber()
    
    def _switch_to_werewolf(self):
        self._moderator_speak("Werewolves, wake up and look for other werewolves.")
        werewolves = self.role_to_player("Werewolf", return_name=True)
        if len(werewolves) > 0:
            self._current_phase = "Night->Werewolf"
            self._moderator_speak(f"All werewolves in the game: {', '.join(werewolves)}.", visible_to=werewolves)
        self._switch_to_seer()
    
    def night_step(self, player_name: str, action: Dict) -> TimeStep:
        if self._current_phase == "Night->Seer":
            check_player = action.get("player") if action.get("player") else ""
            check_player = check_player.lower()
            if check_player == "role pool":
                message = Message(agent_name=player_name, content="I would like to check two roles in role pool.", 
                                  thought=action.get("thought", ""), turn=self._current_turn, visible_to=player_name)
                self.message_pool.append_message(message)
                self._current_turn += 1
                roles_checked = random.sample(self.role_pool, 2)
                self._moderator_speak(f"The two roles you checked in role pool are: {', '.join(roles_checked)}.",
                                      visible_to=player_name)
            elif check_player in self.player_names:
                message = Message(agent_name=player_name, content=f"I would like to check {check_player}.", 
                                  thought=action.get("thought", ""), turn=self._current_turn, visible_to=player_name)
                self.message_pool.append_message(message)
                self._current_turn += 1
                self._moderator_speak(f"The role of {check_player} is {self.roles_ground_truth[check_player]}.", 
                                      visible_to=player_name)
            
            if len(self._current_candidates) == 0:
                self._switch_to_robber()
            else:
                self._next_player_idx = self._current_candidates.pop(0)
        
        elif self._current_phase == "Night->Robber":
            swap_player = action.get("player") if action.get("player") else ""
            swap_player = swap_player.lower()
            if action.get("switch", False) and swap_player in self.player_names:
                message = Message(agent_name=player_name, content=f"I want to switch my role with {swap_player}.", 
                                  thought=action.get("thought", ""), turn=self._current_turn, visible_to=player_name)
                self.message_pool.append_message(message)
                self._current_turn += 1
                self._moderator_speak(f"You switched your role with {swap_player}, and your new role is {self.roles_ground_truth[swap_player]}.", 
                                      visible_to=player_name)
                self.roles_ground_truth[player_name], self.roles_ground_truth[swap_player] = self.roles_ground_truth[swap_player], self.roles_ground_truth[player_name]
            else:
                message = Message(agent_name=player_name, content="I decide not to switch with others.", 
                                  thought=action.get("thought", ""), turn=self._current_turn, visible_to=player_name)
                self.message_pool.append_message(message)
                self._current_turn += 1
                self._moderator_speak("You did not switch with other player, so you remain your role.",
                                      visible_to=player_name)
            
            if len(self._current_candidates) == 0:
                self._switch_to_troublemaker()
            else:
                self._next_player_idx = self._current_candidates.pop(0)
        
        elif self._current_phase == "Night->Troublemaker":
            swap1 = action.get("player_1") if action.get("player_1") else ""
            swap2 = action.get("player_2") if action.get("player_2") else ""
            swap1, swap2 = swap1.lower(), swap2.lower()
            if action.get("swap", False) and swap1 in self.player_names and swap2 in self.player_names:
                message = Message(agent_name=player_name, content=f"I decide to swap roles between {swap1} and {swap2}.", 
                                  thought=action.get("thought", ""), turn=self._current_turn, visible_to=player_name)
                self.message_pool.append_message(message)
                self._current_turn += 1
                self._moderator_speak(f"You successfully swapped roles between {swap1} and {swap2}", 
                                      visible_to=player_name)
                self.roles_ground_truth[swap1], self.roles_ground_truth[swap2] = self.roles_ground_truth[swap2], self.roles_ground_truth[swap1]
            else:
                message = Message(agent_name=player_name, content="I decide not to swap others' role cards.", 
                                  thought=action.get("thought", ""), turn=self._current_turn, visible_to=player_name)
                self.message_pool.append_message(message)
                self._current_turn += 1
                self._moderator_speak("You did not swap other players' role cards.", 
                                      visible_to=player_name)
            
            if len(self._current_candidates) == 0:
                self._switch_to_insomniac()
            else:
                self._next_player_idx = self._current_candidates.pop(0)
        
        # print(self.roles_ground_truth)
        timestep = TimeStep(
            observation=self.get_observation(self.get_next_player()),
            reward=self.get_zero_rewards(),
            terminal=self.is_terminal()
        )
        
        return timestep
    
    def day_step(self, player_name: str, action: Dict) -> TimeStep:
        message = Message(agent_name=player_name, content=action.get("speech", ""), belief=action.get("belief", ""), 
                          strategy=action.get("strategy", ""), thought=action.get("thought", ""), tone=action.get("tone", ""), face=action.get("face", ""), turn=self._current_turn)
        self.message_pool.append_message(message)
        self._current_turn += 1

        if self._next_player_idx < len(self.player_names) - 1:
            self._next_player_idx += 1
        else:
            self._discuss_round += 1
            self._next_player_idx = 0
            if self._discuss_round < self.max_discuss_round:
                self._moderator_speak(f"Discussion round {self._discuss_round} ends, there are {self.max_discuss_round - self._discuss_round} rounds left (including current one). "
                                      f"Let's begin a new round, start with {self.player_names[self._next_player_idx]}.")
            else:
                self._moderator_speak("Day phase ends. Now vote which of the other players (excluding yourself) is the Werewolf. You cannot vote for yourself.")
                self._current_phase = "Voting"
        
        timestep = TimeStep(
            observation=self.get_observation(self.get_next_player()),
            reward=self.get_zero_rewards(),
            terminal=self.is_terminal()
        )

        return timestep
    
    def voting_step(self, player_name: str, action: Dict) -> TimeStep:
        vote = action.get("player") if action.get("player") else ""
        vote = vote.lower()
        if vote in self.player_names:
            self._players_votes[vote] += 1
            message = Message(agent_name=player_name, content=f"I am voting for {vote}.", belief=action.get("belief", ""), 
                              thought=action.get("thought", ""), turn=self._current_turn, visible_to=player_name)
        else:
            message = Message(agent_name=player_name, content="I give up my vote.", belief=action.get("belief", ""), 
                              thought=action.get("thought", ""), turn=self._current_turn, visible_to=player_name)
        self.message_pool.append_message(message)
        self._current_turn += 1

        if self._next_player_idx < len(self.player_names) - 1:
            self._next_player_idx += 1
        else:
            # check whether Werewolf exists among players
            exist_werewolf = False
            for role in self.roles_ground_truth.values():
                if role == "Werewolf":
                    exist_werewolf = True
            
            if not exist_werewolf:
                self._moderator_speak(f"Game over. There are no Werewolves, and everyone is on Team Village.")
                self.winner = "Draw"
            else:
                max_vote = max(self._players_votes.values())  # find all players with the most votes
                elected_players = [name for name, vote in self._players_votes.items() if vote == max_vote]
                
                if len(elected_players) == 1:  # only one player get the most votes
                    if self.roles_ground_truth[elected_players[0]] == "Werewolf":
                        self._moderator_speak(f"Game over. The player with the most votes is {elected_players[0]}. "
                                            f"And {elected_players[0]} is a Werewolf. Team Village wins.")
                        self.winner = "Team Village"
                    else:
                        self._moderator_speak(f"Game over. The player with the most votes is {elected_players[0]}. "
                                            f"And {elected_players[0]} is not a Werewolf. Team Werewolf wins.")
                        self.winner = "Team Werewolf"
                elif max_vote == 1:  # players with the most votes only got 1 vote
                        self._moderator_speak(f"Game over. Since players with the most votes only got 1 vote, no team wins.")
                        self.winner = "Draw"
                else:  # even votes, and max players got more than 1 vote
                    werewolf_player = None
                    for player in elected_players:
                        if self.roles_ground_truth[player] == "Werewolf":
                            werewolf_player = player
                    if werewolf_player is not None:
                        self._moderator_speak(f"Game over. There are even votes. And {werewolf_player} is a Werewolf. Team Village wins.")
                        self.winner = "Team Village"
                    else:
                        self._moderator_speak(f"Game over. There are even votes. But no Werewolf is voted out. Team Werewolf wins.")
                        self.winner = "Team Werewolf"
            
            self._current_phase = "Finish"
            # print(f"The final votes: {self._players_votes}")
        
        timestep = TimeStep(
            observation=self.get_observation(self.get_next_player()),
            reward=self.get_zero_rewards(),
            terminal=self.is_terminal()
        )

        return timestep
    
    def step(self, player_name: str, action: Dict) -> TimeStep:
        if not self._initialized:
            self.reset()
        assert player_name == self.get_next_player(), f"Wrong player! It is {self.get_next_player()} turn."

        if "Night" in self._current_phase:
            return self.night_step(player_name=player_name, action=action)
        elif "Day" in self._current_phase:
            return self.day_step(player_name=player_name, action=action)
        elif "Voting" in self._current_phase:
            return self.voting_step(player_name=player_name, action=action)
        else:
            raise ValueError("It is not a valid game phase. Please check the environment again.")
