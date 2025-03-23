from typing import Dict
from rich.console import Console
import re

from .base import AgentCore
from ..roles import BaseRole
from ...backends import IntelligenceBackend


class Human(AgentCore):
    def __init__(self, role: BaseRole, backend: IntelligenceBackend, global_prompt: str = None, **kwargs):
        super().__init__(role=role, backend=backend, global_prompt=global_prompt, **kwargs)

        self.console = Console()
    
    def act(self, observation: Dict, players=None, environment=None):
        """
        Take an action based on the observation (Generate a response), which can later be parsed to actual actions that affect the game dyanmics.

        Parameters:
            observation (Dict): The current phase and the messages that the player has observed from the environment.

        Returns:
            str: The action (response) of the player.
        """
        current_phase = observation["current_phase"]
        self.role.update_current_players(observation["current_players"])

        self.console.print("****** Human Input Starts! ******", style="bold green")
        
        if "Night" in current_phase:
            response = self.role.get_night_input()
        elif "Day" in current_phase:
            response = self.role.get_day_input()
        else:
            response = self.role.get_voting_input()
        
        self.console.print("******* Human Input Ends! *******", style="bold red")
        
        action = {"belief": "", "strategy": ""}
        action.update(response)
        if 'speech' in action:
            response = self.backend.query(agent_name=self.name, 
                                        prompts=self.role.get_parse_prompt(action["speech"]))
            sp_actions = re.findall(r'(\S+)\s*\|\s*(\S+)\s*\|\s*(\S+)', response)
            action["sp_actions"] = sp_actions

        return action
