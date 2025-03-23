from typing import Dict
from tenacity import RetryError
import logging
import uuid
import time
import re

from .base import AgentCore
from ..roles import BaseRole
from ...backends import IntelligenceBackend
from ...utils import extract_jsons

# A special signal sent by the player to indicate that it is not possible to continue the conversation, and it requests to end the conversation.
# It contains a random UUID string to avoid being exploited by any of the players.
SIGNAL_END_OF_CONVERSATION = f"<<<<<<END_OF_CONVERSATION>>>>>>{uuid.uuid4()}"
# The maximum number of retries when the query of backend fails
MAX_RETRIES = 5


class ReAct(AgentCore):
    """
    Reasoning and Action LLM-based Agent
    """
    def __init__(self, role: BaseRole, backend: IntelligenceBackend, global_prompt: str = None, **kwargs):
        super().__init__(role=role, backend=backend, global_prompt=global_prompt, **kwargs)
    
    def _construct_prompts(self, current_phase, history_messages, **kwargs):
        # Merge the role description and the global prompt as the system prompt for the agent
        if self.global_prompt:
            system_prompt = f"You are a good conversation game player.\n{self.global_prompt.strip()}\n\nYour name is {self.name}.\n\nYour role:{self.role_desc}"
        else:
            system_prompt = f"You are a good conversation game player. Your name is {self.name}.\n\nYour role:{self.role_desc}"
        
        # Concatenate conversations
        conversation_history = ""
        for msg in history_messages:
            conversation_history = f"{conversation_history}\n[{msg.agent_name}]: {msg.content}"
        
        # Instructions for different phases
        if "Night" in current_phase:
            user_prompt = f"""Now it is the Night phase. Notice that you are {self.name}. 
Based on the game rules, role descriptions and your experience, think about your acting strategy and take a proper action."""
        
        elif "Day" in current_phase:
            user_prompt = f"""Now it is the Day phase. Here are some conversation history you can refer to: {conversation_history}
Notice that you are {self.name} in the conversation. You should carefully analyze the conversation history since some ones might deceive during the conversation.
Based on the game rules, role descriptions and messages, think about what you are about to say in your following public speech."""
        
        elif "Voting" in current_phase:
            user_prompt = f"""Now it is the Voting phase. Here are some conversation history you can refer to: {conversation_history}
Notice that you are {self.name} in the conversation. You should carefully analyze the conversation history since some ones might deceive during the conversation.
Based on the game rules, role descriptions and messages, think about who is on your opposite team and then vote for this player."""
        
        else:
            user_prompt = ""
        
        return {"system_prompt": system_prompt, "user_prompt": user_prompt}

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
        
        retries = 0
        while retries < MAX_RETRIES:
            try:
                if "Night" in current_phase:
                    action_prompt = self.role.get_night_prompt()
                elif "Day" in current_phase:
                    action_prompt = self.role.get_day_prompt()
                else:
                    action_prompt = self.role.get_voting_prompt()
                
                response = self.backend.query(agent_name=self.name, 
                                              prompts=self._construct_prompts(current_phase=current_phase, 
                                                                              history_messages=observation["message_history"]), 
                                              request_msg=action_prompt)
                # print("Chosen Action: ", response)
                
                action_list = extract_jsons(response)
                if len(action_list) < 1:
                    raise ValueError(f"Player output {response} is not a valid json.")
                action = action_list[0]
                action["belief"] = ""
                action["strategy"] = ""
                if 'speech' in action:
                    response = self.backend.query(agent_name=self.name, 
                                              prompts=self.role.get_parse_prompt(action["speech"]))
                    sp_actions = re.findall(r'(\S+)\s*\|\s*(\S+)\s*\|\s*(\S+)', response)
                    action["sp_actions"] = sp_actions

                break  # if success, break the loop
            
            except (RetryError, ValueError, KeyError) as e:
                err_msg = f"Agent {self.name} failed to generate a response on attempt {retries}. Error: {e}."
                logging.warning(err_msg)

                if retries < MAX_RETRIES:
                    logging.info(f"Sleep {2**retries} seconds for the next retry.")
                    time.sleep(2**retries)
                else:
                    err_msg += "Reached maximum number of retries."
                    logging.warning(err_msg)
                    action = SIGNAL_END_OF_CONVERSATION + err_msg
                    return action
                
                retries += 1
            
        return action
