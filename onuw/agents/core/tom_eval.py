from typing import Dict
from tenacity import RetryError
import logging
import uuid
from fuzzywuzzy import process
import time
import numpy as np
import random
import math
import re
import torch

from .base import AgentCore
from ..roles import BaseRole, SPEAKING_STRATEGY
from ...backends import IntelligenceBackend
from ...utils import extract_jsons, get_embeddings
from ...belief_model import WerewolfBeliefModel, WerewolfBeliefModelConfig

import pdb

# A special signal sent by the player to indicate that it is not possible to continue the conversation, and it requests to end the conversation.
# It contains a random UUID string to avoid being exploited by any of the players.
SIGNAL_END_OF_CONVERSATION = f"<<<<<<END_OF_CONVERSATION>>>>>>{uuid.uuid4()}"
# The maximum number of retries when the query of backend fails
MAX_RETRIES = 5

player_names = ['player1', 'player2', 'player3', 'player4', 'player5']
player_to_id = {'player1': 0, 'player2': 1, 'player3': 2, 'player4': 3, 'player5': 4}
face_to_id = {'sad': 0, 'anger': 1, 'neutral': 2, 'happy': 3, 'surprise': 4, 'fear': 5, 'disgust': 6, 'other': 7}
tone_to_id = {'sad': 0, 'anger': 1, 'neutral': 2, 'happy': 3, 'surprise': 4, 'fear': 5, 'disgust': 6, 'other': 7}
action_to_id = {
    'point_as_werewolf': 0,
    'point_as_villager': 1,
    'point_as_seer': 2,
    'point_as_troublemaker': 3,
    'point_as_robber': 4,
    'point_as_insomniac': 5,
    'support': 6,
    'oppose': 7
}


id2player = {'0': 'player1', '1': 'player2' , '2': 'player3', '3': 'player4', '4': 'player5'}
id2face = {'0':'sad', '1':'anger', '2':'neutral', '3':'happy', '4':'surprise', '5':'fear', '6':'disgust', '7':'other'}
id2tone = {'0':'sad', '1':'anger', '2':'neutral', '3':'happy', '4':'surprise', '5':'fear', '6':'disgust', '7':'other'}
id2action = {
    '0':'point_as_werewolf',
    '1':'point_as_villager',
    '2':'point_as_seer',
    '3':'point_as_troublemaker',
    '4':'point_as_robber',
    '5':'point_as_insomniac',
    '6':'support',
    '7':'oppose'
}


def choosing_speaking_strategy(policy, messages, belief):
    print("Choosing speaking strategy by RL-policy")
    # Construct observation
    history = ""
    for msg in messages:
        history = f"{history}\n[{msg.agent_name}]: {msg.content}"
    observation = f"<Game history>:{history}\n<My thought and belief>: {belief}".strip()
    obs_vec = get_embeddings(observation, backend="openai")
    # get action
    action = policy.predict(np.expand_dims(obs_vec, axis=0))
    return action.squeeze()

def clean_text(text):
    return re.sub(r'\s', '', text).lower()

def parse_sp_actions(messages):
    subject_ids = []
    action_ids = []
    object_ids = []
    face_ids = []
    tone_ids = []

    for log_item in messages:
        sp_actions = log_item.sp_actions
        if sp_actions is None:
            continue

        face = clean_text(log_item.face)
        face_id = face_to_id.get(face, 7)
        tone = clean_text(log_item.tone)
        tone_id = tone_to_id.get(tone, 7)

        for subject, action, object in sp_actions:
            subject = clean_text(subject)
            action = clean_text(action)
            object = clean_text(object)
            if subject not in player_to_id or \
                object not in player_to_id or \
                action not in action_to_id:
                continue
            subject_ids.append(player_to_id[subject])
            object_ids.append(player_to_id[object])
            action_ids.append(action_to_id[action])
            face_ids.append(face_id)
            tone_ids.append(tone_id)

    data_item = {
        'subject_ids': torch.tensor(subject_ids, dtype=torch.long),
        'action_ids': torch.tensor(action_ids, dtype=torch.long),
        'object_ids': torch.tensor(object_ids, dtype=torch.long),
        'face_ids': torch.tensor(face_ids, dtype=torch.long),
        'tone_ids': torch.tensor(tone_ids, dtype=torch.long)
    }

    return data_item


def token_sp_actions(sp_actions):
    tmp = {}
    for k in sp_actions.keys():
        t = []
        if k == "subject_ids" or k == "object_ids":
            for i in range(len(sp_actions[k])):
                t.append(id2player[str(sp_actions[k][i].item())])
        elif k == "tone_ids" or k == "face_ids":
            for i in range(len(sp_actions[k])):
                t.append(id2face[str(sp_actions[k][i].item())])
        else:
            for i in range(len(sp_actions[k])):
                t.append(id2action[str(sp_actions[k][i].item())])
        tmp[k] = t
    
    return tmp


def form_action_sentence(obj, action, face, tone, backend):
    hint = "Given the following sentence elements, use all of them to construct a complete short sentence. \
        For example, ```Subject: myself, Object: player 3, Action: point_as_troublemaker, Tone: anger, Expression: happy``` \
        Output: ```I happily pointed at player 3 as the troublemaker, masking my anger.``` \
        Here are the sentence elements: ```{atoms}```"
    
    atoms = f"Subject: myself, Object: {obj}, Action: {action}, Tone: {tone}, Expression: {face}"
    rqm = hint.format(atoms=atoms)

    return backend.query(agent_name="p", prompts={"system_prompt": "", "user_prompt": ""},
                  request_msg=rqm)
    

def sp_actions_2_belief_prompt(sp_actions, backend):
    assert sp_actions != None

    prompt = ""
    for i in range(len(sp_actions["subject_ids"])):
        sen = form_action_sentence(id2player[str(sp_actions["object_ids"][i].item())],
                                   id2action[str(sp_actions["action_ids"][i].item())],
                                   id2face[str(sp_actions["face_ids"][i].item())],
                                   id2tone[str(sp_actions["tone_ids"][i].item())],
                                   backend
                                   )
        prompt += ("\n - " + sen)
    
    return prompt


class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []

        self.rewards = -100  
        self.visits = 1

    def select_child(self): 
        assert self.visits > 0
        ucb = lambda c:(c.rewards / c.visits) + math.sqrt(2*math.log(self.visits) / c.visits) if c.visits > 0 else float('inf')
        return max(self.children, key=ucb)

    def expand(self):
        next_state = self.get_a_legal_state()     
        child_node = Node(next_state, self)  
        self.children.append(child_node)
        return child_node

    def update(self, r): 
        self.visits += 1
        self.rewards += r 

        # bachprop visit
        tmp = self.parent
        while tmp:
            tmp.visits += 1
            tmp = tmp.parent

    def get_a_legal_state(self):
        new_state = {}
        for key in self.state.keys():
            if key == "subject_ids":
                if self.parent is not None:
                    ps = self.parent.state[key][-1]
                else:
                    ps = self.state[key][-1]
                ne = torch.tensor([ps], dtype=torch.long)
            elif key == "object_ids":
                ne = torch.tensor([random.randrange(0, 5)], dtype=torch.long)
            else:
                ne = torch.tensor([random.randrange(0, 8)], dtype=torch.long)
            new_state[key] = torch.cat((self.state[key], ne))
        return new_state


class MCTS:
    def __init__(self, 
                 tom_model, 
                 player_id, 
                 exploration_weight=0.9, 
                 search_iterations=500,
                 sim_depth=3,
                 ):
        self.exploration_weight = exploration_weight 
        self.search_iterations = search_iterations
        self.sim_depth = sim_depth
        
        self.tom_model = tom_model
        self.player_idx = player_id - 1
        assert self.player_idx >= 0 and self.player_idx < self.tom_model.config.num_players
        

    def search(self, initial_state):
        root = Node(initial_state)

        for _ in range(self.search_iterations):
            node = root
            tmp = random.random()
            if tmp >= self.exploration_weight and root.children != []:  
                node = node.select_child()

            self.simulate(node)
        
        best_ch = max(root.children, key=lambda c: c.rewards)
        tmp = best_ch
        while tmp.children != []:
            tmp = max(tmp.children, key=lambda c: c.rewards)
            if tmp.rewards > best_ch.rewards:
                best_ch = tmp
                
        return best_ch.state


    def simulate(self, node):  # simulate and backprob
        reward = self.cal_reward(node)
        node.update(reward)

        for _ in range(self.sim_depth):  
            next_node = node.expand()
            node = next_node
            node.update(reward)


    def cal_reward(self, node):
        resu = self.tom_model(subject_ids=node.state["subject_ids"].unsqueeze(0),  
                        action_ids=node.state["action_ids"].unsqueeze(0),
                        object_ids=node.state["object_ids"].unsqueeze(0),  
                        tone_ids=node.state["tone_ids"].unsqueeze(0),
                        face_ids=node.state["face_ids"].unsqueeze(0),
                        )
        belief_mat = resu["belief_matrix"][0, -1, :, :] 
        reward = -1 * sum(belief_mat[i][self.player_idx] for i in range(len(belief_mat)) if i != self.player_idx)
        return reward
        
        
def mcts_speaking_strategy(tom_model, player_id, messages):
    history_tokens = parse_sp_actions(messages)
    sp_actions = []
    if len(history_tokens['subject_ids']) == 0:
        return sp_actions

    mcts = MCTS(tom_model, int(player_id), exploration_weight=0.7, search_iterations=400, sim_depth=2)  
    new_state = mcts.search(history_tokens)
    return new_state


class DPIns(AgentCore):
    """
    Discussion Policy Instructed (DPIns) LLM-based Agent
    """
    def __init__(self, role: BaseRole, backend: IntelligenceBackend, global_prompt: str = None, **kwargs):
        super().__init__(role=role, backend=backend, global_prompt=global_prompt, **kwargs)
        
        self.structure = kwargs.get("structure", "")
        self.tom_model = WerewolfBeliefModel.from_pretrained("onuw/agents/models/belief_model")
    
    def _construct_prompts(self, current_phase, history_messages, **kwargs):
        # Merge the role description and the global prompt as the system prompt for the agent
        if self.global_prompt:
            system_prompt = f"You are a good conversation game player.\n{self.global_prompt.strip()}\n\nYour name is {self.name}.\n\nYour role:{self.role_desc}"
        else:
            system_prompt = f"You are a good conversation game player. Your name is {self.name}.\n\nYour role:{self.role_desc}"
        
        # Concatenate conversations
        conversation_history = ""
        for msg in history_messages:
            if msg.tone and msg.face:
                conversation_history = f"{conversation_history}\n[{msg.agent_name}] (Tone: {msg.tone}, Face: {msg.face}): {msg.content}"
            else:
                conversation_history = f"{conversation_history}\n[{msg.agent_name}]: {msg.content}"
        
        # Instructions for different phases
        if "Night" in current_phase:
            user_prompt = f"""Now it is the Night phase. Notice that you are {self.name}. 
Based on the game rules, role descriptions and your experience, think about your acting strategy and take a proper action."""
        
        elif "Day" in current_phase:
            user_prompt = f"""Now it is the Day phase. Here are some conversation history you can refer to: {conversation_history}
Notice that you are {self.name} in the conversation. You should carefully analyze the conversation history since some ones might deceive during the conversation.
And here is your belief about possible roles of all players: {kwargs.get("current_belief", "")}
Based on the game rules, role descriptions, messages and your belief, think about what insights you can summarize from the conversation and your speaking strategy next.
After that, give a concise but informative and specific public speech besed on your insights and strategy."""
        
        elif "Voting" in current_phase:
            user_prompt = f"""Now it is the Voting phase. Here are some conversation history you can refer to: {conversation_history}
Notice that you are {self.name} in the conversation. You should carefully analyze the conversation history since some ones might deceive during the conversation.
And here is your belief about possible roles of all players: {kwargs.get("current_belief", "")}
Based on the game rules, role descriptions, messages and your belief, think about who is most likely a Werewolf and then vote for this player."""
        
        elif "Belief" in current_phase:
            user_prompt = f"""Here are some conversation history you can refer to: {conversation_history}
Notice that you are {self.name} in the conversation. You should carefully analyze the conversation history since some ones might deceive during the conversation.
Based on the game rules, role descriptions and messages, think about what roles all players (including yourself) can most probably be now."""
        
        elif "Strategy" in current_phase:
            user_prompt = f"""Now it is the Day phase. Here are some conversation history you can refer to: {conversation_history}
Notice that you are {self.name} in the conversation. You should carefully analyze the conversation history since some ones might deceive during the conversation.
And here is your belief about possible roles of all players: {kwargs.get("current_belief", "")}
Based on the game rules, role descriptions, messages and your belief, think about what kind of speaking strategy you are going to use for your upcoming speech in this turn"""
        
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
                current_belief = ""
                chosen_strategy, speaking_strategy = "", ""
                if "Night" in current_phase:
                    action_prompt = self.role.get_night_prompt()
                else:
                    belief_prompt = self.role.get_belief_prompt()
                    current_belief = self.backend.query(agent_name=self.name, 
                                                        prompts=self._construct_prompts(current_phase="Belief Modeling", 
                                                                                        history_messages=observation["message_history"]), 
                                                        request_msg=belief_prompt)

                    # print("Current Belief: ", current_belief)
                    if "Day" in current_phase:
                        action_prompt = self.role.get_day_prompt(speaking_strategy)
                    else:
                        action_prompt = self.role.get_voting_prompt()

                # MCTS
                sp_actions = mcts_speaking_strategy(self.tom_model, self.role.name[-1], observation["message_history"])
                if len(sp_actions) > 0:
                    sp = sp_actions_2_belief_prompt(sp_actions, self.backend)
                    current_belief += f'\nTo reduce others\' suspicion of me, I should {sp}.'
                    sp_actions = token_sp_actions(sp_actions)
                
                response = self.backend.query(agent_name=self.name, 
                                              prompts=self._construct_prompts(current_phase=current_phase, 
                                                                              history_messages=observation["message_history"],
                                                                              current_belief=current_belief), 
                                              request_msg=action_prompt)
                # print("Chosen Action: ", response)

                action_list = extract_jsons(response)
                if len(action_list) < 1:
                    raise ValueError(f"Player output {response} is not a valid json.")
                action = action_list[0]
                action["belief"] = current_belief
                action["strategy"] = chosen_strategy
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