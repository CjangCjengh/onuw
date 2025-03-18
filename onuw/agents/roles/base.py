from typing import List
from prompt_toolkit import prompt
from prompt_toolkit.styles import Style

SPEAKING_STRATEGY = {
    "honest_evidence": "You need to provide some honest evidence or information in your public speech, and your evidence must be consistent with the information or beliefs you know.",
    "deceptive_evidence": "You need to provide some misleading evidence or information in your public speech, and your evidence must be inconsistent with the information or beliefs you know.",
    "honest_accusation": "You need to accuse someone has a specific role or action honestly in your public speech, and your accusation must be consistent with the information or beliefs you know.",
    "deceptive_accusation": "You need to accuse someone has a specific role or action deceptively in your public speech, and your accusation must be misleading and inconsistent with the information or beliefs you know.",
    "honest_defense": "You need to defend yourself or someone else against an accusation honestly, and your defense must be consistent with the information or beliefs you know.",
    "deceptive_defense": "You need to defend yourself or someone else against an accusation deceptively, and your defense must be misleading and inconsistent with the information or beliefs you know."
}
speaking_strategies = ""
for idx, (name, desc) in enumerate(SPEAKING_STRATEGY.items()):
    speaking_strategies = f"{speaking_strategies}\n{idx+1}. {name}: {desc}"


class BaseRole:
    role_name = None

    def __init__(self, name: str):
        self.name = name
        self.current_players = []

        self.role_description = """The Moderator will call you on during the Night phase if necessary. Do not pretend you are the Moderator.
Concealing and deceiving are encouraged during the Day phase.
You can choose to hide your role, or even pretend you are other roles during the discussion. But your role may be changed by other players so your actual role may be different from what you saw at the first place.
You can reason other players' roles step by step.
Your response should be as concise as possible and should less than 50 words."""
    
    def update_current_players(self, current_players: List[str]):
        self.current_players = current_players.copy()
        self.current_players.remove(self.name)
    
    def get_night_prompt(self):
        raise NotImplementedError
    
    def get_day_prompt(self, speaking_strategy=""):
        if speaking_strategy == "":
            speech_prompt = f"""Now it is your turn, {self.name}.\n"""
        else:
            speech_prompt = f"""Now it is your turn, {self.name}. In this turn, your speaking strategy is: {speaking_strategy}\n"""
        speech_prompt += """Please give a concise but informative and specific public speech besed on your insights summarize from the conversation and following your speaking strategy. Your speaking goal is to convince other players to believe what you are going to say and induce them to say their true actions in the Night phase.
Remember, do not repeat statements after other players. And you should be cautious when deciding to reveal your thoughts (especially when you think you are Werewolf) in the public speech. Also, you should pay attention to the number of discussion rounds left while organizing your speech.
You must return your response in a JSON format that can be parsed by Python `json.loads`. Here is the Response Format:
{
    "thought": <the insights you summarized from the conversation and your speaking strategy>,
    "speech": <your public speech content, should less than 50 words>,
    "tone": <choose one from: sad, anger, neutral, happy, surprise, fear, disgust, other>,
    "face": <choose one from: sad, anger, neutral, happy, surprise, fear, disgust, other>
}
"""
        return speech_prompt
    
    def get_voting_prompt(self):
        voting_prompt = f"""Now it is your turn, {self.name}.
Please analyze current situation and vote for one player (excluding yourself) who you think is most likely a Werewolf. 
You can not vote for yourself, but only vote for one other player from the following options: [{', '.join(self.current_players)}].
You must return your response in a JSON format that can be parsed by Python `json.loads`. Here is the Response Format:
{{
    "thought": <the reason why you vote for this player>,
    "player": <the player you vote for>
}}
"""
        return voting_prompt
    
    def get_belief_prompt(self):
        belief_prompt = f"""Now it is your turn, {self.name}.
Please analyze current situation and think about what roles yourself ({self.name}) and other players ({', '.join(self.current_players)}) can most probably be now. You can reason each player's role step by step, based on the real or highly credible information you know.
Remember, you must give out the most likely role of each player (including yourself) in your concise response. And the number of each role have to be less or equal to the number of it in the candidate roles.
Give your step-by-step thought process and your derived concise result (no more that 2 sentences) at the end with following Response Format:
```
My step-by-step thought process: ...
My concise result: ...
Role guesses:
player1 -> role
player2 -> role 
player3 -> role
...
```
Note: If you cannot determine a player's role with confidence, you can use "unknown" as their role.
"""
        return belief_prompt
    
    def get_strategy_prompt(self):
        choosing_strategy_prompt = f"""Now it is your turn, {self.name}.
There are 6 speaking strategies you can choose: {speaking_strategies}
Please choose the most appropriate speaking strategy for your upcoming speech in this turn. Your speaking goal is to convince other players to believe what you are going to say and induce them to say their true actions in the Night phase.
You can only choose one speaking strategy from the options: [{', '.join(SPEAKING_STRATEGY.keys())}].
You must return your response in a JSON format that can be parsed by Python `json.loads`. Here is the Response Format:
{{
    "thought": <the reason why you choose this speaking strategy>,
    "strategy": <the speaking strategy you choose>
}}
"""
        return choosing_strategy_prompt
    
    def get_night_input(self):
        raise NotImplementedError

    def get_day_input(self):
        speech = prompt(
            [('class:user_prompt', "Type your speech content: ")],
            style=Style.from_dict({'user_prompt': 'ansicyan underline'})
        )
        return {"thought": "", "speech": speech}
    
    def get_voting_input(self):
        vote = prompt(
            [('class:info', "[Info] "),
             ('class:info_text', "You can vote for one other player from the following options: "),
             ('class:choices', f"{', '.join(self.current_players)}.\n"),
             ('class:user_prompt', "Type the player you want to vote for: ")],
            style=Style.from_dict({'info': "red bold", 'choices': "orange bold", 'user_prompt': 'ansicyan underline'})
        )
        return {"thought": "", "player": vote}
    
    def get_parse_prompt(self, speech: str):
        system_prompt = """You are a dialogue action parser for a Werewolf game. Your task is to extract structured actions from player dialogues.

Each action should be formatted as a triple: subject | predicate | object

Valid subjects: player1, player2, player3, player4, player5
Valid predicates: 
- point_as_werewolf
- point_as_villager
- point_as_seer
- point_as_troublemaker
- point_as_robber
- point_as_insomniac
- support
- oppose
Valid objects: player1, player2, player3, player4, player5

Extract ALL possible actions from the dialogue. If a player refers to themselves, use their player number.
Format each action on a separate line with following response format:
```
subject | predicate | object
...
```

Example:
Input:
player1: I am a villager. I think player3's statement makes sense, but player2 is definitely a werewolf.
Output:
```
player1 | point_as_villager | player1
player1 | support | player3
player1 | point_as_werewolf | player2
```"""
        user_prompt = f"Extract all actions from the following player dialogue:\n{self.name}: {speech}"""
        return {"system_prompt": system_prompt, "user_prompt": user_prompt}
