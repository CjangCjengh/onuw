from typing import List, Union
from dataclasses import dataclass
import time
import hashlib


def _hash(input: str):
    """
    Helper function that generates a SHA256 hash of a given input string.

    Parameters:
        input (str): The input string to be hashed.

    Returns:
        str: The SHA256 hash of the input string.
    """
    hex_dig = hashlib.sha256(input.encode()).hexdigest()
    return hex_dig


@dataclass
class Message:
    """
    Represents a message in the chatArena environment.

    Attributes:
        agent_name (str): Name of the agent who sent the message.
        content (str): Content of the message.
        belief (str): Belief about other players' role of the agent.
        strategy (str): Speaking strategy of the agent.
        thought (str): Thought of the agent.
        turn (int): The turn at which the message was sent.
        timestamp (int): Wall time at which the message was sent. Defaults to current time in nanoseconds.
        visible_to (Union[str, List[str]]): The receivers of the message. Can be a single agent, multiple agents, or 'all'. Defaults to 'all'.
        msg_type (str): Type of the message, e.g., 'text'. Defaults to 'text'.
        logged (bool): Whether the message is logged in the database. Defaults to False.
    """
    agent_name: str
    content: str
    turn: int
    belief: str = ""
    strategy: str = ""
    thought: str = ""
    tone: str = ""
    face: str = ""
    timestamp: int = time.time_ns()
    visible_to: Union[str, List[str]] = 'all'
    msg_type: str = "text"
    logged: bool = False  # Whether the message is logged in the database

    @property
    def msg_hash(self):
        # Generate a unique message id given the content, timestamp and role
        return _hash(
            f"agent: {self.agent_name}\nbelief: {self.belief}\nspeaking_strategy: {self.strategy}\nthought: {self.thought}\ncontent: {self.content}\n"
            f"timestamp: {str(self.timestamp)}\nturn: {self.turn}\nmsg_type: {self.msg_type}"
        )
