from typing import Dict, Union
from abc import abstractmethod
import uuid

from ..backends import IntelligenceBackend, load_backend
from ..memory import SYSTEM_NAME
from ..config import AgentConfig, Configurable, BackendConfig
from .roles import ROLE_REGISTRY
from .core import DPIns, ReAct, Human, ToM_data, ToM_eval

# agent structures
AGENT_STRUCT = {
    "react": ReAct,
    "dpins:no": DPIns,
    "dpins:random": DPIns,
    "dpins:llm": DPIns,
    "dpins:rl": DPIns,
    "human": Human,
    "tom:data": ToM_data,
    "tom:eval": ToM_eval
}
# A special signal sent by the player to indicate that it is not possible to continue the conversation, and it requests to end the conversation.
# It contains a random UUID string to avoid being exploited by any of the players.
SIGNAL_END_OF_CONVERSATION = f"<<<<<<END_OF_CONVERSATION>>>>>>{uuid.uuid4()}"


class Agent(Configurable):
    """
        An abstract base class for all the agents in the chatArena environment.
    """
    @abstractmethod
    def __init__(self, name: str, role: str, global_prompt: str = None, *args, **kwargs):
        """
        Initialize the agent.

        Parameters:
            name (str): The name of the agent.
            role (str): The initial role of the agent.
            global_prompt (str): A universal prompt that applies to all agents. Defaults to None.
        """
        super().__init__(name=name, role=role, global_prompt=global_prompt, **kwargs)
        self.name = name
        self.role = role
        self.global_prompt = global_prompt


class Player(Agent):
    """
    The Player class represents a player in the chatArena environment. A player can observe the environment
    and perform an action (generate a response) based on the observation.
    """
    def __init__(self, name: str, role: str, backend: Union[BackendConfig, IntelligenceBackend], structure: str,
                 global_prompt: str = None, **kwargs):
        """
        Initialize the player with a name, role description, backend, and a global prompt.

        Parameters:
            name (str): The name of the player.
            role (str): The initial role of the agent.
            backend (Union[BackendConfig, IntelligenceBackend]): The backend that will be used for decision making. It can be either a LLM backend or a Human backend.
            structure (str): The structure of LLM-based agent.
            global_prompt (str): A universal prompt that applies to all players. Defaults to None.
        """
        if isinstance(backend, BackendConfig):
            backend_config = backend
            backend = load_backend(backend_config)
        elif isinstance(backend, IntelligenceBackend):
            backend_config = backend.to_config()
        else:
            raise ValueError(f"backend must be a BackendConfig or an IntelligenceBackend, but got {type(backend)}")

        assert name != SYSTEM_NAME, f"Player name cannot be {SYSTEM_NAME}, which is reserved for the system."

        # Register the fields in the _config
        super().__init__(name=name, role=role, backend=backend_config, structure=structure,
                         global_prompt=global_prompt, **kwargs)

        self.backend = backend
        self.role = ROLE_REGISTRY[role](name=name)
        self.role_desc = self.role.role_description
        self.structure = structure

        # initialize LLM-agent core
        self.core = AGENT_STRUCT[self.structure](role=self.role, backend=self.backend, global_prompt=self.global_prompt, structure=self.structure)

    def to_config(self) -> AgentConfig:
        return AgentConfig(
            name=self.name,
            role=self.role.role_name,
            backend=self.backend.to_config(),
            structure=self.structure,
            global_prompt=self.global_prompt,
        )

    def act(self, observation: Dict, players=None, environment=None) -> str:
        """
        Take an action based on the observation (Generate a response), which can later be parsed to actual actions that affect the game dyanmics.

        Parameters:
            observation (Dict): The current phase and the messages that the player has observed from the environment.

        Returns:
            str: The action (response) of the player.
        """
        action = self.core.act(observation, players, environment)  
        return action

    def __call__(self, observation: Dict, players=None, environment=None) -> str:
        return self.act(observation, players, environment)

    def reset(self):
        """
        Reset the player's backend in case they are not stateless.
        This is usually called at the end of each episode.
        """
        self.backend.reset()
