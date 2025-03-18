from typing import Dict
import os
import re
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .base import IntelligenceBackend

try:
    import openai
except ImportError:
    is_openai_available = False
else:
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    openai.base_url = os.environ.get("OPENAI_API_BASE")
    if openai.api_key is None:
        is_openai_available = False
    else:
        is_openai_available = True

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 256
DEFAULT_MODEL = "Qwen2.5-14B-Instruct"  # "gpt-3.5-turbo-1106"

END_OF_MESSAGE = "<EOS>"  # End of message token specified by us not OpenAI
STOP = ("<|endoftext|>", END_OF_MESSAGE)  # End of sentence token


class OpenAIChat(IntelligenceBackend):
    """
    Interface to the ChatGPT style model with system, user, assistant roles separation
    """
    stateful = False
    type_name = "openai-chat"

    def __init__(self, temperature: float = DEFAULT_TEMPERATURE, max_tokens: int = DEFAULT_MAX_TOKENS,
                 model: str = DEFAULT_MODEL, merge_other_agents_as_one_user: bool = True, **kwargs):
        """
        instantiate the OpenAIChat backend
        args:
            temperature: the temperature of the sampling
            max_tokens: the maximum number of tokens to sample
            model: the model to use
            merge_other_agents_as_one_user: whether to merge messages from other agents as one user message
        """
        assert is_openai_available, "openai package is not installed or the API key is not set"
        super().__init__(temperature=temperature, max_tokens=max_tokens, model=model,
                         merge_other_agents_as_one_user=merge_other_agents_as_one_user, **kwargs)

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model
        self.merge_other_agent_as_user = merge_other_agents_as_one_user
    
    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
    def _get_response(self, messages, *args, **kwargs):
        if kwargs.get("functions"):
            completion = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                functions=kwargs.get("functions"),
                function_call=kwargs.get("function_call", "auto"),
                stop=STOP
            )
        else:
            completion = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=STOP
            )

        response = completion.choices[0]['message']
        return response
    
    def query(self, agent_name: str, prompts: Dict[str, str], request_msg: str = None, *args, **kwargs) -> str:
        print("Using backend with :" + DEFAULT_MODEL)
        
        # Construct the prompts for ChatGPT
        messages = [
            {"role": "system", "content": prompts.get("system_prompt", "")},
            {"role": "user", "content": prompts.get("user_prompt", "")}
        ]

        # Specific action and desired JSON response format
        if request_msg:
            messages.append({"role": "system", "content": request_msg})
        else:  # The default request message that reminds the agent its role and instruct it to speak
            messages.append({"role": "system", "content": f"Now it is your turn, {agent_name}."})
        
        # Generate response
        response = self._get_response(messages, *args, **kwargs)

        # Post-process
        if response.get("function_call"):
            # If called a function, return the arguments
            response = response["function_call"]["arguments"].strip()
        else:
            # Else return the message content
            response = response["content"].strip()
            # Remove the agent name if the response starts with it
            response = re.sub(rf"^\s*\[.*]:", "", response).strip()
            response = re.sub(rf"^\s*{re.escape(agent_name)}\s*:", "", response).strip()

        return response
