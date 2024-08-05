import re
import json
import os
from tenacity import retry, stop_after_attempt, wait_random_exponential
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

try:
    import openai
except ImportError:
    is_openai_available = False
else:
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    if openai.api_key is None:
        is_openai_available = False
    else:
        is_openai_available = True

try:
    import google.generativeai as genai
except ImportError:
    is_gemini_available = False
else:
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if google_api_key is None:
        is_gemini_available = False
    else:
        is_gemini_available = True
        genai.configure(api_key=google_api_key)


BACKEND_MODEL = {
    "gemini": "models/embedding-001",  # 768
    "openai": "text-embedding-ada-002"  # 1536
}


@retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
def get_embeddings(content, backend="gemini"):
    content = content.replace("\n\n", "\n").replace("\n", " ")
    if backend == "gemini":
        result = genai.embed_content(
            model=BACKEND_MODEL[backend],
            content=content,
            task_type="semantic_similarity"
        )
        embedding = result["embedding"]
    elif backend == "openai":
        result = openai.Embedding.create(
            input=content,
            model=BACKEND_MODEL[backend]
        )
        embedding = result.data[0].embedding
    else:
        embedding = []
    return embedding


def is_json(myjson):
    """
    Checks whether a given string is a valid JSON.

    Parameters:
        myjson (str): The string to be checked.

    Returns:
        bool: True if the string is a valid JSON, False otherwise.
    """
    try:
        json_object = json.loads(myjson)
    except ValueError as e:
        return False
    return True


def is_json_inside(text):
    """
    Checks whether a given string contains valid JSON(s).

    Parameters:
        text (str): The string to be checked.

    Returns:
        bool: True if the string contains valid JSON(s), False otherwise.
    """
    text = re.sub('\s+', ' ', text)
    matches = re.findall(r'\{.*?\}', text)
    for match in matches:
        if is_json(match):
            return True
    return False


def extract_code(text):
    """
    Extracts all code blocks encapsulated by '```' from a given string.

    Parameters:
        text (str): The string from which Python code blocks are to be extracted.

    Returns:
        List[str]: A list of all extracted Python code blocks.
    """
    text = re.sub('```python', '```', text)
    matches = re.findall(r'```(.*?)```', text, re.DOTALL)
    parsed_codes = []
    for match in matches:
        parsed_codes.append(match)
    return parsed_codes


def extract_jsons(text):
    """
    Extracts all valid JSON objects from a given string.

    Parameters:
        text (str): The string from which JSON objects are to be extracted.

    Returns:
        List[Dict]: A list of all extracted JSON objects.
    """
    text = re.sub('\s+', ' ', text)
    matches = re.findall(r'\{.*?\}', text)
    parsed_jsons = []
    for match in matches:
        try:
            json_object = json.loads(match)
            parsed_jsons.append(json_object)
        except ValueError as e:
            pass
    return parsed_jsons


class AttributedDict(dict):
    """
    A dictionary class whose keys are automatically set as attributes of the class. The dictionary is serializable to JSON.

    Inherits from:
        dict: Built-in dictionary class in Python.

    Note:
        This class provides attribute-style access to dictionary keys, meaning you can use dot notation
        (like `my_dict.my_key`) in addition to the traditional bracket notation (`my_dict['my_key']`).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError

    def __delattr__(self, key):
        del self[key]

    # check whether the key is string when adding the key
    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise ValueError("The key must be a string")
        super().__setitem__(key, value)

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value
