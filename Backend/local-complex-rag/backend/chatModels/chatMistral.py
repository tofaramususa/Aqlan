import os
from langchain_mistralai import ChatMistralAI

# def _set_env(var: str):
#     if not os.environ.get(var):
#         os.environ[var] = getpass.getpass(f"{var}: ")

# _set_env("MISTRAL_API_KEY")

# MISTRAL API KEY = kUhbEsQCqyNU8072kMthuWxqlbbikLP7


if "MISTRAL_API_KEY" not in os.environ:
    os.environ["MISTRAL_API_KEY"] = "kUhbEsQCqyNU8072kMthuWxqlbbikLP7"

llm = ChatMistralAI(
    model="codestral-latest",
    temperature=0,
    max_retries=2,
    # other params...
)

llm_json_mode = ChatMistralAI(
    model="codestral-latest",
    temperature=0,
    max_retries=2,
    format="json"
    # other params...
)

