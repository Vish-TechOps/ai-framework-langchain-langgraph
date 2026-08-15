import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load .env
load_dotenv()

# Read litellm credentials
litellm_key = os.getenv("LITELLM_KEY")
litellm_url = os.getenv("LITELLM_URL")

# ChatOpenAI to use your litellm endpoint, not api.openai.com
llm = ChatOpenAI(
    model="gpt-4o",                  # confirm the model name available on litellm
    api_key=litellm_key,             # litellm key
    base_url=litellm_url             # litellm url
)

response = llm.invoke("What is Agent Skilks?")

print(response.content)