from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load OPENAI API KEY from .env
load_dotenv()

# Initialize/Create the AI model
llm = ChatOpenAI(model="gpt-4o")

# Call LLM with a user prompt
response = llm.invoke("What is Agent Skills?")

# Print LLM Response
print(response.content)