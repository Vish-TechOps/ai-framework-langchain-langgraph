from dotenv import load_dotenv
from deepagents import create_deep_agent

# Load OPENAI API KEY from .env
load_dotenv()

# Define Tool/Function
# A plain Python function becomes a "tool" the agent can call.
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# Create the AI agent
agent = create_deep_agent(
    model="openai:gpt-5.5",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Call Agent with a user prompt. Agents are invoked with a "messages" list.
result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

# Print Agent Response
print(result["messages"][-1].content_blocks)