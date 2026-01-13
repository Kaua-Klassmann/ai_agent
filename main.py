from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from dataclasses import dataclass

SYSTEM_PROMPT = "You are a helpful assistant"

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

model = init_chat_model(
    model="google_genai:gemini-2.5-flash-lite",
    temperature=0
)

@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    punny_response: str

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_weather],
    response_format=ToolStrategy(ResponseFormat)
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

print(response['structured_response'].punny_response)