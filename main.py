from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from dataclasses import dataclass

SYSTEM_PROMPT = """You are a helpful assistant

You MUST ALWAYS respond using the ResponseFormat schema.
Never answer with plain text.
"""

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

@dataclass
class Context:
    """Custom runtime context schema"""
    user_id: int

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user infomation based on user ID"""
    user_id = runtime.context.user_id
    return "Rio de Janeiro" if user_id == 1 else "São Paulo"

model = init_chat_model(
    model="google_genai:gemini-2.5-flash-lite",
    temperature=0
)

@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    punny_response: str

checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_weather, get_user_location],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer
)

# Id para identificar qual chat está
config = {"configurable": {"thread_id": "1"}}

while True:
    response = agent.invoke(
        {"messages": [{"role": "user", "content": input()}]},
        config=config,
        context=Context(user_id=1)
    )

    print(response['structured_response'].punny_response)