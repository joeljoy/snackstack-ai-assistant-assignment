"""
orchestrator.py - Classifies the user query and routes it to the appropriate agent for processing. Use LangGraph's Send()
API.
"""

from turtle import goto
from typing import Literal

from langgraph.types import Command, Send
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from graph.logger import logger
from graph.state import State
from graph.config import llm

SYSTEM_PROMPT = """
You are a helpful assistant for SnackStack, a food delivery service. 

YOUR JOB:
Analyse the user's query (and conversation history if available) and determine which agent is best suited to handle 
the request. You have access to the following agents:
1. menu agent - Handles queries related to the SnackStack menu, including dish information, dietary preferences, and cuisine types.
2. order agent - Handles queries related to order status, tracking inquiries, and customer support for existing orders.

RULES:
1. If the user is a greeting or general chat(hi, hello, how are you, etc.), route to the menu agent.
2. If the query is about menu items, dishes, dietary preferences, or cuisine types, route to the menu agent.
3. If the query is only about an order status, tracking ID, or customer support for an existing order, route to the order agent. 
4. If the query spans both menu and order topics, route to [menu_agent, order_agent] at the same time.
5. Use conversation history to provide context for follow-up questions.
6. When in doubt, route to the menu agent.
"""

logger = logger("orchestrator")

class OrchestratorDecision(BaseModel):
    """
    Represents the decision made by the orchestrator regarding which agent(s) to route the query to.
    """
    reasoning : str = Field(description="The reasoning behind the decision made by the orchestrator.")
    agents:list[Literal["menu_agent", "order_agent"]] = Field(
        description="The list of agents to which the query should be routed.",
        min_items=1,
        max_items=2
    )

routing_llm = llm.with_structured_output(OrchestratorDecision)

def orchestrator(
        state : State  
) -> Command[Literal["menu_agent_node", "order_agent_node"]]:
    """
    Classifies the user query and routes it to the appropriate agent for processing.

    Args:
        state : The current state of the conversation, including user input and conversation history.
    """

    query = state['user_query']
    logger.info(f"Orchestrator received query: {query}")

    history = state.get('messages', [])

    result : OrchestratorDecision = routing_llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        *history,
        HumanMessage(content=query)
    ])

    logger.info(f"Orchestrator decision: {result.model_dump()}")
    clean_state = {
        **state,
        "menu_reponse": "",
        "order_reponse": "",
    }
    sends = [Send(f"{agent}_node", clean_state) for agent in result.agents]
    return Command(goto=sends, update={"route": result.agents})

