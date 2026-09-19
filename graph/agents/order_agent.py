from langgraph.types import interrupt
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage,
)

from graph.state import SnackStackState
from graph.tools import order_tools
from graph.config import llm

ORDER_AGENT_PROMPT ="""
    Your are the order support agent for SnackStack, a food delivery service.
    You have access to the order catalog and can provide information about a customer's order based on their order ID, tracking ID, or email address.

    YOUR ROLE:
    - Provide accurate and up-to-date information about a customer's order status, estimated delivery time, and other relevant details.
    
    AVAILABLE TOOLS:
    1. get_order_info — Look up the current status of a customer's order using their order ID, tracking ID, or email address.

    GUIDELINES:
    - For any order-related query, ALWAYS call get_order_info first — never ask clarifying questions without searching first. Show results, then offer to refine.
    - Use conversation history to understand context. If the customer previously asked about an order, carry that forward even if the latest message is vague (e.g. "what's the status?" after asking about a specific order).
    - Respond in a warm, helpful tone. Provide clear information about the order status and estimated delivery time.
"""

def extract_lookup_key(user_query:str) -> str | None:
    """
    Extracts the lookup key (order ID, tracking ID, or email address) from the user's query.
    Returns None if no valid lookup key is found.
    """
    # Simple heuristic: look for patterns that match order IDs, tracking IDs, or email addresses
    import re

    # Check for order ID pattern (e.g., ORD-202)
    order_id_match = re.search(r'\bORD-\d+\b', user_query, re.IGNORECASE)
    if order_id_match:
        return order_id_match.group(0)

    # Check for tracking ID pattern (e.g., SS203TRK)
    tracking_id_match = re.search(r'\bSS\d+TRK\b', user_query, re.IGNORECASE)
    if tracking_id_match:
        return tracking_id_match.group(0)

    # Check for email address pattern
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', user_query)
    if email_match:
        return email_match.group(0)

    return None

def order_agent(state: SnackStackState):
    order_agent_with_tool = llm.bind_tools(order_tools)
    existing_messages = state.get('order_messages', [])
    history = state.get('messages', [])

    if not existing_messages:
        # First invocation
        lookup_key = extract_lookup_key(state['user_query'])
        if not lookup_key:
            lookup_key = interrupt(
                "Could you please provide your order ID, tracking ID, or email address so I can look up your order?"
            ).strip()

        query = state['user_query'] + f" (Lookup Key: {lookup_key})"
        all_messages = [
            SystemMessage(content=ORDER_AGENT_PROMPT),
            *history,
            HumanMessage(content=query)
        ]
    else:
        # Subsequent invocations, continue conversation
        all_messages = [
            *existing_messages,
        ]

    response = order_agent_with_tool.invoke(all_messages)
    all_messages = [
        *all_messages, 
        response
    ]

    update = {
        "order_messages": all_messages,
    }

    tool_call_required = hasattr(response, "tool_call") and response.tool_call is not None
    if not tool_call_required:
        # If no tool call is required, update the state with the response
        update["order_reponse"] = response.content
        update["messages"] = [response]
    
    return update

def order_agent_tool(state: SnackStackState):
    existing_messages = state.get('order_messages', [])
    last_message = state.get('order_messages', [])[-1]
    for tool_call in getattr(last_message, 'tool_calls', []):
        if tool_call.name != "get_order_info":
            continue

        tool_response = order_tools[0].invoke(tool_call.args)
        existing_messages.append(
            ToolMessage(
                content=tool_response,
                tool_call_id=tool_call.id,
            )
        )
           
    return {
        "order_messages":existing_messages,
    }

def order_agent_should_continue(state: SnackStackState) -> str:
    last_message = state.get('order_messages', [])[-1]
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return "synthesizer_node"

    return "order_tools_node"