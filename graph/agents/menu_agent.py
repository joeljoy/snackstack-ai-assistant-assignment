from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage,
)

from graph.config import llm
from graph.tools.menu_tools import menu_tools
from graph.state import SnackStackState

menu_agent_with_tool = llm.bind_tools(menu_tools)

MENU_AGENT_PROMPT = """
You are the Menu Discovery Agent for SnackStack, a food delivery platform.

YOUR ROLE:
- Help customers find dishes they'll love
- Provide detailed info about ingredients, dietary tags, and prices
- Handle general greetings and conversation warmly

AVAILABLE TOOLS:
1. search_menu_catalog — semantic search over our live menu (RAG)

GUIDELINES:
- For greetings (with no food context), respond warmly and offer to help.
- For ANY food-related query, ALWAYS call search_menu_catalog first — never ask
  clarifying questions without searching first. Show results, then offer to refine.
- Use conversation history to understand context. If the customer previously
  asked about a cuisine or preference, carry that forward even if the latest
  message is vague (e.g. "anything works" after asking about non-veg → search
  for non-veg dishes).
- Respond in a warm, helpful tone. Mention dietary tags proactively.
- Keep responses concise — this is a voice assistant.
"""

def menu_agent(state: SnackStackState):
    existing_messages = state.get('menu_messages', [])
    history = state.get('messages', [])

    if not existing_messages:
        # First invocation, add system prompt
        all_messages = [
            SystemMessage(content=MENU_AGENT_PROMPT),
            *history,
            HumanMessage(content=state['user_query'])
        ]
    else:
        # Subsequent invocations, continue conversation
        all_messages = [
            *existing_messages,
        ]

    response = menu_agent_with_tool.invoke(all_messages)
    all_messages = [
        *all_messages, 
        response
    ]

    update = {
        "menu_message": all_messages,
    }

    if(not bool(getattr(response, 'tool_calls', None))):
        update["menu_reponse"] = response.content
        update['messages'] = [response]

    return update

def menu_agent_tool(state: SnackStackState):
    existing_messages = state.get('menu_messages', [])
    last_message = state.get('menu_messages', [])[-1]
    for tool_call in last_message.tool_calls:
        if tool_call.name != "search_menu_catelog":
            continue

        tool_response = menu_tools[0].invoke(tool_call.args)
        existing_messages.append(
            ToolMessage(
                content=tool_response,
                tool_call_id=tool_call.id,
            )
        )
           
    return {
        "menu_messages":existing_messages,
    }

def menu_agent_should_continue(state: SnackStackState) -> str :
    last_message = state.get('menu_messages', [])[-1]
    if(not bool(getattr(last_message, 'tool_calls', None))):
        return "synthesizer_node"

    return "menu_tools_node"
    
