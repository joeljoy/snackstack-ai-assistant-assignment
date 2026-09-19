from graph.state import SnackStackState
from graph.config import llm

from langchain_core.messages import SystemMessage, HumanMessage

SYNTHESIZER_SYSTEM_PROMPT = """""
    You are the response synthesizer for SnackStack, a voice-enabled food delivery assistant.
    Combine the responses from specialist agents into a single, coherent, friendly reply.
    Keep it concise and conversational — this will be spoken aloud by TTS, so avoid
    markdown formatting like **, bullet points, or numbered lists. Use natural speech
    phrasing instead.
    If only one agent responded, just clean up and present that response.
"""

def synthesizer(state:SnackStackState):
    menu_response = state.get('menu_reponse', "")
    order_response = state.get('order_reponse', "")

    combined_response = ""
    if menu_response:
       combined_response += f"Menu Agent Response: {menu_response} "
    if order_response:
       combined_response += f"Order Agent Response: {order_response} "

    if not combined_response:
        combined_response = "I'm sorry, I don't have any information to provide at the moment."

    final_response = llm.invoke(
        SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
        HumanMessage(content=f"User Query: {state['user_query']}\nCombined Agent Responses: {combined_response}")
    )

    return {
        "final_answer": final_response.content,
        "messages": [final_response],
    }
   