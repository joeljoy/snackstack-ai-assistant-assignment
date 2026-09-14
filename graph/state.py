from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class State(TypedDict):
    user_query : str
    route:list[str]
    messages:Annotated[list[AnyMessage], add_messages]
    menu_reponse:str
    order_reponse:str
    final_answer:str