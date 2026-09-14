from langchain_core.tools import tool
from langchain_core.documents import Document
from graph.rag import menu_retriever

@tool
def get_menu_info(query:str) -> str:
    """
    Search the SnackStack menu catelog using semantic similarity.

    Use this to find dishes by name, cuisine, dietary preference or description.

    Args:
        query: Natural language search query, e.g. 'vegan pasta' or 'chicken starter' or 'non veg dishes'
    """
    docs:list[Document] = menu_retriever.invoke(query)
    if not docs:
            return "No matching menu items found."

    formatted = []
    formatted.append(f"Top {len(docs)} matches for {query}:")
    for d in docs:
          formatted.append(f"{d.page_content}")

    return "\n".join(formatted)

menu_tools = [get_menu_info]