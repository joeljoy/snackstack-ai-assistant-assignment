from graph.logger import setup_logger
from graph.tools.order_tools import get_order_info
from graph.tools.menu_tools import get_menu_info

logger = setup_logger(__name__)


def check_order_tools() -> None:
    logger.info("Checking order_tools.get_order_info")

    test_keys = [
        "ORD-202",          # order id
        "SS203TRK",         # tracking id
        "rahul@example.com",  # customer email
        "NOT-A-REAL-ORDER",   # missing order
    ]

    for key in test_keys:
        result = get_order_info(key)
        logger.info("Lookup '%s' ->\n%s", key, result)


def check_menu_tools() -> None:
    logger.info("Checking menu_tools.get_menu_info")

    test_queries = [
        "vegan pasta",
        "chicken starter",
        "non veg dishes",
    ]

    for query in test_queries:
        result = get_menu_info.invoke(query)
        logger.info("Query '%s' ->\n%s", query, result)


if __name__ == "__main__":
    check_order_tools()
    check_menu_tools()
