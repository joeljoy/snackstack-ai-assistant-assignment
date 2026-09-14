from graph.data.order_catelog import ORDER_CATALOG


def find_order(key:str) -> tuple[str | None, dict | None]:
    """
    Search ORDER_CATALOG by order ID, tracking ID or customer's email.

    Returns (order_id, order_dic) or (None, None) if not found.
    """
    key = key.strip()
    upper_key = key.upper()
    # Order ID match
    if upper_key in ORDER_CATALOG:
        return upper_key, ORDER_CATALOG[upper_key]

    # Tracking ID or customer's email match
    for oid, order in ORDER_CATALOG.items():
        if order["tracking_id"].upper() == upper_key:
            return oid, order
        if order["customer_email"].upper() == upper_key:
            return oid, order

    return None, None

def format_order(order_id:str, order : dict) -> str:
    """Format order details into a readable string"""
    return (
        f"Order {order_id}\n"
        f"Item: {order['item_name']}\n"
        f"Customer: {order['customer_name']} ({order['customer_email']})\n"
        f"Status: {order['status']}\n"
        f"Price: {order['price']}\n"
        f"Order Date: {order['order_date']}\n"
        f"Estimated Delivery: {order['estimated_delivery']}\n"
        f"Tracking ID: {order['tracking_id']}"
    )

def get_order_info(lookup_key:str) -> str:
    """
    Look up the current of a customer's order.

    Args:
        lookup_key : An order ID(eg : ORD-202), a tracking ID(eg : SS203TRK), or
            customer email(eg : arjun@example.com).

    Returns:
        Formatted order details including status and estimated delivery date. If no matching
            order was found, an error message
    """
    order_id, order = find_order(lookup_key)
    if order is None:
        return f"No order found matching '{lookup_key}'."
    return format_order(order_id, order)

order_tools = [get_order_info]
