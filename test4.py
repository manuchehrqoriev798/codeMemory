"""
Additional test module with more function relationships.
"""

def helper1(x: int) -> int:
    """Helper function 1"""
    return x * 2

def helper2(y: str) -> str:
    """Helper function 2"""
    return y.upper()

def combine_helpers(data: dict) -> dict:
    """Function that calls both helpers"""
    result = {}
    for key, value in data.items():
        if isinstance(value, int):
            result[key] = helper1(value)
        elif isinstance(value, str):
            result[key] = helper2(value)
    return result

def process_collection(items: list) -> dict:
    """Process a collection of items"""
    data = {item: len(item) if isinstance(item, str) else item for item in items}
    return combine_helpers(data)
