"""
Simple test module for function relationship demonstration.
"""

def helper_function(param1: int, param2: str) -> str:
    """A helper function that formats a string."""
    result = f"{param2}: {param1 * 2}"
    return result

def process_data(data: list) -> dict:
    """Processes a list of data and returns results."""
    result = {}
    for item in data:
        # Call helper_function
        formatted = helper_function(len(item), item)
        result[item] = formatted
    return result

def main():
    """Main entry point that calls process_data."""
    test_data = ["apple", "banana", "cherry"]
    results = process_data(test_data)
    print(results)
    return results

if __name__ == "__main__":
    main()