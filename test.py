def example_function(x, y):
    """
    A simple function that performs addition and returns the result.
    """
    result = x + y
    print(f"The sum of {x} and {y} is {result}")
    return result

def calculate_average(numbers):
    """Calculate the average of a list of numbers."""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

def reverse_string(text):
    """Reverse a string."""
    return text[::-1]

def is_palindrome(text):
    """Check if a string is a palindrome."""
    cleaned_text = ''.join(c.lower() for c in text if c.isalnum())
    return cleaned_text == cleaned_text[::-1]

def fibonacci(n):
    """Return the nth Fibonacci number."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

def count_words(text):
    """Count the number of words in a text."""
    if not text:
        return 0
    return len(text.split())

def find_max(numbers):
    """Find the maximum number in a list."""
    if not numbers:
        return None
    return max(numbers)

def capitalize_words(text):
    """Capitalize the first letter of each word in a text."""
    return ' '.join(word.capitalize() for word in text.split())

def filter_even_numbers(numbers):
    """Filter out even numbers from a list."""
    return [num for num in numbers if num % 2 == 0]

def merge_dictionaries(dict1, dict2):
    """Merge two dictionaries into one."""
    result = dict1.copy()
    result.update(dict2)
    return result

# You can add other functions or code here if needed