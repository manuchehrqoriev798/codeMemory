"""
Mathematical Utility Functions Library

This module provides various mathematical utility functions for 
common operations and calculations.
"""

import math
from typing import List, Union, Tuple, Optional

def factorial(n: int) -> int:
    """Calculate the factorial of a non-negative integer."""
    if n < 0:
        raise ValueError("Factorial is only defined for non-negative integers")
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n < 0:
        raise ValueError("Fibonacci sequence is only defined for non-negative integers")
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def gcd(a: int, b: int) -> int:
    """Calculate the greatest common divisor of two integers."""
    while b:
        a, b = b, a % b
    return a

def lcm(a: int, b: int) -> int:
    """Calculate the least common multiple of two integers."""
    return abs(a * b) // gcd(a, b)

def solve_quadratic(a: float, b: float, c: float) -> Tuple[Optional[float], Optional[float]]:
    """Solve a quadratic equation of the form ax² + bx + c = 0."""
    # Calculate the discriminant
    discriminant = b**2 - 4*a*c
    
    if discriminant < 0:
        return None, None  # No real solutions
    
    # Calculate the solutions using the quadratic formula
    x1 = (-b + math.sqrt(discriminant)) / (2*a)
    x2 = (-b - math.sqrt(discriminant)) / (2*a)
    
    return x1, x2

def main():
    """Demonstrate the use of the mathematical functions."""
    print("Factorial of 5:", factorial(5))
    print("10th Fibonacci number:", fibonacci(10))
    print("Is 17 prime?", is_prime(17))
    print("GCD of 48 and 18:", gcd(48, 18))
    print("LCM of 12 and 15:", lcm(12, 15))
    print("Solutions for x² - 5x + 6 = 0:", solve_quadratic(1, -5, 6))

if __name__ == "__main__":
    main() 