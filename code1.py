"""
Comprehensive Geometric Shape Area Calculator

This module provides classes and functions for calculating the area of various
geometric shapes, demonstrating OOP principles, inheritance, and mathematical operations.
"""

import math
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Optional, Tuple
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for input validation errors."""
    pass

def validate_positive(value: Union[int, float], name: str) -> None:
    """Validate that a value is positive."""
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")

def validate_non_negative(value: Union[int, float], name: str) -> None:
    """Validate that a value is non-negative."""
    if value < 0:
        raise ValidationError(f"{name} must be non-negative, got {value}")

class Shape(ABC):
    """Abstract base class for all geometric shapes."""
    
    def __init__(self, name: str):
        """Initialize the shape with a name."""
        self.name = name
        logger.debug(f"Created {name} shape object")
    
    @abstractmethod
    def area(self) -> float:
        """Calculate the area of the shape."""
        pass
    
    @abstractmethod
    def perimeter(self) -> float:
        """Calculate the perimeter of the shape."""
        pass
    
    def __str__(self) -> str:
        """Return a string representation of the shape."""
        return f"{self.name} (Area: {self.area():.2f}, Perimeter: {self.perimeter():.2f})"
    
    def describe(self) -> Dict[str, Union[str, float]]:
        """Return a dictionary with the shape's properties."""
        return {
            "name": self.name,
            "area": self.area(),
            "perimeter": self.perimeter(),
            "type": self.__class__.__name__
        }

class Circle(Shape):
    """A circle shape defined by its radius."""
    
    def __init__(self, radius: float):
        """Initialize a circle with the given radius."""
        super().__init__("Circle")
        validate_positive(radius, "Radius")
        self.radius = radius
    
    def area(self) -> float:
        """Calculate the area of the circle using πr²."""
        return math.pi * self.radius ** 2
    
    def perimeter(self) -> float:
        """Calculate the circumference of the circle using 2πr."""
        return 2 * math.pi * self.radius
    
    def describe(self) -> Dict[str, Union[str, float]]:
        """Return a dictionary with the circle's properties."""
        result = super().describe()
        result["radius"] = self.radius
        return result
    
    @staticmethod
    def from_diameter(diameter: float) -> 'Circle':
        """Create a circle from its diameter."""
        validate_positive(diameter, "Diameter")
        return Circle(diameter / 2)
    
    @staticmethod
    def from_circumference(circumference: float) -> 'Circle':
        """Create a circle from its circumference."""
        validate_positive(circumference, "Circumference")
        return Circle(circumference / (2 * math.pi))

class Rectangle(Shape):
    """A rectangle shape defined by its width and height."""
    
    def __init__(self, width: float, height: float):
        """Initialize a rectangle with the given width and height."""
        super().__init__("Rectangle")
        validate_positive(width, "Width")
        validate_positive(height, "Height")
        self.width = width
        self.height = height
    
    def area(self) -> float:
        """Calculate the area of the rectangle using width × height."""
        return self.width * self.height
    
    def perimeter(self) -> float:
        """Calculate the perimeter of the rectangle using 2(width + height)."""
        return 2 * (self.width + self.height)
    
    def is_square(self) -> bool:
        """Check if the rectangle is a square."""
        return math.isclose(self.width, self.height)
    
    def describe(self) -> Dict[str, Union[str, float, bool]]:
        """Return a dictionary with the rectangle's properties."""
        result = super().describe()
        result["width"] = self.width
        result["height"] = self.height
        result["is_square"] = self.is_square()
        return result
    
    @staticmethod
    def create_square(side_length: float) -> 'Rectangle':
        """Create a square (special case of rectangle)."""
        return Rectangle(side_length, side_length)

class Square(Rectangle):
    """A square shape, which is a special case of a rectangle with equal sides."""
    
    def __init__(self, side_length: float):
        """Initialize a square with the given side length."""
        super().__init__(side_length, side_length)
        self.name = "Square"  # Override the name from Rectangle
    
    @property
    def side_length(self) -> float:
        """Get the side length of the square."""
        return self.width
    
    @side_length.setter
    def side_length(self, value: float) -> None:
        """Set the side length of the square."""
        validate_positive(value, "Side length")
        self.width = value
        self.height = value

class Triangle(Shape):
    """A triangle shape defined by its three sides."""
    
    def __init__(self, a: float, b: float, c: float):
        """Initialize a triangle with the given sides."""
        super().__init__("Triangle")
        validate_positive(a, "Side a")
        validate_positive(b, "Side b")
        validate_positive(c, "Side c")
        
        # Check if the sides can form a triangle
        if a + b <= c or a + c <= b or b + c <= a:
            raise ValidationError(f"Sides {a}, {b}, {c} cannot form a triangle")
            
        self.a = a
        self.b = b
        self.c = c
    
    def area(self) -> float:
        """Calculate the area of the triangle using Heron's formula."""
        # Semi-perimeter
        s = self.perimeter() / 2
        # Heron's formula
        return math.sqrt(s * (s - self.a) * (s - self.b) * (s - self.c))
    
    def perimeter(self) -> float:
        """Calculate the perimeter of the triangle as the sum of its sides."""
        return self.a + self.b + self.c
    
    def is_equilateral(self) -> bool:
        """Check if the triangle is equilateral (all sides equal)."""
        return (math.isclose(self.a, self.b) and 
                math.isclose(self.b, self.c))
    
    def is_isosceles(self) -> bool:
        """Check if the triangle is isosceles (at least two sides equal)."""
        return (math.isclose(self.a, self.b) or 
                math.isclose(self.b, self.c) or 
                math.isclose(self.a, self.c))
    
    def is_right(self) -> bool:
        """Check if the triangle is right (has a 90-degree angle)."""
        sides = sorted([self.a, self.b, self.c])
        return math.isclose(sides[0]**2 + sides[1]**2, sides[2]**2)
    
    def describe(self) -> Dict[str, Union[str, float, bool]]:
        """Return a dictionary with the triangle's properties."""
        result = super().describe()
        result.update({
            "side_a": self.a,
            "side_b": self.b,
            "side_c": self.c,
            "is_equilateral": self.is_equilateral(),
            "is_isosceles": self.is_isosceles(),
            "is_right": self.is_right()
        })
        return result
    
    @classmethod
    def equilateral(cls, side: float) -> 'Triangle':
        """Create an equilateral triangle with the given side length."""
        return cls(side, side, side)
    
    @classmethod
    def right(cls, a: float, b: float) -> 'Triangle':
        """Create a right triangle with the two shorter sides."""
        validate_positive(a, "Side a")
        validate_positive(b, "Side b")
        # Calculate the hypotenuse using the Pythagorean theorem
        c = math.sqrt(a**2 + b**2)
        return cls(a, b, c)

class RegularPolygon(Shape):
    """A regular polygon with n equal sides."""
    
    def __init__(self, num_sides: int, side_length: float):
        """Initialize a regular polygon with the given number of sides and side length."""
        super().__init__(f"Regular {num_sides}-gon")
        
        if num_sides < 3:
            raise ValidationError(f"Number of sides must be at least 3, got {num_sides}")
            
        validate_positive(side_length, "Side length")
        
        self.num_sides = num_sides
        self.side_length = side_length
    
    def area(self) -> float:
        """Calculate the area of the regular polygon."""
        # Formula: (n × s²) / (4 × tan(π/n))
        return (self.num_sides * self.side_length**2) / (4 * math.tan(math.pi / self.num_sides))
    
    def perimeter(self) -> float:
        """Calculate the perimeter of the regular polygon."""
        return self.num_sides * self.side_length
    
    def get_interior_angle(self) -> float:
        """Calculate the interior angle of the regular polygon."""
        return ((self.num_sides - 2) * 180) / self.num_sides
    
    def describe(self) -> Dict[str, Union[str, float, int]]:
        """Return a dictionary with the regular polygon's properties."""
        result = super().describe()
        result.update({
            "number_of_sides": self.num_sides,
            "side_length": self.side_length,
            "interior_angle": self.get_interior_angle()
        })
        return result

class Ellipse(Shape):
    """An ellipse shape defined by its semi-major and semi-minor axes."""
    
    def __init__(self, a: float, b: float):
        """Initialize an ellipse with semi-major axis a and semi-minor axis b."""
        super().__init__("Ellipse")
        validate_positive(a, "Semi-major axis")
        validate_positive(b, "Semi-minor axis")
        
        # Ensure a is the semi-major axis
        if a < b:
            a, b = b, a
            
        self.a = a  # Semi-major axis
        self.b = b  # Semi-minor axis
    
    def area(self) -> float:
        """Calculate the area of the ellipse using πab."""
        return math.pi * self.a * self.b
    
    def perimeter(self) -> float:
        """Calculate the approximate perimeter of the ellipse using Ramanujan's formula."""
        # Ramanujan's approximation
        h = ((self.a - self.b) / (self.a + self.b))**2
        return math.pi * (self.a + self.b) * (1 + (3*h)/(10 + math.sqrt(4 - 3*h)))
    
    def eccentricity(self) -> float:
        """Calculate the eccentricity of the ellipse."""
        return math.sqrt(1 - (self.b/self.a)**2)
    
    def describe(self) -> Dict[str, Union[str, float]]:
        """Return a dictionary with the ellipse's properties."""
        result = super().describe()
        result.update({
            "semi_major_axis": self.a,
            "semi_minor_axis": self.b,
            "eccentricity": self.eccentricity()
        })
        return result

class ShapeCollection:
    """A collection of shapes with utility methods for analysis."""
    
    def __init__(self, shapes: Optional[List[Shape]] = None):
        """Initialize a collection of shapes."""
        self.shapes = shapes or []
    
    def add_shape(self, shape: Shape) -> None:
        """Add a shape to the collection."""
        self.shapes.append(shape)
    
    def total_area(self) -> float:
        """Calculate the sum of all shape areas in the collection."""
        return sum(shape.area() for shape in self.shapes)
    
    def total_perimeter(self) -> float:
        """Calculate the sum of all shape perimeters in the collection."""
        return sum(shape.perimeter() for shape in self.shapes)
    
    def count_by_type(self) -> Dict[str, int]:
        """Count the number of shapes by their type."""
        counts = {}
        for shape in self.shapes:
            shape_type = shape.__class__.__name__
            counts[shape_type] = counts.get(shape_type, 0) + 1
        return counts
    
    def find_largest_area(self) -> Optional[Shape]:
        """Find the shape with the largest area."""
        if not self.shapes:
            return None
        return max(self.shapes, key=lambda s: s.area())
    
    def find_smallest_area(self) -> Optional[Shape]:
        """Find the shape with the smallest area."""
        if not self.shapes:
            return None
        return min(self.shapes, key=lambda s: s.area())
    
    def sort_by_area(self) -> List[Shape]:
        """Sort the shapes by their area in ascending order."""
        return sorted(self.shapes, key=lambda s: s.area())
    
    def get_areas_dict(self) -> Dict[str, float]:
        """Return a dictionary mapping shape descriptions to their areas."""
        return {f"{shape.name}_{i}": shape.area() 
                for i, shape in enumerate(self.shapes)}
    
    def get_statistics(self) -> Dict[str, float]:
        """Calculate statistics on the areas of shapes in the collection."""
        areas = [shape.area() for shape in self.shapes]
        if not areas:
            return {"count": 0}
        
        return {
            "count": len(areas),
            "min": min(areas),
            "max": max(areas),
            "mean": sum(areas) / len(areas),
            "median": sorted(areas)[len(areas) // 2],
            "total": sum(areas)
        }

def find_shape_with_closest_area(shapes: List[Shape], target_area: float) -> Optional[Shape]:
    """Find the shape with the area closest to the target area."""
    if not shapes:
        return None
    return min(shapes, key=lambda s: abs(s.area() - target_area))

def generate_random_shape() -> Shape:
    """Generate a random shape with random dimensions."""
    import random
    
    shape_type = random.choice(["Circle", "Rectangle", "Square", "Triangle", "RegularPolygon", "Ellipse"])
    
    if shape_type == "Circle":
        radius = random.uniform(1, 10)
        return Circle(radius)
    
    elif shape_type == "Rectangle":
        width = random.uniform(1, 10)
        height = random.uniform(1, 10)
        return Rectangle(width, height)
    
    elif shape_type == "Square":
        side = random.uniform(1, 10)
        return Square(side)
    
    elif shape_type == "Triangle":
        # Generate valid triangle sides
        while True:
            a = random.uniform(1, 10)
            b = random.uniform(1, 10)
            c = random.uniform(max(a, b) - min(a, b) + 0.01, a + b - 0.01)
            try:
                return Triangle(a, b, c)
            except ValidationError:
                continue
    
    elif shape_type == "RegularPolygon":
        num_sides = random.randint(3, 12)
        side_length = random.uniform(1, 10)
        return RegularPolygon(num_sides, side_length)
    
    else:  # Ellipse
        a = random.uniform(1, 10)
        b = random.uniform(1, a)
        return Ellipse(a, b)

def compare_shapes(shape1: Shape, shape2: Shape) -> Dict[str, Union[str, float, bool]]:
    """Compare two shapes and return a dictionary with the comparison results."""
    return {
        "shape1": shape1.name,
        "shape2": shape2.name,
        "area1": shape1.area(),
        "area2": shape2.area(),
        "perimeter1": shape1.perimeter(),
        "perimeter2": shape2.perimeter(),
        "area_difference": abs(shape1.area() - shape2.area()),
        "perimeter_difference": abs(shape1.perimeter() - shape2.perimeter()),
        "area_ratio": shape1.area() / shape2.area() if shape2.area() != 0 else float('inf'),
        "perimeter_ratio": shape1.perimeter() / shape2.perimeter() if shape2.perimeter() != 0 else float('inf')
    }

def create_shape_by_name(name: str, **kwargs) -> Shape:
    """Create a shape by its name and parameters."""
    shapes = {
        "circle": Circle,
        "rectangle": Rectangle,
        "square": Square,
        "triangle": Triangle,
        "regular_polygon": RegularPolygon,
        "ellipse": Ellipse
    }
    
    shape_class = shapes.get(name.lower())
    if not shape_class:
        raise ValueError(f"Unknown shape: {name}")
    
    try:
        return shape_class(**kwargs)
    except TypeError as e:
        raise TypeError(f"Invalid parameters for {name}: {e}")

def demo():
    """Demonstrate the use of the shape classes."""
    # Create a collection of shapes
    collection = ShapeCollection()
    
    # Add various shapes
    collection.add_shape(Circle(5))
    collection.add_shape(Rectangle(4, 6))
    collection.add_shape(Square(4))
    collection.add_shape(Triangle(3, 4, 5))
    collection.add_shape(RegularPolygon(6, 3))
    collection.add_shape(Ellipse(5, 3))
    
    # Add some random shapes
    for _ in range(5):
        collection.add_shape(generate_random_shape())
    
    # Print information about each shape
    print("Shapes in the collection:")
    for i, shape in enumerate(collection.shapes):
        print(f"{i + 1}. {shape}")
    
    # Print statistics
    print("\nStatistics:")
    stats = collection.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
    
    # Find shapes with extreme areas
    largest = collection.find_largest_area()
    smallest = collection.find_smallest_area()
    print(f"\nLargest area: {largest}")
    print(f"Smallest area: {smallest}")
    
    # Count shapes by type
    print("\nShape counts:")
    counts = collection.count_by_type()
    for shape_type, count in counts.items():
        print(f"{shape_type}: {count}")
    
    # Find a shape with area closest to 30
    target_area = 30
    closest = find_shape_with_closest_area(collection.shapes, target_area)
    print(f"\nShape with area closest to {target_area}: {closest}")
    
    # Compare two shapes
    shape1 = collection.shapes[0]
    shape2 = collection.shapes[1]
    comparison = compare_shapes(shape1, shape2)
    print(f"\nComparison between {shape1.name} and {shape2.name}:")
    for key, value in comparison.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    demo()
