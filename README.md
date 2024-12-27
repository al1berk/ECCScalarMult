---

# ECCScalarMult

ECCScalarMult is a Python library that performs Scalar Multiplication operations under Elliptic Curve Cryptography (ECC). This project provides essential tools for developers working in the fields of cryptography and security.

## Table of Contents

- [Features](#features)
- [Usage](#usage)
- [Examples](#examples)

## Features

- **Scalar Multiplication:** Calculates the multiplication of a point on the curve by a specific scalar value.
- **Simple and Understandable:** Written using basic Python structures, making it easy to learn and customize.
- **Extensibility:** Designed with an open structure to add different elliptic curves and cryptographic operations.

## Usage

To use the ECCScalarMult library in your project, follow these steps:

1. **Import the Library:**

   ```python
   from algorithms import EllipticCurve, Point
   ```

2. **Define the Curve and Point:**

   ```python
   # Elliptic curve equation: y^2 = x^3 + ax + b
   curve = EllipticCurve(a=2, b=3)

   # A point on the curve
   point = Point(x=5, y=7)
   ```

3. **Perform Scalar Multiplication:**

   ```python
   k = 20  # Scalar value
   result = curve.scalar_multiplication(point, k)
   print(f"Resulting Point: ({result.x}, {result.y})")
   ```

## Examples

Below is a simple example of how to use the ECCScalarMult library:

```python
from algorithms import EllipticCurve, Point

# Define the elliptic curve
curve = EllipticCurve(a=-1, b=1)

# A point on the curve
point = Point(x=0, y=1)

# Perform scalar multiplication
k = 10
result = curve.scalar_multiplication(point, k)

print(f"The {k}th multiple of the point is: ({result.x}, {result.y})")
```

---
