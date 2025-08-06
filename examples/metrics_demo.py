"""Demo script showing how to use the metrics module."""

from src.analyzer.metrics import calculate_metrics, MetricsCalculator


def main():
    """Demonstrate metrics calculation."""
    
    # Example 1: Simple, clean function
    simple_code = '''
def calculate_discount(price, discount_percent):
    """Calculate discounted price."""
    if discount_percent < 0 or discount_percent > 100:
        raise ValueError("Discount must be between 0 and 100")
    
    discount_amount = price * (discount_percent / 100)
    return price - discount_amount
'''
    
    print("=" * 60)
    print("SIMPLE FUNCTION METRICS")
    print("=" * 60)
    
    metrics = calculate_metrics(simple_code)
    print(f"Lines of Code: {metrics.lines_of_code}")
    print(f"Cyclomatic Complexity: {metrics.cyclomatic_complexity}")
    print(f"Maintainability Index: {metrics.maintainability_index:.2f}")
    print(f"Maintainability Grade: {metrics.maintainability_grade}")
    print(f"Quality Score: {metrics.complexity_score}/100")
    print(f"Risk Level: {metrics.risk_level}")
    
    # Example 2: Complex, problematic function
    complex_code = '''
def process_order(order_data, user, inventory, promotions, config):
    """Process an order with multiple conditions."""
    if not order_data:
        return None
    
    total = 0
    items_processed = []
    
    for item in order_data.get("items", []):
        if item.get("id") in inventory:
            stock = inventory[item["id"]]
            if stock["quantity"] >= item.get("quantity", 1):
                price = stock["price"]
                
                # Check for promotions
                if promotions:
                    for promo in promotions:
                        if promo["type"] == "percentage" and item["id"] in promo["items"]:
                            if user.get("membership") == "gold":
                                price = price * (1 - promo["discount"] * 1.2)
                            elif user.get("membership") == "silver":
                                price = price * (1 - promo["discount"] * 1.1)
                            else:
                                price = price * (1 - promo["discount"])
                        elif promo["type"] == "fixed" and item["id"] in promo["items"]:
                            price = max(0, price - promo["amount"])
                
                # Apply taxes
                if config.get("apply_tax"):
                    if user.get("location", {}).get("state") == "CA":
                        price = price * 1.0875
                    elif user.get("location", {}).get("state") == "NY":
                        price = price * 1.08
                    else:
                        price = price * 1.06
                
                total += price * item.get("quantity", 1)
                items_processed.append({
                    "id": item["id"],
                    "quantity": item.get("quantity", 1),
                    "price": price
                })
            else:
                if config.get("allow_backorder"):
                    items_processed.append({
                        "id": item["id"],
                        "quantity": item.get("quantity", 1),
                        "status": "backorder"
                    })
    
    if not items_processed:
        return None
    
    return {
        "total": total,
        "items": items_processed,
        "user_id": user.get("id"),
        "status": "processed"
    }
'''
    
    print("\n" + "=" * 60)
    print("COMPLEX FUNCTION METRICS")
    print("=" * 60)
    
    metrics = calculate_metrics(complex_code)
    print(f"Lines of Code: {metrics.lines_of_code}")
    print(f"Cyclomatic Complexity: {metrics.cyclomatic_complexity}")
    print(f"Maintainability Index: {metrics.maintainability_index:.2f}")
    print(f"Maintainability Grade: {metrics.maintainability_grade}")
    print(f"Quality Score: {metrics.complexity_score}/100")
    print(f"Risk Level: {metrics.risk_level}")
    
    print("\nFunction Details:")
    for func in metrics.functions:
        print(f"\n  Function: {func.name}")
        print(f"  - Complexity: {func.cyclomatic_complexity} ({func.complexity_rank})")
        print(f"  - Parameters: {func.parameter_count}")
        print(f"  - Lines: {func.lines_of_code}")
        print(f"  - Max Nesting: {func.max_nesting_depth}")
        print(f"  - Issues:")
        if func.is_complex:
            print("    * High complexity")
        if func.is_long:
            print("    * Too many lines")
        if func.has_many_params:
            print("    * Too many parameters")
        if func.is_deeply_nested:
            print("    * Deep nesting")
    
    print("\nAnti-patterns Detected:")
    for pattern in metrics.anti_patterns:
        print(f"  - {pattern['type']}: {pattern['message']} (line {pattern['line']})")
    
    # Example 3: Using the calculator directly
    print("\n" + "=" * 60)
    print("HALSTEAD METRICS EXAMPLE")
    print("=" * 60)
    
    math_code = '''
def quadratic_formula(a, b, c):
    """Solve quadratic equation ax^2 + bx + c = 0."""
    import math
    
    discriminant = b**2 - 4*a*c
    
    if discriminant < 0:
        return None  # No real solutions
    elif discriminant == 0:
        return -b / (2*a)  # One solution
    else:
        sqrt_discriminant = math.sqrt(discriminant)
        x1 = (-b + sqrt_discriminant) / (2*a)
        x2 = (-b - sqrt_discriminant) / (2*a)
        return (x1, x2)
'''
    
    calculator = MetricsCalculator()
    metrics = calculator.calculate_metrics(math_code)
    
    if metrics.halstead:
        print(f"Halstead Volume: {metrics.halstead.volume:.2f}")
        print(f"Halstead Difficulty: {metrics.halstead.difficulty:.2f}")
        print(f"Halstead Effort: {metrics.halstead.effort:.2f}")
        print(f"Estimated Time (seconds): {metrics.halstead.time:.2f}")
        print(f"Estimated Bugs: {metrics.halstead.bugs:.4f}")
        print(f"Vocabulary Size: {metrics.halstead.vocabulary}")
        print(f"Program Length: {metrics.halstead.length}")


if __name__ == "__main__":
    main()