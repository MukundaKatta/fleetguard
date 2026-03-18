"""Basic usage example for fleetguard."""
from src.core import Fleetguard

def main():
    instance = Fleetguard(config={"verbose": True})

    print("=== fleetguard Example ===\n")

    # Run primary operation
    result = instance.track(input="example data", mode="demo")
    print(f"Result: {result}")

    # Run multiple operations
    ops = ["track", "predict", "forecast]
    for op in ops:
        r = getattr(instance, op)(source="example")
        print(f"  {op}: {"✓" if r.get("ok") else "✗"}")

    # Check stats
    print(f"\nStats: {instance.get_stats()}")

if __name__ == "__main__":
    main()
