"""CLI for fleetguard."""
import sys, json, argparse
from .core import Fleetguard

def main():
    parser = argparse.ArgumentParser(description="FleetGuard — Fleet Maintenance Predictor. Predictive maintenance for vehicle fleets using telematics data.")
    parser.add_argument("command", nargs="?", default="status", choices=["status", "run", "info"])
    parser.add_argument("--input", "-i", default="")
    args = parser.parse_args()
    instance = Fleetguard()
    if args.command == "status":
        print(json.dumps(instance.get_stats(), indent=2))
    elif args.command == "run":
        print(json.dumps(instance.track(input=args.input or "test"), indent=2, default=str))
    elif args.command == "info":
        print(f"fleetguard v0.1.0 — FleetGuard — Fleet Maintenance Predictor. Predictive maintenance for vehicle fleets using telematics data.")

if __name__ == "__main__":
    main()
