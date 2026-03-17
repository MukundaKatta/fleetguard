"""CLI interface for FleetGuard."""

from __future__ import annotations

import click
from rich.console import Console

from fleetguard.fleet.cost import CostAnalyzer
from fleetguard.fleet.telematics import TelematicsProcessor
from fleetguard.predictor.components import ComponentDatabase
from fleetguard.predictor.model import FailurePredictor
from fleetguard.predictor.scheduler import MaintenanceScheduler
from fleetguard.report import (
    print_cost_analysis,
    print_fleet_summary,
    print_predictions,
    print_schedule,
)
from fleetguard.simulator import FleetSimulator

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="fleetguard")
def cli() -> None:
    """FleetGuard - Fleet Maintenance Predictor.

    Predict component failures and optimize maintenance scheduling
    for your vehicle fleet.
    """


@cli.command()
@click.option("--vehicles", "-n", default=10, help="Number of vehicles to simulate.")
@click.option("--seed", default=42, help="Random seed for reproducibility.")
def demo(vehicles: int, seed: int) -> None:
    """Run a full demonstration with simulated fleet data."""
    console.print("[bold cyan]FleetGuard Demo[/]")
    console.print("=" * 60)

    # Setup
    component_db = ComponentDatabase()
    simulator = FleetSimulator(component_db=component_db, seed=seed)
    predictor = FailurePredictor(component_db=component_db)
    scheduler = MaintenanceScheduler(predictor=predictor, component_db=component_db)
    cost_analyzer = CostAnalyzer(component_db=component_db)
    telematics_processor = TelematicsProcessor()

    # Generate fleet
    console.print(f"\n[bold]Generating fleet of {vehicles} vehicles...[/]")
    fleet_manager = simulator.generate_fleet(num_vehicles=vehicles)
    fleet = fleet_manager.get_all()
    print_fleet_summary(fleet, console)

    # Predictions for first vehicle
    vehicle = fleet[0]
    console.print(f"\n[bold]Failure Predictions for {vehicle.vehicle_id}[/]")
    telematics = simulator.generate_telematics(vehicle)
    features = telematics_processor.process(telematics)
    if features:
        health = telematics_processor.compute_health_scores(features)
        console.print(f"  Health scores: {health}")

    predictions = predictor.predict_vehicle(vehicle, telematics)
    print_predictions(predictions, vehicle.vehicle_id, console=console)

    # Fleet schedule
    console.print("\n[bold]Fleet Maintenance Schedule (90-day horizon)[/]")
    all_predictions = {}
    for v in fleet:
        tel = simulator.generate_telematics(v)
        all_predictions[v.vehicle_id] = predictor.predict_vehicle(v, tel)

    schedule = scheduler.schedule_fleet(fleet, all_predictions)
    print_schedule(schedule, console)

    # Cost analysis
    console.print("\n[bold]Cost Analysis[/]")
    analyses = cost_analyzer.analyze_fleet(fleet)
    print_cost_analysis(analyses, console)


@cli.command()
@click.option("--vehicles", "-n", default=10, help="Number of vehicles.")
@click.option("--seed", default=42, help="Random seed.")
def predict(vehicles: int, seed: int) -> None:
    """Run failure predictions for the fleet."""
    component_db = ComponentDatabase()
    simulator = FleetSimulator(component_db=component_db, seed=seed)
    predictor = FailurePredictor(component_db=component_db)

    fleet_manager = simulator.generate_fleet(num_vehicles=vehicles)
    fleet = fleet_manager.get_all()

    for vehicle in fleet:
        telematics = simulator.generate_telematics(vehicle)
        predictions = predictor.predict_vehicle(vehicle, telematics)
        print_predictions(predictions, vehicle.vehicle_id, top_n=5, console=console)
        console.print()


@cli.command()
@click.option("--vehicles", "-n", default=10, help="Number of vehicles.")
@click.option("--horizon", default=90, help="Planning horizon in days.")
@click.option("--seed", default=42, help="Random seed.")
def schedule(vehicles: int, horizon: int, seed: int) -> None:
    """Generate optimized maintenance schedule."""
    component_db = ComponentDatabase()
    simulator = FleetSimulator(component_db=component_db, seed=seed)
    predictor = FailurePredictor(component_db=component_db)
    scheduler = MaintenanceScheduler(predictor=predictor, component_db=component_db)

    fleet_manager = simulator.generate_fleet(num_vehicles=vehicles)
    fleet = fleet_manager.get_all()

    all_predictions = {}
    for v in fleet:
        tel = simulator.generate_telematics(v)
        all_predictions[v.vehicle_id] = predictor.predict_vehicle(v, tel)

    result = scheduler.schedule_fleet(fleet, all_predictions, horizon_days=horizon)
    print_schedule(result, console)


@cli.command()
@click.option("--vehicles", "-n", default=10, help="Number of vehicles.")
@click.option("--seed", default=42, help="Random seed.")
def costs(vehicles: int, seed: int) -> None:
    """Analyze maintenance costs vs. replacement."""
    component_db = ComponentDatabase()
    simulator = FleetSimulator(component_db=component_db, seed=seed)
    cost_analyzer = CostAnalyzer(component_db=component_db)

    fleet_manager = simulator.generate_fleet(num_vehicles=vehicles)
    fleet = fleet_manager.get_all()

    analyses = cost_analyzer.analyze_fleet(fleet)
    print_cost_analysis(analyses, console)


@cli.command()
def components() -> None:
    """List all tracked components and their parameters."""
    from rich.table import Table

    db = ComponentDatabase()
    table = Table(title="Component Database", show_lines=True)
    table.add_column("Name", style="cyan")
    table.add_column("Category")
    table.add_column("Mean Life (mi)", justify="right")
    table.add_column("Weibull Shape", justify="right")
    table.add_column("Weibull Scale", justify="right")
    table.add_column("Repair Cost", justify="right")
    table.add_column("Severity")
    table.add_column("PM Interval (mi)", justify="right")

    for comp in db:
        table.add_row(
            comp.name,
            comp.category.value,
            f"{comp.mean_life_miles:,.0f}",
            f"{comp.weibull_shape:.1f}",
            f"{comp.weibull_scale:,.0f}",
            f"${comp.total_repair_cost:,.0f}",
            comp.severity.value,
            f"{comp.preventive_maintenance_interval_miles:,.0f}"
            if comp.preventive_maintenance_interval_miles
            else "-",
        )

    console.print(table)
    console.print(f"\n[bold]Total components:[/] {len(db)}")


if __name__ == "__main__":
    cli()
