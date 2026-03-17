"""Rich console reporting for FleetGuard."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fleetguard.fleet.cost import CostAnalysis
from fleetguard.models import FailurePrediction, FailureSeverity, Vehicle
from fleetguard.predictor.scheduler import MaintenanceSchedule


SEVERITY_COLORS = {
    FailureSeverity.LOW: "green",
    FailureSeverity.MEDIUM: "yellow",
    FailureSeverity.HIGH: "red",
    FailureSeverity.CRITICAL: "bold red",
}


def print_fleet_summary(
    vehicles: list[Vehicle],
    console: Console | None = None,
) -> None:
    """Print a summary of the fleet."""
    console = console or Console()

    table = Table(title="Fleet Summary", show_lines=True)
    table.add_column("Vehicle ID", style="cyan")
    table.add_column("Make/Model")
    table.add_column("Year", justify="right")
    table.add_column("Mileage", justify="right")
    table.add_column("Age (months)", justify="right")
    table.add_column("Avg Daily Mi", justify="right")
    table.add_column("Service Records", justify="right")

    for v in vehicles:
        table.add_row(
            v.vehicle_id,
            f"{v.make} {v.model}",
            str(v.year),
            f"{v.current_mileage:,.0f}",
            f"{v.age_months:.0f}",
            f"{v.avg_daily_miles:.0f}",
            str(len(v.service_history)),
        )

    console.print(table)
    console.print(
        f"\n[bold]Total vehicles:[/] {len(vehicles)}  |  "
        f"[bold]Avg mileage:[/] {sum(v.current_mileage for v in vehicles) / max(len(vehicles), 1):,.0f}  |  "
        f"[bold]Avg age:[/] {sum(v.age_months for v in vehicles) / max(len(vehicles), 1):.0f} months"
    )


def print_predictions(
    predictions: list[FailurePrediction],
    vehicle_id: str | None = None,
    top_n: int = 10,
    console: Console | None = None,
) -> None:
    """Print failure predictions."""
    console = console or Console()

    title = "Failure Predictions"
    if vehicle_id:
        title += f" - {vehicle_id}"

    table = Table(title=title, show_lines=True)
    table.add_column("Component", style="cyan")
    table.add_column("Risk", justify="right")
    table.add_column("Survival", justify="right")
    table.add_column("Severity")
    table.add_column("Fail @ Miles", justify="right")
    table.add_column("Fail Date", justify="right")
    table.add_column("Service Date", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Confidence", justify="right")

    for pred in predictions[:top_n]:
        color = SEVERITY_COLORS.get(pred.severity, "white")
        risk_color = "red" if pred.risk_score > 0.7 else ("yellow" if pred.risk_score > 0.4 else "green")

        table.add_row(
            pred.component_name,
            f"[{risk_color}]{pred.risk_score:.1%}[/]",
            f"{pred.survival_probability:.1%}",
            f"[{color}]{pred.severity.value}[/]",
            f"{pred.predicted_failure_mileage:,.0f}",
            str(pred.predicted_failure_date),
            str(pred.recommended_service_date),
            f"${pred.estimated_cost:,.0f}",
            f"{pred.confidence:.0%}",
        )

    console.print(table)


def print_schedule(
    schedule: MaintenanceSchedule,
    console: Console | None = None,
) -> None:
    """Print the maintenance schedule."""
    console = console or Console()

    table = Table(title="Maintenance Schedule", show_lines=True)
    table.add_column("Priority", justify="center")
    table.add_column("Vehicle", style="cyan")
    table.add_column("Date")
    table.add_column("Components")
    table.add_column("Est. Cost", justify="right")
    table.add_column("Downtime (hrs)", justify="right")
    table.add_column("Reason")

    for svc in schedule.services:
        table.add_row(
            str(svc.priority),
            svc.vehicle_id,
            str(svc.service_date),
            ", ".join(svc.components[:3]) + ("..." if len(svc.components) > 3 else ""),
            f"${svc.estimated_cost:,.0f}",
            f"{svc.estimated_downtime_hours:.1f}",
            svc.reason[:60] + ("..." if len(svc.reason) > 60 else ""),
        )

    console.print(table)
    console.print(
        Panel(
            f"[bold]Total cost:[/] ${schedule.total_cost:,.0f}  |  "
            f"[bold]Total downtime:[/] {schedule.total_downtime_hours:.0f} hrs  |  "
            f"[bold]Vehicles affected:[/] {schedule.vehicles_affected}",
            title="Schedule Summary",
        )
    )


def print_cost_analysis(
    analyses: list[CostAnalysis],
    console: Console | None = None,
) -> None:
    """Print cost analysis results."""
    console = console or Console()

    table = Table(title="Cost Analysis - Maintenance vs. Replacement", show_lines=True)
    table.add_column("Vehicle", style="cyan")
    table.add_column("Maint. to Date", justify="right")
    table.add_column("Proj. Annual", justify="right")
    table.add_column("Vehicle Value", justify="right")
    table.add_column("Cost/Mile", justify="right")
    table.add_column("Replace?", justify="center")
    table.add_column("Recommendation")

    for a in analyses:
        replace_style = "[bold red]YES[/]" if a.replacement_threshold_reached else "[green]NO[/]"
        table.add_row(
            a.vehicle_id,
            f"${a.total_maintenance_cost_to_date:,.0f}",
            f"${a.projected_annual_maintenance:,.0f}",
            f"${a.vehicle_current_value:,.0f}",
            f"${a.cost_per_mile:.3f}",
            replace_style,
            a.recommendation[:50] + ("..." if len(a.recommendation) > 50 else ""),
        )

    console.print(table)
