"""Maintenance scheduler that optimizes service timing to minimize downtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np

from fleetguard.models import FailurePrediction, Vehicle
from fleetguard.predictor.components import ComponentDatabase
from fleetguard.predictor.model import FailurePredictor


@dataclass
class ScheduledService:
    """A scheduled maintenance service."""

    vehicle_id: str
    service_date: date
    components: list[str]
    estimated_cost: float
    estimated_downtime_hours: float
    priority: int  # 1 = highest
    reason: str


@dataclass
class MaintenanceSchedule:
    """Complete maintenance schedule for the fleet."""

    services: list[ScheduledService] = field(default_factory=list)
    total_cost: float = 0.0
    total_downtime_hours: float = 0.0
    vehicles_affected: int = 0

    def sort_by_priority(self) -> None:
        self.services.sort(key=lambda s: (s.priority, s.service_date))

    def sort_by_date(self) -> None:
        self.services.sort(key=lambda s: s.service_date)


class MaintenanceScheduler:
    """Optimize maintenance scheduling to minimize fleet downtime.

    Strategy:
    1. Group nearby services to reduce total shop visits.
    2. Prioritize critical/high-severity components.
    3. Balance workload across available service windows.
    4. Consider cost trade-offs between early preventive and reactive maintenance.
    """

    def __init__(
        self,
        predictor: Optional[FailurePredictor] = None,
        component_db: Optional[ComponentDatabase] = None,
        grouping_window_days: int = 14,
        max_concurrent_vehicles: int = 3,
        cost_of_downtime_per_hour: float = 150.0,
    ) -> None:
        self.component_db = component_db or ComponentDatabase()
        self.predictor = predictor or FailurePredictor(self.component_db)
        self.grouping_window_days = grouping_window_days
        self.max_concurrent_vehicles = max_concurrent_vehicles
        self.cost_of_downtime_per_hour = cost_of_downtime_per_hour

    def schedule_vehicle(
        self,
        vehicle: Vehicle,
        predictions: Optional[list[FailurePrediction]] = None,
        horizon_days: int = 90,
    ) -> list[ScheduledService]:
        """Create an optimized maintenance schedule for a single vehicle.

        Groups services that fall within the grouping window to minimize
        the number of shop visits.
        """
        if predictions is None:
            predictions = self.predictor.predict_vehicle(vehicle)

        cutoff = date.today() + timedelta(days=horizon_days)

        # Filter predictions within the horizon
        actionable = [
            p for p in predictions if p.recommended_service_date <= cutoff
        ]

        if not actionable:
            return []

        # Sort by recommended date
        actionable.sort(key=lambda p: p.recommended_service_date)

        # Group services within the grouping window
        groups: list[list[FailurePrediction]] = []
        current_group: list[FailurePrediction] = [actionable[0]]

        for pred in actionable[1:]:
            anchor_date = current_group[0].recommended_service_date
            if (pred.recommended_service_date - anchor_date).days <= self.grouping_window_days:
                current_group.append(pred)
            else:
                groups.append(current_group)
                current_group = [pred]
        groups.append(current_group)

        # Convert groups to scheduled services
        services = []
        for priority, group in enumerate(groups, start=1):
            total_cost = sum(p.estimated_cost for p in group)
            components = [p.component_name for p in group]

            # Downtime: overlapping work reduces total time
            downtimes = []
            for p in group:
                comp = self.component_db.get(p.component_name)
                if comp:
                    downtimes.append(comp.downtime_hours)
                else:
                    downtimes.append(2.0)
            # Parallel work: total downtime = max + 30% of remaining
            downtimes.sort(reverse=True)
            if downtimes:
                total_downtime = downtimes[0] + 0.3 * sum(downtimes[1:])
            else:
                total_downtime = 0.0

            # Service date = earliest recommended date in the group
            service_date = min(p.recommended_service_date for p in group)

            max_severity = max(p.severity.value for p in group)
            reasons = []
            for p in group:
                reasons.append(
                    f"{p.component_name} (risk={p.risk_score:.0%})"
                )
            reason = "Service: " + ", ".join(reasons)

            services.append(
                ScheduledService(
                    vehicle_id=vehicle.vehicle_id,
                    service_date=service_date,
                    components=components,
                    estimated_cost=round(total_cost, 2),
                    estimated_downtime_hours=round(total_downtime, 1),
                    priority=priority,
                    reason=reason,
                )
            )

        return services

    def schedule_fleet(
        self,
        vehicles: list[Vehicle],
        predictions: Optional[dict[str, list[FailurePrediction]]] = None,
        horizon_days: int = 90,
    ) -> MaintenanceSchedule:
        """Create an optimized maintenance schedule for the entire fleet.

        Balances workload to avoid exceeding max_concurrent_vehicles on
        any given day.
        """
        all_services: list[ScheduledService] = []

        for vehicle in vehicles:
            vehicle_preds = (
                predictions.get(vehicle.vehicle_id) if predictions else None
            )
            vehicle_services = self.schedule_vehicle(
                vehicle, vehicle_preds, horizon_days
            )
            all_services.extend(vehicle_services)

        # Load-balance: shift services if too many on the same day
        all_services.sort(key=lambda s: (s.service_date, s.priority))
        all_services = self._balance_workload(all_services)

        total_cost = sum(s.estimated_cost for s in all_services)
        total_downtime = sum(s.estimated_downtime_hours for s in all_services)
        vehicles_affected = len(set(s.vehicle_id for s in all_services))

        schedule = MaintenanceSchedule(
            services=all_services,
            total_cost=round(total_cost, 2),
            total_downtime_hours=round(total_downtime, 1),
            vehicles_affected=vehicles_affected,
        )
        schedule.sort_by_date()
        return schedule

    def _balance_workload(
        self, services: list[ScheduledService]
    ) -> list[ScheduledService]:
        """Shift lower-priority services to balance daily workload."""
        if not services:
            return services

        day_counts: dict[date, int] = {}
        balanced: list[ScheduledService] = []

        for service in services:
            current_date = service.service_date
            count = day_counts.get(current_date, 0)

            if count >= self.max_concurrent_vehicles:
                # Find next available day
                shift = 1
                while day_counts.get(current_date + timedelta(days=shift), 0) >= self.max_concurrent_vehicles:
                    shift += 1
                    if shift > 30:  # safety valve
                        break
                new_date = current_date + timedelta(days=shift)
                service = ScheduledService(
                    vehicle_id=service.vehicle_id,
                    service_date=new_date,
                    components=service.components,
                    estimated_cost=service.estimated_cost,
                    estimated_downtime_hours=service.estimated_downtime_hours,
                    priority=service.priority,
                    reason=service.reason,
                )
                day_counts[new_date] = day_counts.get(new_date, 0) + 1
            else:
                day_counts[current_date] = count + 1

            balanced.append(service)

        return balanced

    def compute_cost_benefit(
        self,
        prediction: FailurePrediction,
    ) -> dict[str, float]:
        """Compute cost-benefit analysis of preventive vs. reactive maintenance.

        Returns a dict with cost comparison and recommendation score.
        """
        preventive_cost = prediction.estimated_cost
        # Include downtime cost for preventive
        comp = self.component_db.get(prediction.component_name)
        preventive_downtime_cost = 0.0
        reactive_downtime_cost = 0.0
        if comp:
            preventive_downtime_cost = comp.downtime_hours * self.cost_of_downtime_per_hour
            reactive_downtime_cost = (
                comp.downtime_hours * 2.0  # emergency takes longer
                * self.cost_of_downtime_per_hour
            )

        total_preventive = preventive_cost + preventive_downtime_cost
        total_reactive = prediction.cost_if_failure + reactive_downtime_cost

        # Expected reactive cost considers probability of failure
        expected_reactive = total_reactive * prediction.risk_score

        savings = expected_reactive - total_preventive
        roi = savings / total_preventive if total_preventive > 0 else 0.0

        return {
            "preventive_parts_cost": preventive_cost,
            "preventive_downtime_cost": round(preventive_downtime_cost, 2),
            "total_preventive_cost": round(total_preventive, 2),
            "reactive_parts_cost": prediction.cost_if_failure,
            "reactive_downtime_cost": round(reactive_downtime_cost, 2),
            "total_reactive_cost": round(total_reactive, 2),
            "failure_probability": prediction.risk_score,
            "expected_reactive_cost": round(expected_reactive, 2),
            "savings_from_preventive": round(savings, 2),
            "roi": round(roi, 4),
        }
