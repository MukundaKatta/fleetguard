"""Vehicle manager for fleet tracking."""

from __future__ import annotations

from datetime import date
from typing import Optional

from fleetguard.models import ServiceRecord, Vehicle


class VehicleManager:
    """Manage a fleet of vehicles with mileage, age, and service history tracking."""

    def __init__(self) -> None:
        self._vehicles: dict[str, Vehicle] = {}

    def add_vehicle(self, vehicle: Vehicle) -> None:
        """Add a vehicle to the fleet."""
        self._vehicles[vehicle.vehicle_id] = vehicle

    def remove_vehicle(self, vehicle_id: str) -> Optional[Vehicle]:
        """Remove a vehicle from the fleet."""
        return self._vehicles.pop(vehicle_id, None)

    def get_vehicle(self, vehicle_id: str) -> Optional[Vehicle]:
        """Get a vehicle by ID."""
        return self._vehicles.get(vehicle_id)

    def get_all(self) -> list[Vehicle]:
        """Get all vehicles in the fleet."""
        return list(self._vehicles.values())

    def update_mileage(self, vehicle_id: str, new_mileage: float) -> None:
        """Update a vehicle's current mileage."""
        vehicle = self._vehicles.get(vehicle_id)
        if vehicle and new_mileage >= vehicle.current_mileage:
            vehicle.current_mileage = new_mileage

    def add_service_record(
        self, vehicle_id: str, record: ServiceRecord
    ) -> None:
        """Add a service record to a vehicle's history."""
        vehicle = self._vehicles.get(vehicle_id)
        if vehicle:
            vehicle.service_history.append(record)

    def get_service_history(
        self,
        vehicle_id: str,
        component_name: Optional[str] = None,
    ) -> list[ServiceRecord]:
        """Get service history for a vehicle, optionally filtered by component."""
        vehicle = self._vehicles.get(vehicle_id)
        if not vehicle:
            return []
        records = vehicle.service_history
        if component_name:
            records = [r for r in records if r.component_name == component_name]
        return sorted(records, key=lambda r: r.service_date, reverse=True)

    def vehicles_needing_service(
        self,
        max_miles_since_service: float = 10000,
    ) -> list[Vehicle]:
        """Find vehicles that may need service based on mileage since last service."""
        result = []
        for vehicle in self._vehicles.values():
            if not vehicle.service_history:
                result.append(vehicle)
                continue
            latest = max(vehicle.service_history, key=lambda r: r.mileage_at_service)
            if vehicle.current_mileage - latest.mileage_at_service >= max_miles_since_service:
                result.append(vehicle)
        return result

    def fleet_avg_mileage(self) -> float:
        """Compute average fleet mileage."""
        if not self._vehicles:
            return 0.0
        return sum(v.current_mileage for v in self._vehicles.values()) / len(
            self._vehicles
        )

    def fleet_avg_age_months(self) -> float:
        """Compute average fleet age in months."""
        if not self._vehicles:
            return 0.0
        return sum(v.age_months for v in self._vehicles.values()) / len(
            self._vehicles
        )

    def total_maintenance_cost(
        self,
        since: Optional[date] = None,
    ) -> float:
        """Compute total maintenance cost across the fleet."""
        total = 0.0
        for vehicle in self._vehicles.values():
            for record in vehicle.service_history:
                if since and record.service_date < since:
                    continue
                total += record.cost
        return total

    def __len__(self) -> int:
        return len(self._vehicles)

    def __iter__(self):
        return iter(self._vehicles.values())
