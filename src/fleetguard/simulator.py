"""Fleet simulator for generating realistic test data."""

from __future__ import annotations

import random
import uuid
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np

from fleetguard.fleet.vehicle import VehicleManager
from fleetguard.models import ServiceRecord, TelematicsReading, Vehicle
from fleetguard.predictor.components import ComponentDatabase


class FleetSimulator:
    """Generate realistic fleet data for testing and demonstration.

    Simulates vehicle aging, component wear, service events, and
    telematics readings with realistic distributions.
    """

    MAKES_MODELS = [
        ("Ford", "Transit"),
        ("Ford", "F-150"),
        ("Chevrolet", "Express"),
        ("Chevrolet", "Silverado"),
        ("RAM", "ProMaster"),
        ("RAM", "1500"),
        ("Toyota", "Tacoma"),
        ("GMC", "Sierra"),
        ("Nissan", "NV200"),
        ("Mercedes-Benz", "Sprinter"),
    ]

    def __init__(
        self,
        component_db: Optional[ComponentDatabase] = None,
        seed: int = 42,
    ) -> None:
        self.component_db = component_db or ComponentDatabase()
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

    def generate_fleet(
        self,
        num_vehicles: int = 10,
        min_age_years: float = 1.0,
        max_age_years: float = 8.0,
        min_daily_miles: float = 40.0,
        max_daily_miles: float = 150.0,
    ) -> VehicleManager:
        """Generate a fleet of vehicles with realistic attributes."""
        manager = VehicleManager()

        for i in range(num_vehicles):
            make, model = random.choice(self.MAKES_MODELS)
            age_years = self.rng.uniform(min_age_years, max_age_years)
            purchase_date = date.today() - timedelta(days=int(age_years * 365.25))
            avg_daily = self.rng.uniform(min_daily_miles, max_daily_miles)
            mileage = avg_daily * age_years * 365.25
            # Add some variance
            mileage *= self.rng.uniform(0.85, 1.15)

            year = purchase_date.year
            vehicle_id = f"VH-{i+1:04d}"

            vehicle = Vehicle(
                vehicle_id=vehicle_id,
                make=make,
                model=model,
                year=year,
                vin=self._random_vin(),
                current_mileage=round(mileage, 0),
                purchase_date=purchase_date,
                avg_daily_miles=round(avg_daily, 1),
            )

            # Generate service history
            records = self._generate_service_history(vehicle)
            vehicle.service_history = records

            manager.add_vehicle(vehicle)

        return manager

    def generate_telematics(
        self,
        vehicle: Vehicle,
        num_readings: int = 50,
        days_back: int = 30,
    ) -> list[TelematicsReading]:
        """Generate realistic telematics readings for a vehicle."""
        readings = []
        now = datetime.now()
        age_factor = max(1.0, vehicle.age_years / 5.0)

        for i in range(num_readings):
            ts = now - timedelta(
                days=self.rng.uniform(0, days_back),
                hours=self.rng.uniform(0, 24),
            )
            mileage = vehicle.current_mileage - self.rng.uniform(
                0, vehicle.avg_daily_miles * days_back
            )

            # Normal values with age-dependent noise
            reading = TelematicsReading(
                vehicle_id=vehicle.vehicle_id,
                timestamp=ts,
                mileage=round(max(0, mileage), 1),
                engine_rpm=round(self.rng.normal(2500, 500), 0),
                engine_temp_celsius=round(
                    self.rng.normal(92 + age_factor * 2, 5 * age_factor), 1
                ),
                oil_pressure_psi=round(
                    self.rng.normal(42 - age_factor * 2, 5 * age_factor), 1
                ),
                coolant_temp_celsius=round(
                    self.rng.normal(90 + age_factor, 4 * age_factor), 1
                ),
                vibration_level=round(
                    max(0, self.rng.normal(1.5 * age_factor, 0.8 * age_factor)), 2
                ),
                brake_pad_thickness_mm=round(
                    max(1.0, self.rng.normal(8 / age_factor, 1.5)), 1
                ),
                tire_pressure_psi={
                    "FL": round(self.rng.normal(33, 1.5), 1),
                    "FR": round(self.rng.normal(33, 1.5), 1),
                    "RL": round(self.rng.normal(33, 1.5), 1),
                    "RR": round(self.rng.normal(33, 1.5), 1),
                },
                battery_voltage=round(
                    self.rng.normal(13.5 - age_factor * 0.2, 0.5), 2
                ),
                transmission_temp_celsius=round(
                    self.rng.normal(85 + age_factor * 3, 8), 1
                ),
                fuel_consumption_lph=round(
                    self.rng.normal(8 + age_factor, 2), 2
                ),
                ambient_temp_celsius=round(self.rng.normal(22, 10), 1),
            )
            readings.append(reading)

        readings.sort(key=lambda r: r.timestamp)
        return readings

    def _generate_service_history(
        self, vehicle: Vehicle
    ) -> list[ServiceRecord]:
        """Generate realistic service history based on mileage and age."""
        records: list[ServiceRecord] = []
        purchase_date = vehicle.purchase_date

        for component in self.component_db:
            if component.preventive_maintenance_interval_miles is None:
                continue

            interval = component.preventive_maintenance_interval_miles
            num_services = int(vehicle.current_mileage / interval)

            for s in range(num_services):
                service_mileage = interval * (s + 1)
                # Add realistic variance (+/- 15%)
                service_mileage *= self.rng.uniform(0.85, 1.15)
                if service_mileage > vehicle.current_mileage:
                    continue

                days_into_ownership = (
                    service_mileage / vehicle.current_mileage
                ) * (date.today() - purchase_date).days
                service_date = purchase_date + timedelta(
                    days=int(days_into_ownership)
                )

                # Determine service type
                if self.rng.random() < 0.1:
                    service_type = "corrective"
                    cost = component.total_repair_cost * self.rng.uniform(1.0, 1.5)
                else:
                    service_type = "preventive"
                    cost = component.total_repair_cost * self.rng.uniform(0.8, 1.1)

                record = ServiceRecord(
                    record_id=str(uuid.uuid4())[:8],
                    vehicle_id=vehicle.vehicle_id,
                    component_name=component.name,
                    service_date=service_date,
                    mileage_at_service=round(service_mileage, 0),
                    service_type=service_type,
                    cost=round(cost, 2),
                    parts_replaced=[component.name],
                    downtime_hours=component.downtime_hours,
                )
                records.append(record)

        records.sort(key=lambda r: r.service_date)
        return records

    def _random_vin(self) -> str:
        """Generate a random VIN-like string."""
        chars = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"
        return "".join(random.choices(chars, k=17))
