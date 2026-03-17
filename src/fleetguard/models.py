"""Pydantic models for FleetGuard."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ComponentCategory(str, Enum):
    ENGINE = "engine"
    TRANSMISSION = "transmission"
    BRAKES = "brakes"
    TIRES = "tires"
    ELECTRICAL = "electrical"
    SUSPENSION = "suspension"
    COOLING = "cooling"
    FUEL = "fuel"
    EXHAUST = "exhaust"
    DRIVETRAIN = "drivetrain"
    STEERING = "steering"
    HVAC = "hvac"
    BODY = "body"


class FailureSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Component(BaseModel):
    """A vehicle component with failure distribution parameters."""

    name: str
    category: ComponentCategory
    mean_life_miles: float = Field(gt=0, description="Mean life in miles")
    std_life_miles: float = Field(gt=0, description="Std deviation of life in miles")
    mean_life_months: float = Field(gt=0, description="Mean life in months")
    weibull_shape: float = Field(gt=0, description="Weibull shape parameter (beta)")
    weibull_scale: float = Field(gt=0, description="Weibull scale parameter (eta) in miles")
    replacement_cost: float = Field(ge=0)
    labor_hours: float = Field(ge=0)
    labor_rate_per_hour: float = Field(default=120.0, ge=0)
    downtime_hours: float = Field(ge=0, description="Expected downtime for repair")
    severity: FailureSeverity = FailureSeverity.MEDIUM
    preventive_maintenance_interval_miles: Optional[float] = None
    preventive_maintenance_interval_months: Optional[float] = None

    @property
    def total_repair_cost(self) -> float:
        return self.replacement_cost + self.labor_hours * self.labor_rate_per_hour


class Vehicle(BaseModel):
    """A fleet vehicle."""

    vehicle_id: str
    make: str
    model: str
    year: int
    vin: Optional[str] = None
    current_mileage: float = Field(ge=0)
    purchase_date: date
    avg_daily_miles: float = Field(default=80.0, ge=0)
    service_history: list[ServiceRecord] = Field(default_factory=list)

    @property
    def age_months(self) -> float:
        delta = date.today() - self.purchase_date
        return delta.days / 30.44

    @property
    def age_years(self) -> float:
        return self.age_months / 12.0


class ServiceRecord(BaseModel):
    """A record of a service event."""

    record_id: str
    vehicle_id: str
    component_name: str
    service_date: date
    mileage_at_service: float = Field(ge=0)
    service_type: str = Field(description="preventive, corrective, or emergency")
    cost: float = Field(ge=0)
    notes: Optional[str] = None
    parts_replaced: list[str] = Field(default_factory=list)
    downtime_hours: float = Field(default=0.0, ge=0)


class FailurePrediction(BaseModel):
    """A prediction of component failure."""

    vehicle_id: str
    component_name: str
    predicted_failure_mileage: float
    predicted_failure_date: date
    confidence: float = Field(ge=0.0, le=1.0)
    survival_probability: float = Field(ge=0.0, le=1.0)
    risk_score: float = Field(ge=0.0, le=1.0)
    severity: FailureSeverity
    recommended_service_mileage: float
    recommended_service_date: date
    estimated_cost: float = Field(ge=0)
    cost_if_failure: float = Field(ge=0, description="Cost if component fails unexpectedly")


class TelematicsReading(BaseModel):
    """A single telematics sensor reading."""

    vehicle_id: str
    timestamp: datetime
    mileage: float = Field(ge=0)
    engine_rpm: Optional[float] = None
    engine_temp_celsius: Optional[float] = None
    oil_pressure_psi: Optional[float] = None
    coolant_temp_celsius: Optional[float] = None
    vibration_level: Optional[float] = None
    brake_pad_thickness_mm: Optional[float] = None
    tire_pressure_psi: Optional[dict[str, float]] = None
    battery_voltage: Optional[float] = None
    transmission_temp_celsius: Optional[float] = None
    fuel_consumption_lph: Optional[float] = None
    ambient_temp_celsius: Optional[float] = None


class FleetSummary(BaseModel):
    """Summary statistics for the fleet."""

    total_vehicles: int
    avg_fleet_mileage: float
    avg_fleet_age_months: float
    upcoming_services: int
    critical_alerts: int
    total_monthly_maintenance_cost: float
    predicted_downtime_hours: float


# Rebuild models to resolve forward references
Vehicle.model_rebuild()
