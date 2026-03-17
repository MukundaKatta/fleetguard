"""Cost analyzer for maintenance vs. replacement decisions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np

from fleetguard.models import Vehicle
from fleetguard.predictor.components import ComponentDatabase


@dataclass
class CostAnalysis:
    """Result of a maintenance cost analysis."""

    vehicle_id: str
    total_maintenance_cost_to_date: float
    projected_annual_maintenance: float
    vehicle_current_value: float
    annual_depreciation: float
    cost_per_mile: float
    replacement_threshold_reached: bool
    recommendation: str
    details: dict[str, float]


class CostAnalyzer:
    """Analyze maintenance costs vs. vehicle replacement economics.

    Implements a total-cost-of-ownership model that compares ongoing
    maintenance costs against the cost of vehicle replacement.
    """

    # Average new vehicle cost for fleet
    DEFAULT_NEW_VEHICLE_COST = 35000.0
    # Depreciation: vehicles lose ~15% per year (declining balance)
    ANNUAL_DEPRECIATION_RATE = 0.15
    # When annual maintenance exceeds this fraction of vehicle value, consider replacement
    REPLACEMENT_COST_RATIO = 0.50
    # Salvage value floor as fraction of original cost
    SALVAGE_FLOOR_RATIO = 0.05

    def __init__(
        self,
        component_db: Optional[ComponentDatabase] = None,
        new_vehicle_cost: float = DEFAULT_NEW_VEHICLE_COST,
    ) -> None:
        self.component_db = component_db or ComponentDatabase()
        self.new_vehicle_cost = new_vehicle_cost

    def analyze_vehicle(
        self,
        vehicle: Vehicle,
        projected_annual_miles: Optional[float] = None,
    ) -> CostAnalysis:
        """Perform cost analysis for a single vehicle."""
        if projected_annual_miles is None:
            projected_annual_miles = vehicle.avg_daily_miles * 365.25

        # Historical maintenance costs
        total_cost = sum(r.cost for r in vehicle.service_history)

        # Cost per mile
        cost_per_mile = (
            total_cost / vehicle.current_mileage
            if vehicle.current_mileage > 0
            else 0.0
        )

        # Project annual maintenance using historical rate + age factor
        age_years = max(vehicle.age_years, 0.5)
        annual_rate = total_cost / age_years if age_years > 0 else 0.0
        # Maintenance costs increase ~8% per year as vehicles age
        age_factor = 1.08 ** max(0, age_years - 3)
        projected_annual = annual_rate * age_factor

        # Vehicle current value (declining balance depreciation)
        current_value = self._estimate_vehicle_value(vehicle)
        annual_depreciation = current_value * self.ANNUAL_DEPRECIATION_RATE

        # Replacement decision
        threshold_reached = projected_annual > (
            current_value * self.REPLACEMENT_COST_RATIO
        )

        # Build recommendation
        if threshold_reached:
            recommendation = (
                f"Consider replacement. Projected annual maintenance "
                f"(${projected_annual:,.0f}) exceeds {self.REPLACEMENT_COST_RATIO:.0%} "
                f"of vehicle value (${current_value:,.0f})."
            )
        elif projected_annual > current_value * 0.30:
            recommendation = (
                f"Monitor closely. Maintenance costs are approaching "
                f"replacement threshold."
            )
        else:
            recommendation = (
                f"Continue maintaining. Maintenance costs are within "
                f"acceptable range."
            )

        # TCO comparison: keep vs replace over next 3 years
        keep_3yr = self._project_keep_cost(vehicle, 3)
        replace_3yr = self._project_replace_cost(3)

        details = {
            "historical_cost_per_mile": round(cost_per_mile, 4),
            "age_factor": round(age_factor, 2),
            "keep_3yr_tco": round(keep_3yr, 2),
            "replace_3yr_tco": round(replace_3yr, 2),
            "breakeven_years": round(
                self._breakeven_years(vehicle), 1
            ),
        }

        return CostAnalysis(
            vehicle_id=vehicle.vehicle_id,
            total_maintenance_cost_to_date=round(total_cost, 2),
            projected_annual_maintenance=round(projected_annual, 2),
            vehicle_current_value=round(current_value, 2),
            annual_depreciation=round(annual_depreciation, 2),
            cost_per_mile=round(cost_per_mile, 4),
            replacement_threshold_reached=threshold_reached,
            recommendation=recommendation,
            details=details,
        )

    def analyze_fleet(self, vehicles: list[Vehicle]) -> list[CostAnalysis]:
        """Analyze costs for all vehicles in the fleet."""
        analyses = [self.analyze_vehicle(v) for v in vehicles]
        analyses.sort(
            key=lambda a: a.projected_annual_maintenance, reverse=True
        )
        return analyses

    def _estimate_vehicle_value(self, vehicle: Vehicle) -> float:
        """Estimate current vehicle value using declining balance depreciation."""
        age = vehicle.age_years
        value = self.new_vehicle_cost * (
            (1 - self.ANNUAL_DEPRECIATION_RATE) ** age
        )
        floor = self.new_vehicle_cost * self.SALVAGE_FLOOR_RATIO
        return max(value, floor)

    def _project_keep_cost(self, vehicle: Vehicle, years: int) -> float:
        """Project total cost of keeping the vehicle for N years."""
        total_cost = sum(r.cost for r in vehicle.service_history)
        age_years = max(vehicle.age_years, 0.5)
        annual_rate = total_cost / age_years

        total = 0.0
        for yr in range(years):
            future_age = age_years + yr
            age_factor = 1.08 ** max(0, future_age - 3)
            total += annual_rate * age_factor
        return total

    def _project_replace_cost(self, years: int) -> float:
        """Project total cost of buying a new vehicle and maintaining it for N years.

        New vehicles have lower maintenance costs but higher capital cost.
        """
        capital = self.new_vehicle_cost
        # New vehicle: ~$500/yr maintenance, growing 5% per year
        maintenance = sum(500.0 * (1.05 ** yr) for yr in range(years))
        # Residual value after N years
        residual = self.new_vehicle_cost * (
            (1 - self.ANNUAL_DEPRECIATION_RATE) ** years
        )
        return capital + maintenance - residual

    def _breakeven_years(self, vehicle: Vehicle) -> float:
        """Estimate years until replacement becomes cheaper than keeping."""
        for yr in range(1, 15):
            keep = self._project_keep_cost(vehicle, yr)
            replace = self._project_replace_cost(yr)
            if replace < keep:
                return float(yr)
        return 15.0
