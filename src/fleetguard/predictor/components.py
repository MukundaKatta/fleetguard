"""Component database with real-world failure distributions and maintenance intervals."""

from __future__ import annotations

from fleetguard.models import Component, ComponentCategory, FailureSeverity


class ComponentDatabase:
    """Database of vehicle components with failure distribution parameters.

    Failure distributions are based on typical commercial fleet data.
    Weibull parameters model wear-out failures (shape > 1 = wear-out pattern).
    Mean life values represent average expected component life.
    """

    def __init__(self) -> None:
        self._components: dict[str, Component] = {}
        self._load_defaults()

    def _load_defaults(self) -> None:
        """Load default component definitions with real maintenance intervals."""
        defaults = [
            # --- ENGINE ---
            Component(
                name="engine_oil",
                category=ComponentCategory.ENGINE,
                mean_life_miles=7500,
                std_life_miles=1500,
                mean_life_months=6,
                weibull_shape=3.5,
                weibull_scale=8500,
                replacement_cost=45,
                labor_hours=0.5,
                downtime_hours=1,
                severity=FailureSeverity.MEDIUM,
                preventive_maintenance_interval_miles=7500,
                preventive_maintenance_interval_months=6,
            ),
            Component(
                name="engine_air_filter",
                category=ComponentCategory.ENGINE,
                mean_life_miles=30000,
                std_life_miles=5000,
                mean_life_months=24,
                weibull_shape=2.8,
                weibull_scale=33000,
                replacement_cost=25,
                labor_hours=0.25,
                downtime_hours=0.5,
                severity=FailureSeverity.LOW,
                preventive_maintenance_interval_miles=30000,
                preventive_maintenance_interval_months=24,
            ),
            Component(
                name="spark_plugs",
                category=ComponentCategory.ENGINE,
                mean_life_miles=60000,
                std_life_miles=10000,
                mean_life_months=48,
                weibull_shape=3.0,
                weibull_scale=65000,
                replacement_cost=80,
                labor_hours=1.5,
                downtime_hours=2,
                severity=FailureSeverity.MEDIUM,
                preventive_maintenance_interval_miles=60000,
                preventive_maintenance_interval_months=48,
            ),
            Component(
                name="timing_belt",
                category=ComponentCategory.ENGINE,
                mean_life_miles=90000,
                std_life_miles=15000,
                mean_life_months=84,
                weibull_shape=3.2,
                weibull_scale=100000,
                replacement_cost=350,
                labor_hours=4.0,
                downtime_hours=6,
                severity=FailureSeverity.CRITICAL,
                preventive_maintenance_interval_miles=90000,
                preventive_maintenance_interval_months=84,
            ),
            Component(
                name="serpentine_belt",
                category=ComponentCategory.ENGINE,
                mean_life_miles=60000,
                std_life_miles=12000,
                mean_life_months=60,
                weibull_shape=2.5,
                weibull_scale=68000,
                replacement_cost=40,
                labor_hours=0.75,
                downtime_hours=1,
                severity=FailureSeverity.HIGH,
                preventive_maintenance_interval_miles=60000,
                preventive_maintenance_interval_months=60,
            ),
            # --- TRANSMISSION ---
            Component(
                name="transmission_fluid",
                category=ComponentCategory.TRANSMISSION,
                mean_life_miles=60000,
                std_life_miles=10000,
                mean_life_months=48,
                weibull_shape=3.0,
                weibull_scale=65000,
                replacement_cost=150,
                labor_hours=1.5,
                downtime_hours=2,
                severity=FailureSeverity.MEDIUM,
                preventive_maintenance_interval_miles=60000,
                preventive_maintenance_interval_months=48,
            ),
            Component(
                name="transmission_assembly",
                category=ComponentCategory.TRANSMISSION,
                mean_life_miles=200000,
                std_life_miles=40000,
                mean_life_months=144,
                weibull_shape=2.2,
                weibull_scale=220000,
                replacement_cost=3500,
                labor_hours=12.0,
                downtime_hours=24,
                severity=FailureSeverity.CRITICAL,
            ),
            # --- BRAKES ---
            Component(
                name="brake_pads_front",
                category=ComponentCategory.BRAKES,
                mean_life_miles=40000,
                std_life_miles=10000,
                mean_life_months=36,
                weibull_shape=2.8,
                weibull_scale=45000,
                replacement_cost=150,
                labor_hours=1.0,
                downtime_hours=1.5,
                severity=FailureSeverity.HIGH,
                preventive_maintenance_interval_miles=40000,
                preventive_maintenance_interval_months=36,
            ),
            Component(
                name="brake_pads_rear",
                category=ComponentCategory.BRAKES,
                mean_life_miles=50000,
                std_life_miles=12000,
                mean_life_months=42,
                weibull_shape=2.8,
                weibull_scale=55000,
                replacement_cost=130,
                labor_hours=1.0,
                downtime_hours=1.5,
                severity=FailureSeverity.HIGH,
                preventive_maintenance_interval_miles=50000,
                preventive_maintenance_interval_months=42,
            ),
            Component(
                name="brake_rotors",
                category=ComponentCategory.BRAKES,
                mean_life_miles=70000,
                std_life_miles=15000,
                mean_life_months=60,
                weibull_shape=2.5,
                weibull_scale=78000,
                replacement_cost=300,
                labor_hours=2.0,
                downtime_hours=3,
                severity=FailureSeverity.HIGH,
                preventive_maintenance_interval_miles=70000,
                preventive_maintenance_interval_months=60,
            ),
            Component(
                name="brake_fluid",
                category=ComponentCategory.BRAKES,
                mean_life_miles=45000,
                std_life_miles=8000,
                mean_life_months=24,
                weibull_shape=3.0,
                weibull_scale=50000,
                replacement_cost=20,
                labor_hours=0.5,
                downtime_hours=1,
                severity=FailureSeverity.MEDIUM,
                preventive_maintenance_interval_miles=45000,
                preventive_maintenance_interval_months=24,
            ),
            # --- TIRES ---
            Component(
                name="tires",
                category=ComponentCategory.TIRES,
                mean_life_miles=50000,
                std_life_miles=10000,
                mean_life_months=48,
                weibull_shape=3.5,
                weibull_scale=55000,
                replacement_cost=600,
                labor_hours=1.0,
                downtime_hours=1.5,
                severity=FailureSeverity.HIGH,
                preventive_maintenance_interval_miles=50000,
                preventive_maintenance_interval_months=48,
            ),
            # --- ELECTRICAL ---
            Component(
                name="battery",
                category=ComponentCategory.ELECTRICAL,
                mean_life_miles=50000,
                std_life_miles=15000,
                mean_life_months=48,
                weibull_shape=2.0,
                weibull_scale=55000,
                replacement_cost=180,
                labor_hours=0.5,
                downtime_hours=1,
                severity=FailureSeverity.HIGH,
                preventive_maintenance_interval_months=48,
            ),
            Component(
                name="alternator",
                category=ComponentCategory.ELECTRICAL,
                mean_life_miles=100000,
                std_life_miles=25000,
                mean_life_months=84,
                weibull_shape=2.0,
                weibull_scale=110000,
                replacement_cost=400,
                labor_hours=2.0,
                downtime_hours=3,
                severity=FailureSeverity.HIGH,
            ),
            Component(
                name="starter_motor",
                category=ComponentCategory.ELECTRICAL,
                mean_life_miles=120000,
                std_life_miles=30000,
                mean_life_months=96,
                weibull_shape=2.2,
                weibull_scale=130000,
                replacement_cost=350,
                labor_hours=2.5,
                downtime_hours=4,
                severity=FailureSeverity.HIGH,
            ),
            # --- COOLING ---
            Component(
                name="coolant",
                category=ComponentCategory.COOLING,
                mean_life_miles=30000,
                std_life_miles=5000,
                mean_life_months=24,
                weibull_shape=3.0,
                weibull_scale=33000,
                replacement_cost=25,
                labor_hours=0.5,
                downtime_hours=1,
                severity=FailureSeverity.MEDIUM,
                preventive_maintenance_interval_miles=30000,
                preventive_maintenance_interval_months=24,
            ),
            Component(
                name="radiator",
                category=ComponentCategory.COOLING,
                mean_life_miles=150000,
                std_life_miles=30000,
                mean_life_months=120,
                weibull_shape=2.5,
                weibull_scale=165000,
                replacement_cost=500,
                labor_hours=3.0,
                downtime_hours=5,
                severity=FailureSeverity.HIGH,
            ),
            Component(
                name="water_pump",
                category=ComponentCategory.COOLING,
                mean_life_miles=100000,
                std_life_miles=20000,
                mean_life_months=84,
                weibull_shape=2.8,
                weibull_scale=110000,
                replacement_cost=350,
                labor_hours=3.0,
                downtime_hours=4,
                severity=FailureSeverity.HIGH,
            ),
            Component(
                name="thermostat",
                category=ComponentCategory.COOLING,
                mean_life_miles=100000,
                std_life_miles=25000,
                mean_life_months=96,
                weibull_shape=2.0,
                weibull_scale=110000,
                replacement_cost=50,
                labor_hours=1.0,
                downtime_hours=2,
                severity=FailureSeverity.MEDIUM,
            ),
            # --- SUSPENSION ---
            Component(
                name="shock_absorbers",
                category=ComponentCategory.SUSPENSION,
                mean_life_miles=75000,
                std_life_miles=15000,
                mean_life_months=60,
                weibull_shape=2.5,
                weibull_scale=82000,
                replacement_cost=600,
                labor_hours=3.0,
                downtime_hours=4,
                severity=FailureSeverity.MEDIUM,
                preventive_maintenance_interval_miles=75000,
            ),
            # --- FUEL ---
            Component(
                name="fuel_filter",
                category=ComponentCategory.FUEL,
                mean_life_miles=30000,
                std_life_miles=5000,
                mean_life_months=24,
                weibull_shape=3.0,
                weibull_scale=33000,
                replacement_cost=30,
                labor_hours=0.5,
                downtime_hours=1,
                severity=FailureSeverity.MEDIUM,
                preventive_maintenance_interval_miles=30000,
                preventive_maintenance_interval_months=24,
            ),
            Component(
                name="fuel_pump",
                category=ComponentCategory.FUEL,
                mean_life_miles=120000,
                std_life_miles=30000,
                mean_life_months=96,
                weibull_shape=2.0,
                weibull_scale=130000,
                replacement_cost=500,
                labor_hours=3.0,
                downtime_hours=5,
                severity=FailureSeverity.HIGH,
            ),
            # --- EXHAUST ---
            Component(
                name="catalytic_converter",
                category=ComponentCategory.EXHAUST,
                mean_life_miles=100000,
                std_life_miles=20000,
                mean_life_months=96,
                weibull_shape=2.5,
                weibull_scale=110000,
                replacement_cost=1200,
                labor_hours=2.0,
                downtime_hours=3,
                severity=FailureSeverity.HIGH,
            ),
            # --- STEERING ---
            Component(
                name="power_steering_fluid",
                category=ComponentCategory.STEERING,
                mean_life_miles=50000,
                std_life_miles=10000,
                mean_life_months=36,
                weibull_shape=3.0,
                weibull_scale=55000,
                replacement_cost=20,
                labor_hours=0.5,
                downtime_hours=1,
                severity=FailureSeverity.LOW,
                preventive_maintenance_interval_miles=50000,
                preventive_maintenance_interval_months=36,
            ),
        ]

        for component in defaults:
            self._components[component.name] = component

    def get(self, name: str) -> Component | None:
        """Get a component by name."""
        return self._components.get(name)

    def get_all(self) -> list[Component]:
        """Get all components."""
        return list(self._components.values())

    def get_by_category(self, category: ComponentCategory) -> list[Component]:
        """Get components by category."""
        return [c for c in self._components.values() if c.category == category]

    def get_critical(self) -> list[Component]:
        """Get components with critical severity."""
        return [
            c
            for c in self._components.values()
            if c.severity == FailureSeverity.CRITICAL
        ]

    def add(self, component: Component) -> None:
        """Add or update a component in the database."""
        self._components[component.name] = component

    def names(self) -> list[str]:
        """Get all component names."""
        return list(self._components.keys())

    def __len__(self) -> int:
        return len(self._components)

    def __iter__(self):
        return iter(self._components.values())
