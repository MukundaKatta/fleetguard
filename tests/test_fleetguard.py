"""Tests for FleetGuard."""

from datetime import date, datetime

import numpy as np
import pytest

from fleetguard.fleet.cost import CostAnalyzer
from fleetguard.fleet.telematics import TelematicsProcessor
from fleetguard.fleet.vehicle import VehicleManager
from fleetguard.models import (
    Component,
    ComponentCategory,
    FailureSeverity,
    ServiceRecord,
    TelematicsReading,
    Vehicle,
)
from fleetguard.predictor.components import ComponentDatabase
from fleetguard.predictor.model import FailurePredictor
from fleetguard.predictor.scheduler import MaintenanceScheduler
from fleetguard.simulator import FleetSimulator


# --- Fixtures ---


@pytest.fixture
def component_db():
    return ComponentDatabase()


@pytest.fixture
def sample_vehicle():
    return Vehicle(
        vehicle_id="TEST-001",
        make="Ford",
        model="Transit",
        year=2021,
        current_mileage=55000,
        purchase_date=date(2021, 1, 15),
        avg_daily_miles=80,
        service_history=[
            ServiceRecord(
                record_id="SR-001",
                vehicle_id="TEST-001",
                component_name="engine_oil",
                service_date=date(2024, 6, 1),
                mileage_at_service=48000,
                service_type="preventive",
                cost=95.0,
                parts_replaced=["engine_oil"],
                downtime_hours=1.0,
            ),
            ServiceRecord(
                record_id="SR-002",
                vehicle_id="TEST-001",
                component_name="brake_pads_front",
                service_date=date(2023, 9, 15),
                mileage_at_service=38000,
                service_type="preventive",
                cost=270.0,
                parts_replaced=["brake_pads_front"],
                downtime_hours=1.5,
            ),
        ],
    )


@pytest.fixture
def sample_telematics(sample_vehicle):
    readings = []
    for i in range(20):
        readings.append(
            TelematicsReading(
                vehicle_id=sample_vehicle.vehicle_id,
                timestamp=datetime(2025, 1, 1 + i, 10, 0),
                mileage=54000 + i * 50,
                engine_rpm=2500,
                engine_temp_celsius=93 + np.random.normal(0, 3),
                oil_pressure_psi=40 + np.random.normal(0, 3),
                coolant_temp_celsius=90 + np.random.normal(0, 2),
                vibration_level=max(0, 2.0 + np.random.normal(0, 0.5)),
                brake_pad_thickness_mm=6.5,
                tire_pressure_psi={
                    "FL": 33.0,
                    "FR": 33.5,
                    "RL": 32.5,
                    "RR": 33.0,
                },
                battery_voltage=13.2,
                transmission_temp_celsius=88,
            )
        )
    return readings


# --- Component Database Tests ---


class TestComponentDatabase:
    def test_has_minimum_components(self, component_db):
        assert len(component_db) >= 20

    def test_get_component(self, component_db):
        oil = component_db.get("engine_oil")
        assert oil is not None
        assert oil.category == ComponentCategory.ENGINE
        assert oil.mean_life_miles == 7500

    def test_get_by_category(self, component_db):
        brakes = component_db.get_by_category(ComponentCategory.BRAKES)
        assert len(brakes) >= 3
        assert all(c.category == ComponentCategory.BRAKES for c in brakes)

    def test_get_critical(self, component_db):
        critical = component_db.get_critical()
        assert len(critical) >= 1
        assert all(c.severity == FailureSeverity.CRITICAL for c in critical)

    def test_all_components_valid(self, component_db):
        for comp in component_db:
            assert comp.mean_life_miles > 0
            assert comp.weibull_shape > 0
            assert comp.weibull_scale > 0
            assert comp.replacement_cost >= 0

    def test_add_component(self, component_db):
        custom = Component(
            name="custom_part",
            category=ComponentCategory.ENGINE,
            mean_life_miles=10000,
            std_life_miles=2000,
            mean_life_months=12,
            weibull_shape=2.0,
            weibull_scale=11000,
            replacement_cost=100,
            labor_hours=1.0,
            downtime_hours=2.0,
        )
        component_db.add(custom)
        assert component_db.get("custom_part") is not None


# --- Failure Predictor Tests ---


class TestFailurePredictor:
    def test_weibull_survival_new(self, component_db):
        predictor = FailurePredictor(component_db)
        comp = component_db.get("engine_oil")
        # At 0 miles, survival should be 1.0
        assert predictor.weibull_survival_probability(comp, 0) == 1.0

    def test_weibull_survival_decreases(self, component_db):
        predictor = FailurePredictor(component_db)
        comp = component_db.get("engine_oil")
        s1 = predictor.weibull_survival_probability(comp, 3000)
        s2 = predictor.weibull_survival_probability(comp, 6000)
        s3 = predictor.weibull_survival_probability(comp, 9000)
        assert s1 > s2 > s3

    def test_weibull_hazard_rate_increases(self, component_db):
        predictor = FailurePredictor(component_db)
        comp = component_db.get("tires")  # shape > 1 = wear-out
        h1 = predictor.weibull_hazard_rate(comp, 20000)
        h2 = predictor.weibull_hazard_rate(comp, 40000)
        assert h2 > h1

    def test_remaining_life(self, component_db):
        predictor = FailurePredictor(component_db)
        comp = component_db.get("brake_pads_front")
        remaining = predictor.weibull_remaining_life(comp, 0)
        assert remaining > 0
        remaining_later = predictor.weibull_remaining_life(comp, 30000)
        assert remaining_later < remaining

    def test_predict_component(self, component_db, sample_vehicle):
        predictor = FailurePredictor(component_db)
        comp = component_db.get("engine_oil")
        pred = predictor.predict_component(sample_vehicle, comp)
        assert pred.vehicle_id == "TEST-001"
        assert pred.component_name == "engine_oil"
        assert 0 <= pred.risk_score <= 1
        assert 0 <= pred.survival_probability <= 1
        assert pred.estimated_cost > 0

    def test_predict_vehicle(self, component_db, sample_vehicle):
        predictor = FailurePredictor(component_db)
        predictions = predictor.predict_vehicle(sample_vehicle)
        assert len(predictions) == len(component_db)
        # Should be sorted by risk score descending
        for i in range(len(predictions) - 1):
            assert predictions[i].risk_score >= predictions[i + 1].risk_score

    def test_predict_with_telematics(
        self, component_db, sample_vehicle, sample_telematics
    ):
        predictor = FailurePredictor(component_db)
        comp = component_db.get("engine_oil")
        pred = predictor.predict_component(
            sample_vehicle, comp, sample_telematics
        )
        assert pred.risk_score >= 0


# --- Vehicle Manager Tests ---


class TestVehicleManager:
    def test_add_get_vehicle(self, sample_vehicle):
        mgr = VehicleManager()
        mgr.add_vehicle(sample_vehicle)
        assert mgr.get_vehicle("TEST-001") is not None
        assert len(mgr) == 1

    def test_update_mileage(self, sample_vehicle):
        mgr = VehicleManager()
        mgr.add_vehicle(sample_vehicle)
        mgr.update_mileage("TEST-001", 60000)
        assert mgr.get_vehicle("TEST-001").current_mileage == 60000

    def test_reject_lower_mileage(self, sample_vehicle):
        mgr = VehicleManager()
        mgr.add_vehicle(sample_vehicle)
        mgr.update_mileage("TEST-001", 10000)
        assert mgr.get_vehicle("TEST-001").current_mileage == 55000

    def test_service_history(self, sample_vehicle):
        mgr = VehicleManager()
        mgr.add_vehicle(sample_vehicle)
        history = mgr.get_service_history("TEST-001")
        assert len(history) == 2

    def test_fleet_avg_mileage(self, sample_vehicle):
        mgr = VehicleManager()
        mgr.add_vehicle(sample_vehicle)
        assert mgr.fleet_avg_mileage() == 55000


# --- Telematics Processor Tests ---


class TestTelematicsProcessor:
    def test_process_readings(self, sample_telematics):
        processor = TelematicsProcessor()
        features = processor.process(sample_telematics)
        assert features is not None
        assert features.reading_count == 20
        assert features.engine_temp_mean > 0
        assert features.oil_pressure_mean > 0

    def test_health_scores(self, sample_telematics):
        processor = TelematicsProcessor()
        features = processor.process(sample_telematics)
        scores = processor.compute_health_scores(features)
        assert "engine" in scores
        assert "brakes" in scores
        assert all(0 <= v <= 1 for v in scores.values())

    def test_anomaly_detection(self):
        processor = TelematicsProcessor()
        readings = [
            TelematicsReading(
                vehicle_id="TEST",
                timestamp=datetime(2025, 1, 1, 10, 0),
                mileage=50000,
                engine_temp_celsius=120.0,  # Over temp
                oil_pressure_psi=15.0,  # Low pressure
                battery_voltage=11.5,  # Low voltage
            )
        ]
        anomalies = processor.detect_anomalies(readings)
        assert len(anomalies) >= 3

    def test_empty_readings(self):
        processor = TelematicsProcessor()
        assert processor.process([]) is None


# --- Maintenance Scheduler Tests ---


class TestMaintenanceScheduler:
    def test_schedule_vehicle(self, component_db, sample_vehicle):
        scheduler = MaintenanceScheduler(component_db=component_db)
        services = scheduler.schedule_vehicle(sample_vehicle)
        assert isinstance(services, list)

    def test_schedule_groups_services(self, component_db, sample_vehicle):
        scheduler = MaintenanceScheduler(
            component_db=component_db,
            grouping_window_days=30,
        )
        services = scheduler.schedule_vehicle(sample_vehicle)
        # Grouped services should have multiple components
        if services:
            total_components = sum(len(s.components) for s in services)
            assert total_components >= len(services)

    def test_cost_benefit(self, component_db, sample_vehicle):
        predictor = FailurePredictor(component_db)
        scheduler = MaintenanceScheduler(
            predictor=predictor, component_db=component_db
        )
        comp = component_db.get("brake_pads_front")
        pred = predictor.predict_component(sample_vehicle, comp)
        analysis = scheduler.compute_cost_benefit(pred)
        assert "total_preventive_cost" in analysis
        assert "total_reactive_cost" in analysis
        assert "savings_from_preventive" in analysis


# --- Cost Analyzer Tests ---


class TestCostAnalyzer:
    def test_analyze_vehicle(self, component_db, sample_vehicle):
        analyzer = CostAnalyzer(component_db=component_db)
        result = analyzer.analyze_vehicle(sample_vehicle)
        assert result.vehicle_id == "TEST-001"
        assert result.total_maintenance_cost_to_date == 365.0  # 95 + 270
        assert result.vehicle_current_value > 0
        assert result.recommendation

    def test_cost_per_mile(self, component_db, sample_vehicle):
        analyzer = CostAnalyzer(component_db=component_db)
        result = analyzer.analyze_vehicle(sample_vehicle)
        expected = 365.0 / 55000
        assert abs(result.cost_per_mile - expected) < 0.001


# --- Simulator Tests ---


class TestSimulator:
    def test_generate_fleet(self, component_db):
        sim = FleetSimulator(component_db=component_db, seed=123)
        manager = sim.generate_fleet(num_vehicles=5)
        assert len(manager) == 5
        for v in manager:
            assert v.current_mileage > 0
            assert len(v.service_history) > 0

    def test_generate_telematics(self, component_db, sample_vehicle):
        sim = FleetSimulator(component_db=component_db, seed=123)
        readings = sim.generate_telematics(sample_vehicle, num_readings=30)
        assert len(readings) == 30
        assert all(r.vehicle_id == "TEST-001" for r in readings)

    def test_deterministic_seed(self, component_db):
        sim1 = FleetSimulator(component_db=component_db, seed=99)
        sim2 = FleetSimulator(component_db=component_db, seed=99)
        fleet1 = sim1.generate_fleet(3)
        fleet2 = sim2.generate_fleet(3)
        v1 = fleet1.get_all()
        v2 = fleet2.get_all()
        for a, b in zip(v1, v2):
            assert a.vehicle_id == b.vehicle_id
            assert a.current_mileage == b.current_mileage
