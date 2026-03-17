"""Failure prediction using survival analysis and Gradient Boosting."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier

from fleetguard.models import (
    Component,
    FailurePrediction,
    FailureSeverity,
    ServiceRecord,
    TelematicsReading,
    Vehicle,
)
from fleetguard.predictor.components import ComponentDatabase


class FailurePredictor:
    """Predict component failures using survival analysis and gradient boosting.

    Combines Weibull survival analysis (physics-based) with a GradientBoosting
    model trained on telematics features for more accurate predictions.
    """

    def __init__(self, component_db: Optional[ComponentDatabase] = None) -> None:
        self.component_db = component_db or ComponentDatabase()
        self._gb_model: Optional[GradientBoostingClassifier] = None
        self._is_trained = False

    # --- Survival Analysis ---

    def weibull_survival_probability(
        self,
        component: Component,
        miles_since_service: float,
    ) -> float:
        """Compute survival probability using Weibull distribution.

        S(t) = exp(-(t/eta)^beta)
        where beta = shape, eta = scale, t = miles since last service.
        """
        if miles_since_service <= 0:
            return 1.0
        beta = component.weibull_shape
        eta = component.weibull_scale
        return float(np.exp(-((miles_since_service / eta) ** beta)))

    def weibull_hazard_rate(
        self,
        component: Component,
        miles_since_service: float,
    ) -> float:
        """Compute instantaneous hazard rate from Weibull distribution.

        h(t) = (beta/eta) * (t/eta)^(beta-1)
        """
        if miles_since_service <= 0:
            return 0.0
        beta = component.weibull_shape
        eta = component.weibull_scale
        return (beta / eta) * ((miles_since_service / eta) ** (beta - 1))

    def weibull_remaining_life(
        self,
        component: Component,
        miles_since_service: float,
        target_survival: float = 0.5,
    ) -> float:
        """Estimate remaining miles until survival drops to target probability.

        Finds t* such that S(t*) = target_survival, then returns t* - current miles.
        """
        beta = component.weibull_shape
        eta = component.weibull_scale
        t_star = eta * ((-np.log(target_survival)) ** (1.0 / beta))
        remaining = t_star - miles_since_service
        return max(0.0, float(remaining))

    # --- Feature Engineering ---

    def _extract_features(
        self,
        vehicle: Vehicle,
        component: Component,
        telematics: Optional[list[TelematicsReading]] = None,
    ) -> np.ndarray:
        """Extract features for the gradient boosting model.

        Features:
        0: miles_since_last_service (normalized by mean life)
        1: months_since_last_service (normalized by mean life months)
        2: survival_probability (Weibull)
        3: hazard_rate (normalized)
        4: vehicle_age_months (normalized)
        5: avg_daily_miles (normalized)
        6: service_count_for_component
        7: avg_vibration (from telematics, 0 if unavailable)
        8: avg_engine_temp_deviation (from telematics, 0 if unavailable)
        9: avg_oil_pressure_deviation (from telematics, 0 if unavailable)
        """
        miles_since = self._miles_since_last_service(vehicle, component.name)
        months_since = self._months_since_last_service(vehicle, component.name)

        survival = self.weibull_survival_probability(component, miles_since)
        hazard = self.weibull_hazard_rate(component, miles_since)

        service_count = sum(
            1
            for r in vehicle.service_history
            if r.component_name == component.name
        )

        # Telematics features
        avg_vibration = 0.0
        avg_temp_dev = 0.0
        avg_oil_dev = 0.0
        if telematics:
            vibrations = [
                r.vibration_level for r in telematics if r.vibration_level is not None
            ]
            if vibrations:
                avg_vibration = np.mean(vibrations)

            temps = [
                r.engine_temp_celsius
                for r in telematics
                if r.engine_temp_celsius is not None
            ]
            if temps:
                # Normal operating temp ~95C
                avg_temp_dev = np.mean([abs(t - 95.0) for t in temps]) / 30.0

            oils = [
                r.oil_pressure_psi
                for r in telematics
                if r.oil_pressure_psi is not None
            ]
            if oils:
                # Normal oil pressure ~40 PSI
                avg_oil_dev = np.mean([abs(p - 40.0) for p in oils]) / 20.0

        features = np.array([
            miles_since / component.mean_life_miles,
            months_since / component.mean_life_months,
            survival,
            hazard * component.mean_life_miles,  # normalized hazard
            vehicle.age_months / 120.0,  # normalize by 10 years
            vehicle.avg_daily_miles / 100.0,
            min(service_count / 5.0, 1.0),
            avg_vibration,
            avg_temp_dev,
            avg_oil_dev,
        ])
        return features

    # --- Gradient Boosting Model ---

    def train(
        self,
        vehicles: list[Vehicle],
        failure_labels: list[dict[str, bool]],
        telematics_data: Optional[dict[str, list[TelematicsReading]]] = None,
    ) -> None:
        """Train the gradient boosting model on historical fleet data.

        Args:
            vehicles: List of vehicles with service history.
            failure_labels: List of dicts mapping component_name -> failed (bool)
                           for each vehicle.
            telematics_data: Optional dict mapping vehicle_id -> telematics readings.
        """
        X_list = []
        y_list = []

        for vehicle, labels in zip(vehicles, failure_labels):
            tel = (
                telematics_data.get(vehicle.vehicle_id, [])
                if telematics_data
                else None
            )
            for component in self.component_db:
                if component.name in labels:
                    features = self._extract_features(vehicle, component, tel)
                    X_list.append(features)
                    y_list.append(1 if labels[component.name] else 0)

        if not X_list:
            return

        X = np.array(X_list)
        y = np.array(y_list)

        self._gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        self._gb_model.fit(X, y)
        self._is_trained = True

    def predict_failure_probability(
        self,
        vehicle: Vehicle,
        component: Component,
        telematics: Optional[list[TelematicsReading]] = None,
    ) -> float:
        """Predict failure probability combining survival analysis and ML.

        If the GB model is trained, blends ML prediction with Weibull.
        Otherwise, uses pure survival analysis.
        """
        miles_since = self._miles_since_last_service(vehicle, component.name)
        weibull_failure_prob = 1.0 - self.weibull_survival_probability(
            component, miles_since
        )

        if self._is_trained and self._gb_model is not None:
            features = self._extract_features(vehicle, component, telematics)
            ml_prob = self._gb_model.predict_proba(features.reshape(1, -1))[0, 1]
            # Weighted blend: 40% Weibull, 60% ML when model is trained
            return 0.4 * weibull_failure_prob + 0.6 * ml_prob

        return weibull_failure_prob

    def predict_component(
        self,
        vehicle: Vehicle,
        component: Component,
        telematics: Optional[list[TelematicsReading]] = None,
    ) -> FailurePrediction:
        """Generate a full failure prediction for a vehicle component."""
        miles_since = self._miles_since_last_service(vehicle, component.name)
        survival_prob = self.weibull_survival_probability(component, miles_since)
        failure_prob = self.predict_failure_probability(
            vehicle, component, telematics
        )

        remaining_miles = self.weibull_remaining_life(component, miles_since)
        predicted_failure_mileage = vehicle.current_mileage + remaining_miles

        days_to_failure = remaining_miles / max(vehicle.avg_daily_miles, 1.0)
        predicted_failure_date = date.today() + timedelta(days=days_to_failure)

        # Recommend service at 80% of remaining life
        recommended_service_miles = vehicle.current_mileage + remaining_miles * 0.8
        days_to_service = (remaining_miles * 0.8) / max(vehicle.avg_daily_miles, 1.0)
        recommended_service_date = date.today() + timedelta(days=days_to_service)

        # Unexpected failure costs 2.5x more (towing, emergency, cascading damage)
        cost_if_failure = component.total_repair_cost * 2.5

        return FailurePrediction(
            vehicle_id=vehicle.vehicle_id,
            component_name=component.name,
            predicted_failure_mileage=round(predicted_failure_mileage, 0),
            predicted_failure_date=predicted_failure_date,
            confidence=self._compute_confidence(component, miles_since),
            survival_probability=round(survival_prob, 4),
            risk_score=round(failure_prob, 4),
            severity=component.severity,
            recommended_service_mileage=round(recommended_service_miles, 0),
            recommended_service_date=recommended_service_date,
            estimated_cost=round(component.total_repair_cost, 2),
            cost_if_failure=round(cost_if_failure, 2),
        )

    def predict_vehicle(
        self,
        vehicle: Vehicle,
        telematics: Optional[list[TelematicsReading]] = None,
    ) -> list[FailurePrediction]:
        """Generate failure predictions for all components of a vehicle."""
        predictions = []
        for component in self.component_db:
            pred = self.predict_component(vehicle, component, telematics)
            predictions.append(pred)
        predictions.sort(key=lambda p: p.risk_score, reverse=True)
        return predictions

    # --- Helpers ---

    def _miles_since_last_service(
        self, vehicle: Vehicle, component_name: str
    ) -> float:
        """Calculate miles since the last service for a component."""
        relevant = [
            r
            for r in vehicle.service_history
            if r.component_name == component_name
        ]
        if not relevant:
            return vehicle.current_mileage
        latest = max(relevant, key=lambda r: r.mileage_at_service)
        return vehicle.current_mileage - latest.mileage_at_service

    def _months_since_last_service(
        self, vehicle: Vehicle, component_name: str
    ) -> float:
        """Calculate months since the last service for a component."""
        relevant = [
            r
            for r in vehicle.service_history
            if r.component_name == component_name
        ]
        if not relevant:
            return vehicle.age_months
        latest = max(relevant, key=lambda r: r.service_date)
        delta = date.today() - latest.service_date
        return delta.days / 30.44

    def _compute_confidence(
        self, component: Component, miles_since_service: float
    ) -> float:
        """Compute prediction confidence based on data quality.

        Confidence is higher when:
        - Component has known Weibull parameters
        - Current mileage is within typical failure range
        - ML model is trained
        """
        # Base confidence from Weibull model
        ratio = miles_since_service / component.mean_life_miles
        # Confidence peaks when ratio is near 1.0 (in expected failure zone)
        base = float(stats.norm.pdf(ratio, loc=1.0, scale=0.3) / stats.norm.pdf(1.0, loc=1.0, scale=0.3))
        base = max(0.3, min(0.9, base))

        if self._is_trained:
            base = min(1.0, base + 0.1)

        return round(base, 3)
