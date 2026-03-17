"""Telematics data processor for extracting maintenance-relevant features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from fleetguard.models import TelematicsReading


@dataclass
class TelematicsFeatures:
    """Extracted features from telematics sensor data."""

    vehicle_id: str
    reading_count: int

    # Vibration
    vibration_mean: float = 0.0
    vibration_std: float = 0.0
    vibration_max: float = 0.0
    vibration_trend: float = 0.0  # positive = increasing

    # Engine temperature
    engine_temp_mean: float = 0.0
    engine_temp_std: float = 0.0
    engine_temp_max: float = 0.0
    overtemp_events: int = 0  # count of readings > 110 C

    # Oil pressure
    oil_pressure_mean: float = 0.0
    oil_pressure_std: float = 0.0
    oil_pressure_min: float = 0.0
    low_pressure_events: int = 0  # count of readings < 20 PSI

    # Coolant temperature
    coolant_temp_mean: float = 0.0
    coolant_temp_max: float = 0.0

    # Battery
    battery_voltage_mean: float = 0.0
    battery_voltage_min: float = 0.0
    low_voltage_events: int = 0  # count of readings < 12.0V

    # Transmission
    trans_temp_mean: float = 0.0
    trans_temp_max: float = 0.0
    trans_overtemp_events: int = 0  # count of readings > 120 C

    # Brake
    brake_pad_min_mm: float = 0.0

    # Tire pressure
    tire_pressure_variance: float = 0.0
    tire_low_pressure_events: int = 0


class TelematicsProcessor:
    """Process raw telematics readings into maintenance-relevant features.

    Extracts statistical features from sensor data streams including
    vibration analysis, temperature monitoring, and oil pressure tracking.
    """

    # Normal operating ranges
    NORMAL_ENGINE_TEMP_RANGE = (85.0, 105.0)  # Celsius
    NORMAL_OIL_PRESSURE_RANGE = (25.0, 65.0)  # PSI
    NORMAL_BATTERY_VOLTAGE_RANGE = (12.4, 14.7)  # Volts
    NORMAL_TRANS_TEMP_RANGE = (70.0, 110.0)  # Celsius
    NORMAL_TIRE_PRESSURE_RANGE = (30.0, 36.0)  # PSI
    MIN_BRAKE_PAD_MM = 3.0

    def process(
        self, readings: list[TelematicsReading]
    ) -> Optional[TelematicsFeatures]:
        """Process a batch of telematics readings into features."""
        if not readings:
            return None

        vehicle_id = readings[0].vehicle_id
        features = TelematicsFeatures(
            vehicle_id=vehicle_id,
            reading_count=len(readings),
        )

        self._process_vibration(readings, features)
        self._process_engine_temp(readings, features)
        self._process_oil_pressure(readings, features)
        self._process_coolant(readings, features)
        self._process_battery(readings, features)
        self._process_transmission(readings, features)
        self._process_brakes(readings, features)
        self._process_tires(readings, features)

        return features

    def compute_health_scores(
        self, features: TelematicsFeatures
    ) -> dict[str, float]:
        """Compute component health scores from telematics features.

        Returns scores from 0.0 (poor) to 1.0 (excellent) for each category.
        """
        scores: dict[str, float] = {}

        # Engine health based on temperature and vibration
        engine_temp_score = self._range_score(
            features.engine_temp_mean,
            self.NORMAL_ENGINE_TEMP_RANGE[0],
            self.NORMAL_ENGINE_TEMP_RANGE[1],
            margin=15.0,
        )
        vibration_score = max(0.0, 1.0 - features.vibration_mean / 10.0)
        scores["engine"] = (engine_temp_score + vibration_score) / 2.0

        # Oil system health
        oil_score = self._range_score(
            features.oil_pressure_mean,
            self.NORMAL_OIL_PRESSURE_RANGE[0],
            self.NORMAL_OIL_PRESSURE_RANGE[1],
            margin=10.0,
        )
        penalty = min(0.3, features.low_pressure_events * 0.05)
        scores["oil_system"] = max(0.0, oil_score - penalty)

        # Electrical health
        battery_score = self._range_score(
            features.battery_voltage_mean,
            self.NORMAL_BATTERY_VOLTAGE_RANGE[0],
            self.NORMAL_BATTERY_VOLTAGE_RANGE[1],
            margin=1.0,
        )
        penalty = min(0.3, features.low_voltage_events * 0.05)
        scores["electrical"] = max(0.0, battery_score - penalty)

        # Cooling health
        coolant_score = self._range_score(
            features.coolant_temp_mean,
            self.NORMAL_ENGINE_TEMP_RANGE[0],
            self.NORMAL_ENGINE_TEMP_RANGE[1],
            margin=15.0,
        )
        penalty = min(0.4, features.overtemp_events * 0.1)
        scores["cooling"] = max(0.0, coolant_score - penalty)

        # Transmission health
        trans_score = self._range_score(
            features.trans_temp_mean,
            self.NORMAL_TRANS_TEMP_RANGE[0],
            self.NORMAL_TRANS_TEMP_RANGE[1],
            margin=15.0,
        )
        penalty = min(0.3, features.trans_overtemp_events * 0.1)
        scores["transmission"] = max(0.0, trans_score - penalty)

        # Brake health
        if features.brake_pad_min_mm > 0:
            scores["brakes"] = min(
                1.0, features.brake_pad_min_mm / 12.0
            )  # New pads ~12mm
        else:
            scores["brakes"] = 0.5  # Unknown

        # Tire health
        tire_score = max(0.0, 1.0 - features.tire_pressure_variance / 5.0)
        penalty = min(0.3, features.tire_low_pressure_events * 0.05)
        scores["tires"] = max(0.0, tire_score - penalty)

        return {k: round(v, 3) for k, v in scores.items()}

    def detect_anomalies(
        self, readings: list[TelematicsReading]
    ) -> list[dict[str, str]]:
        """Detect anomalous sensor readings that may indicate issues."""
        anomalies: list[dict[str, str]] = []

        for reading in readings:
            ts = reading.timestamp.isoformat()

            if (
                reading.engine_temp_celsius is not None
                and reading.engine_temp_celsius > 110.0
            ):
                anomalies.append(
                    {
                        "timestamp": ts,
                        "sensor": "engine_temp",
                        "value": str(reading.engine_temp_celsius),
                        "issue": "Engine overtemperature",
                        "severity": "high",
                    }
                )

            if (
                reading.oil_pressure_psi is not None
                and reading.oil_pressure_psi < 20.0
            ):
                anomalies.append(
                    {
                        "timestamp": ts,
                        "sensor": "oil_pressure",
                        "value": str(reading.oil_pressure_psi),
                        "issue": "Low oil pressure",
                        "severity": "critical",
                    }
                )

            if (
                reading.battery_voltage is not None
                and reading.battery_voltage < 11.8
            ):
                anomalies.append(
                    {
                        "timestamp": ts,
                        "sensor": "battery_voltage",
                        "value": str(reading.battery_voltage),
                        "issue": "Low battery voltage",
                        "severity": "high",
                    }
                )

            if (
                reading.vibration_level is not None
                and reading.vibration_level > 8.0
            ):
                anomalies.append(
                    {
                        "timestamp": ts,
                        "sensor": "vibration",
                        "value": str(reading.vibration_level),
                        "issue": "Excessive vibration",
                        "severity": "medium",
                    }
                )

            if reading.tire_pressure_psi:
                for pos, psi in reading.tire_pressure_psi.items():
                    if psi < 28.0:
                        anomalies.append(
                            {
                                "timestamp": ts,
                                "sensor": f"tire_pressure_{pos}",
                                "value": str(psi),
                                "issue": f"Low tire pressure ({pos})",
                                "severity": "medium",
                            }
                        )

        return anomalies

    # --- Internal helpers ---

    def _process_vibration(
        self,
        readings: list[TelematicsReading],
        features: TelematicsFeatures,
    ) -> None:
        values = [r.vibration_level for r in readings if r.vibration_level is not None]
        if not values:
            return
        arr = np.array(values)
        features.vibration_mean = float(np.mean(arr))
        features.vibration_std = float(np.std(arr))
        features.vibration_max = float(np.max(arr))
        if len(arr) >= 2:
            x = np.arange(len(arr))
            slope = float(np.polyfit(x, arr, 1)[0])
            features.vibration_trend = slope

    def _process_engine_temp(
        self,
        readings: list[TelematicsReading],
        features: TelematicsFeatures,
    ) -> None:
        values = [
            r.engine_temp_celsius
            for r in readings
            if r.engine_temp_celsius is not None
        ]
        if not values:
            return
        arr = np.array(values)
        features.engine_temp_mean = float(np.mean(arr))
        features.engine_temp_std = float(np.std(arr))
        features.engine_temp_max = float(np.max(arr))
        features.overtemp_events = int(np.sum(arr > 110.0))

    def _process_oil_pressure(
        self,
        readings: list[TelematicsReading],
        features: TelematicsFeatures,
    ) -> None:
        values = [
            r.oil_pressure_psi for r in readings if r.oil_pressure_psi is not None
        ]
        if not values:
            return
        arr = np.array(values)
        features.oil_pressure_mean = float(np.mean(arr))
        features.oil_pressure_std = float(np.std(arr))
        features.oil_pressure_min = float(np.min(arr))
        features.low_pressure_events = int(np.sum(arr < 20.0))

    def _process_coolant(
        self,
        readings: list[TelematicsReading],
        features: TelematicsFeatures,
    ) -> None:
        values = [
            r.coolant_temp_celsius
            for r in readings
            if r.coolant_temp_celsius is not None
        ]
        if not values:
            return
        arr = np.array(values)
        features.coolant_temp_mean = float(np.mean(arr))
        features.coolant_temp_max = float(np.max(arr))

    def _process_battery(
        self,
        readings: list[TelematicsReading],
        features: TelematicsFeatures,
    ) -> None:
        values = [
            r.battery_voltage for r in readings if r.battery_voltage is not None
        ]
        if not values:
            return
        arr = np.array(values)
        features.battery_voltage_mean = float(np.mean(arr))
        features.battery_voltage_min = float(np.min(arr))
        features.low_voltage_events = int(np.sum(arr < 12.0))

    def _process_transmission(
        self,
        readings: list[TelematicsReading],
        features: TelematicsFeatures,
    ) -> None:
        values = [
            r.transmission_temp_celsius
            for r in readings
            if r.transmission_temp_celsius is not None
        ]
        if not values:
            return
        arr = np.array(values)
        features.trans_temp_mean = float(np.mean(arr))
        features.trans_temp_max = float(np.max(arr))
        features.trans_overtemp_events = int(np.sum(arr > 120.0))

    def _process_brakes(
        self,
        readings: list[TelematicsReading],
        features: TelematicsFeatures,
    ) -> None:
        values = [
            r.brake_pad_thickness_mm
            for r in readings
            if r.brake_pad_thickness_mm is not None
        ]
        if values:
            features.brake_pad_min_mm = float(np.min(values))

    def _process_tires(
        self,
        readings: list[TelematicsReading],
        features: TelematicsFeatures,
    ) -> None:
        all_pressures: list[float] = []
        low_events = 0
        for r in readings:
            if r.tire_pressure_psi:
                pressures = list(r.tire_pressure_psi.values())
                all_pressures.extend(pressures)
                low_events += sum(1 for p in pressures if p < 28.0)
        if all_pressures:
            features.tire_pressure_variance = float(np.var(all_pressures))
            features.tire_low_pressure_events = low_events

    @staticmethod
    def _range_score(
        value: float, low: float, high: float, margin: float
    ) -> float:
        """Score how well a value falls within a normal range.

        Returns 1.0 if within range, decreasing to 0.0 as it moves
        further outside by `margin` amount.
        """
        if value < 0.01 and low > 0:
            return 0.5  # No data
        if low <= value <= high:
            return 1.0
        if value < low:
            return max(0.0, 1.0 - (low - value) / margin)
        return max(0.0, 1.0 - (value - high) / margin)
