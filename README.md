# FleetGuard

Fleet Maintenance Predictor -- predict component failures and optimize service scheduling for vehicle fleets.

## Features

- **Failure Prediction**: Combines Weibull survival analysis with Gradient Boosting to predict when vehicle components will fail.
- **Component Database**: 20+ vehicle components (engine, transmission, brakes, tires, battery, alternator, etc.) with real-world failure distributions and maintenance intervals.
- **Maintenance Scheduling**: Optimizes service timing by grouping nearby services, balancing shop workload, and minimizing fleet downtime.
- **Telematics Processing**: Extracts maintenance-relevant features from sensor data (vibration, temperature, oil pressure) and detects anomalies.
- **Cost Analysis**: Compares ongoing maintenance costs against vehicle replacement using total-cost-of-ownership modeling.
- **Fleet Simulation**: Generates realistic fleet data for testing and demonstration.

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Run full demo with simulated fleet
fleetguard demo --vehicles 10

# Run failure predictions
fleetguard predict --vehicles 15

# Generate maintenance schedule
fleetguard schedule --vehicles 10 --horizon 90

# Cost analysis
fleetguard costs --vehicles 10

# List tracked components
fleetguard components
```

## Project Structure

```
src/fleetguard/
  cli.py              # Click CLI
  models.py            # Pydantic models (Vehicle, Component, ServiceRecord, FailurePrediction)
  simulator.py         # Fleet data simulator
  report.py            # Rich console reporting
  predictor/
    model.py           # FailurePredictor (Weibull survival + GradientBoosting)
    components.py      # ComponentDatabase with 20+ components
    scheduler.py       # MaintenanceScheduler (service grouping, workload balancing)
  fleet/
    vehicle.py         # VehicleManager (fleet tracking)
    telematics.py      # TelematicsProcessor (sensor feature extraction)
    cost.py            # CostAnalyzer (maintenance vs. replacement)
```

## Dependencies

- numpy, scipy -- numerical computation and survival distributions
- scikit-learn -- Gradient Boosting for failure prediction
- pydantic -- data validation and models
- click -- CLI framework
- rich -- terminal reporting

## Author

Mukunda Katta
