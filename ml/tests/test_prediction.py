from src.prediction.predictor import Predictor
from src.prediction.schemas import MachineInput


def test_predictor_returns_all_models():
    machine = MachineInput(
        type="M",
        air_temperature=300.0,
        process_temperature=310.0,
        rotational_speed=1500,
        torque=40.0,
        tool_wear=100.0,
    )

    predictor = Predictor()

    results = predictor.predict_all(machine.to_model_input())

    assert set(results.keys()) == {
        "logistic_regression",
        "decision_tree",
        "random_forest",
    }


def test_prediction_probabilities_are_valid():
    machine = MachineInput(
        type="M",
        air_temperature=300.0,
        process_temperature=310.0,
        rotational_speed=1500,
        torque=40.0,
        tool_wear=100.0,
    )

    predictor = Predictor()

    results = predictor.predict_all(machine.to_model_input())

    for result in results.values():
        assert 0.0 <= result["probability"] <= 1.0
        assert result["prediction"] in (0, 1)