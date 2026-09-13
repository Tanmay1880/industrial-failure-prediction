from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


VALID_MACHINE = {
    "type": "M",
    "air_temperature": 300.0,
    "process_temperature": 310.0,
    "rotational_speed": 1500,
    "torque": 40.0,
    "tool_wear": 100.0,
}


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "UP"}


def test_user_prediction():
    response = client.post("/predict", json=VALID_MACHINE)

    assert response.status_code == 200

    data = response.json()

    assert "status" in data
    assert "failure_probability" in data
    assert "models" not in data


def test_admin_prediction():
    response = client.post("/admin/predict", json=VALID_MACHINE)

    assert response.status_code == 200

    data = response.json()

    assert "status" in data
    assert "failure_probability" in data
    assert "models" in data

    assert set(data["models"].keys()) == {
        "logistic_regression",
        "decision_tree",
        "random_forest",
    }

    assert data["selected_model"] == "random_forest"
    assert data["threshold"] == 0.20


def test_invalid_machine_input():
    invalid_machine = {
        "type": "M",
        "air_temperature": "not-a-number",
        "process_temperature": 310.0,
        "rotational_speed": 1500,
        "torque": 40.0,
        "tool_wear": 100.0,
    }

    response = client.post("/predict", json=invalid_machine)

    assert response.status_code == 422