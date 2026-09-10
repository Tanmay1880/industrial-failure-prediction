from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.utils.config import MODEL_CONFIG


def create_model(model_name):
    if model_name == "logistic_regression":
        return LogisticRegression(
            **MODEL_CONFIG["logistic_regression"]
        )

    if model_name == "decision_tree":
        return DecisionTreeClassifier(
            **MODEL_CONFIG["decision_tree"]
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            **MODEL_CONFIG["random_forest"]
        )

    raise ValueError(f"Unknown model: {model_name}")


def create_all_models():
    return {
        "logistic_regression": create_model("logistic_regression"),
        "decision_tree": create_model("decision_tree"),
        "random_forest": create_model("random_forest"),
    }