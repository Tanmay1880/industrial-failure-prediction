from pathlib import Path

from sklearn.model_selection import train_test_split

from src.data.load_dataset import load_dataset
from src.models.model_factory import create_all_models
from src.preprocessing.pipeline import create_pipeline
from src.utils.config import FEATURES, TARGET, RANDOM_STATE


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"
MODEL_DIRECTORY = PROJECT_ROOT / "models"


def train_models():
    print("Loading dataset...")

    df = load_dataset(DATASET_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"Development rows: {len(X_dev)}")
    print(f"Final test rows : {len(X_test)}")

    models = create_all_models()

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        pipeline = create_pipeline(model)

        pipeline.fit(X_dev, y_dev)

        model_path = MODEL_DIRECTORY / f"{model_name}.joblib"

        from joblib import dump

        dump(pipeline, model_path)

        print(f"Saved: {model_path}")

    print("\nAll models trained successfully.")


if __name__ == "__main__":
    train_models()