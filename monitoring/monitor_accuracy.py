import os
import time
from pathlib import Path
from typing import List, Tuple

import mlflow
import requests


API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
TEST_DIR = Path(os.getenv("TEST_DIR", "data/test_monitor"))  # expected: TEST_DIR/Cat/*.jpg and TEST_DIR/Dog/*.jpg


def iter_labeled_images(base: Path) -> List[Tuple[str, Path]]:
    items: List[Tuple[str, Path]] = []
    for label in ["Cat", "Dog"]:
        folder = base / label
        if not folder.exists():
            continue
        for p in sorted(folder.glob("*")):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                items.append((label, p))
    return items


def evaluate() -> float:
    mlflow.set_tracking_uri(TRACKING_URI)

    items = iter_labeled_images(TEST_DIR)
    if not items:
        raise SystemExit(
            f"No images found under {TEST_DIR}. Create folders "
            f"{TEST_DIR}/Cat and {TEST_DIR}/Dog with a few labeled images."
        )

    correct = 0
    total = 0

    for true_label, path in items:
        with path.open("rb") as img:
            resp = requests.post(API_URL, files={"file": img}, timeout=30)
        resp.raise_for_status()
        pred_label = resp.json().get("label")

        total += 1
        correct += int(pred_label == true_label)

    acc = correct / max(total, 1)

    with mlflow.start_run(run_name="post_deploy_monitor"):
        mlflow.log_param("api_url", API_URL)
        mlflow.log_param("num_samples", total)
        mlflow.log_metric("post_deploy_accuracy", acc)

    print(f"Post-deploy accuracy over {total} samples: {acc:.4f}")
    return acc


if __name__ == "__main__":
    # Optional: run periodically (e.g., cron) by setting SLEEP_SECONDS
    sleep_s = int(os.getenv("SLEEP_SECONDS", "0"))
    if sleep_s > 0:
        while True:
            try:
                evaluate()
            except Exception as e:
                print("monitor failed:", repr(e))
            time.sleep(sleep_s)
    else:
        evaluate()