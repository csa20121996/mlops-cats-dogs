import os
import time
from pathlib import Path
from typing import List, Tuple, Optional

import mlflow
import requests


# In-cluster defaults
API_URL = os.getenv("API_URL", "http://mlops-service:8000/predict")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://host.minikube.internal:5000")

# If you baked small samples into the image, keep them here.
# Expected structure:
#   TEST_DIR/Cat/*.png|jpg
#   TEST_DIR/Dog/*.png|jpg
TEST_DIR = Path(os.getenv("TEST_DIR", "/app/tests/smoke_data"))


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


def _predict_bytes(filename: str, content: bytes, content_type: str) -> str:
    resp = requests.post(API_URL, files={"file": (filename, content, content_type)}, timeout=30)
    resp.raise_for_status()
    return str(resp.json().get("label"))


def _predict_one(path: Path) -> str:
    with path.open("rb") as img:
        resp = requests.post(API_URL, files={"file": img}, timeout=30)
    resp.raise_for_status()
    return str(resp.json().get("label"))


def evaluate_once() -> Tuple[Optional[float], int]:
    """Returns (accuracy or None, total_samples)."""
    items = iter_labeled_images(TEST_DIR)

    # If no labeled samples exist, just validate predict endpoint is working
    if not items:
        import base64
        tiny_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
        )
        _ = _predict_bytes("smoke.png", tiny_png, "image/png")
        return None, 0

    correct = 0
    total = 0
    for true_label, path in items:
        pred_label = _predict_one(path)
        total += 1
        correct += int(pred_label == true_label)

    acc = correct / max(total, 1)
    return acc, total


def log_to_mlflow(acc: Optional[float], total: int) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    with mlflow.start_run(run_name="post_deploy_monitor"):
        mlflow.log_param("api_url", API_URL)
        mlflow.log_param("test_dir", str(TEST_DIR))

        if acc is None:
            mlflow.log_metric("post_deploy_predict_success", 1.0)
        else:
            mlflow.log_param("num_samples", total)
            mlflow.log_metric("post_deploy_accuracy", float(acc))


def main():
    sleep_s = int(os.getenv("SLEEP_SECONDS", "0"))

    def _run():
        acc, total = evaluate_once()
        log_to_mlflow(acc, total)
        if acc is None:
            print("Post-deploy monitor: predict endpoint OK (no labeled images found).")
        else:
            print(f"Post-deploy accuracy over {total} samples: {acc:.4f}")

    if sleep_s > 0:
        while True:
            try:
                _run()
            except Exception as e:
                print("monitor failed:", repr(e))
            time.sleep(sleep_s)
    else:
        _run()


if __name__ == "__main__":
    main()
