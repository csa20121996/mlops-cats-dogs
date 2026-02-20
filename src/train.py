import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.model import SimpleCNN
from src.dataset import get_dataloaders
from src.config import MODEL_PATH, EPOCHS, LEARNING_RATE, BATCH_SIZE, IMAGE_SIZE


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    y_true = []
    y_pred = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += float(loss.item())

        preds = torch.argmax(logits, dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    acc = (np.array(y_true) == np.array(y_pred)).mean() if len(y_true) else 0.0
    return total_loss, acc, y_true, y_pred


def train():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_dataloaders()

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    mlflow.set_experiment("cats_vs_dogs")

    train_losses = []
    val_losses = []

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "image_size": IMAGE_SIZE,
            "model": "SimpleCNN",
        })

        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item())

            val_loss, val_acc, _, _ = eval_epoch(model, val_loader, device)

            train_losses.append(running_loss)
            val_losses.append(val_loss)

            print(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"train_loss={running_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
            )

            mlflow.log_metric("train_loss", running_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        # Save model + log artifacts
        Path(os.path.dirname(MODEL_PATH)).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH, artifact_path="artifacts")
        mlflow.pytorch.log_model(model, artifact_path="model")

        # Final test evaluation + confusion matrix
        test_loss, test_acc, y_true, y_pred = eval_epoch(model, test_loader, device)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Cat", "Dog"])
        fig, ax = plt.subplots()
        disp.plot(ax=ax, colorbar=False)
        ax.set_title("Confusion Matrix (Test)")
        cm_path = "confusion_matrix.png"
        fig.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(cm_path, artifact_path="artifacts")

        # Loss curve
        fig2, ax2 = plt.subplots()
        ax2.plot(range(1, EPOCHS + 1), train_losses, label="train_loss")
        ax2.plot(range(1, EPOCHS + 1), val_losses, label="val_loss")
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("loss")
        ax2.set_title("Loss Curves")
        ax2.legend()
        lc_path = "loss_curves.png"
        fig2.savefig(lc_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        mlflow.log_artifact(lc_path, artifact_path="artifacts")

        print(f"Done. test_acc={test_acc:.4f} | model saved at {MODEL_PATH}")


if __name__ == "__main__":
    train()