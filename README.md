# MLOps Assignment 2 — Cats vs Dogs

This repository contains:

- **Training (PyTorch)** with **MLflow tracking**
- **FastAPI** inference service with:
  - `/health`
  - `/predict` (image upload)
  - `/metrics` (Prometheus)
- **CI** with GitHub Actions (pytest + Docker build + push to GHCR)
- **CD** to **Minikube** using a self-hosted runner (Windows CMD) + smoke test (health + predict)
- Monitoring:
  - **Kubernetes (recommended)**: Prometheus + Grafana manifests in `k8s/monitoring/`
  - **docker-compose (dev only)**: `docker-compose.yml`

> Jenkins is **not used** (GitHub Actions is used for CI/CD). Any Jenkins files/passwords have been removed.

---

## 1) Local setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
# source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2) Train + track with MLflow

### A) Local MLflow UI (file store)

```bash
python -m src.train
python -m mlflow ui
```

This uses the local `./mlruns` folder by default.

### B) Recommended: MLflow tracking server for Minikube jobs

If you want **Minikube CronJobs** (post-deploy monitor) to log to MLflow, run MLflow as a **server** on your host:

```bash
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

In Kubernetes, set:
- `MLFLOW_TRACKING_URI=http://host.minikube.internal:5000`

---

## 3) Run the inference API locally

```bash
uvicorn src.inference.app:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health`
- `POST /predict`
- `GET /metrics`

---

## 4) Minikube deployment (CD target)

Start Minikube:

```bash
minikube start --driver=docker
```

Deploy app:

```bash
kubectl apply -f k8s/k8s-deployment.yaml
kubectl apply -f k8s/k8s-service.yaml
```

Open service:

```bash
minikube service mlops-service
```

---

## 5) Monitoring

### Option A — Kubernetes (recommended for CD)

```bash
kubectl apply -f k8s/monitoring/00-namespace.yaml
kubectl apply -f k8s/monitoring/10-prometheus-configmap.yaml
kubectl apply -f k8s/monitoring/11-prometheus-deployment.yaml
kubectl apply -f k8s/monitoring/12-prometheus-service.yaml

kubectl apply -f k8s/monitoring/20-grafana-datasource-configmap.yaml
kubectl apply -f k8s/monitoring/21-grafana-dashboard-provisioning-configmap.yaml
kubectl apply -f k8s/monitoring/22-grafana-dashboard-configmap.yaml
kubectl apply -f k8s/monitoring/23-grafana-deployment.yaml
kubectl apply -f k8s/monitoring/24-grafana-service.yaml
```

Access:
- Prometheus: `minikube service -n monitoring prometheus`
- Grafana: `minikube service -n monitoring grafana` (admin/admin)

Prometheus scrapes:
- `mlops-service.default.svc.cluster.local:8000/metrics`

### Option B — docker-compose (dev only)

```bash
docker compose up -d
```

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

---

## 6) Post-deploy monitoring (accuracy / smoke)

A tiny baked-in sample set is included for monitoring under:

`tests/smoke_data/{Cat,Dog}/*.png` (10 images each)

Run locally:

```bash
set API_URL=http://localhost:8000/predict
set MLFLOW_TRACKING_URI=http://localhost:5000
python monitoring/monitor_accuracy.py
```

Kubernetes CronJob (optional):

```bash
kubectl apply -f k8s/monitoring/30-accuracy-cronjob.yaml
```

Metrics logged:
- `post_deploy_accuracy` if labeled images exist
- otherwise `post_deploy_predict_success` in smoke mode

---

## 7) Notes

- `src/train.py` logs:
  - `train_loss`, `val_loss`, `val_accuracy`, `test_accuracy`
  - artifacts: confusion matrix + loss curves
- `src/inference/app.py` loads the model via `state_dict` to match training export.
- `.dockerignore` excludes `data/` and `mlruns/` to keep image builds clean.


minikube service -n monitoring grafana --url