# MLOps Assignment 2 group 121 — Cats vs Dogs

This repository contains:

- **Training (PyTorch)** with **MLflow tracking**
- **FastAPI** inference service with:
  - `/health`
  - `/predict` (image upload)
  - `/metrics` (Prometheus and Grafana)
- **CI** with GitHub Actions (pytest + Docker build + push to GHCR)
- **CD** to **Minikube** using a self-hosted runner (Windows CMD) + smoke test (health + predict)
- Monitoring:
  - **Kubernetes**: Prometheus + Grafana manifests in `k8s/monitoring/`

---

## 1) Local setup

```bash
python -m venv venv
# Windows
source venv\Scripts\activate
# Linux/macOS
# source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## 1.1) Using Docker Desktop, minikube and self-host for deployments, all neccessary things needs to be pre-installed

---

## 2) Train + track with MLflow
## 2.1) Training model locally
```bash
python -m src.train
```
## 2.2) Starting mlflow
```bash
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri file:./mlruns --default-artifact-root ./mlruns
```

## 3) Starting minikube locally
```bash
minikube start --driver=docker
```

## 3.1) verifying minikube by kubectl cmds
```bash
kubectl get pods
kubectl get nodes
```

## 4) setting up self-host for continuous deployment
```bash
<link to document of setting self-host>
```

## 4.1) starting self-host runner locally
```bash
run.cmd  or
./run.cmd
```

## 5) bringing up monitoring which is installed/containarized
```bash
kubectl apply -f k8s/monitoring/
```

## 5) enabling / getting minikube ip to monitor / access from local system
```bash
minikube service mlops-service
```

## 5.1) application deployed will be available from the tunnel url provided by above cmd
url(http://127.0.01:port)/docs

Endpoints:
- `GET /health`
- `POST /predict`
- `GET /metrics`

## 5.2) Accessing grafana and prometheus from local 
```bash
Access:
- Prometheus: `minikube service -n monitoring prometheus`
- Grafana: `minikube service -n monitoring grafana` (admin/admin)
```


## CI flow: when any push from code, CI from github actions will be triggered and executed
## CD flow: when any pull request merged to mainline, CD actions from github will be executed and self-hosted localhost/ local system will pick the job and completes the action

