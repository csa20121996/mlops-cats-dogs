# MLOps Assignment 2 (Group 121) — Cats vs Dogs

This project implements an end-to-end MLOps workflow for a Cats vs Dogs classifier:

- **Training (PyTorch)** with **MLflow** experiment tracking (metrics + artifacts)
- **Inference API (FastAPI)**:
  - `GET /health`
  - `POST /predict` (image upload)
  - `GET /metrics` (Prometheus format)
- **CI (GitHub Actions)**: tests + Docker build + push to GHCR
- **CD (GitHub Actions + Self-hosted Runner on Windows)**: deploy to **Minikube** and run smoke tests
- **Monitoring (Kubernetes)**:
  - **Prometheus** scrapes `/metrics`
  - **Grafana** dashboard for request counts/latency
- **Post-deploy monitoring (Kubernetes CronJob)**: runs prediction checks periodically and logs to MLflow

---

## 1) Prerequisites (Local machine)

### 1.1 Tools required
- Python 3.10+
- Docker Desktop
- Minikube (Docker driver)
- kubectl
- Git
- GitHub Self-hosted Runner configured on the deployment machine (Windows)

> Note (Windows + Minikube Docker driver): services are typically accessed using `minikube service ...` tunnel URLs.

---

## 2) Local Python setup

```bash
python -m venv venv

# Windows (Git Bash / PowerShell may differ)
source venv/Scripts/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3) MLflow Tracking (Local)

### 3.1 Start MLflow Tracking Server
Run this from the repo root (where `mlruns/` exists):

```bash
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri file:./mlruns \
  --default-artifact-root ./mlruns
```

Open MLflow UI:
- http://127.0.0.1:5000

> This configuration ensures both local training runs and Kubernetes CronJob runs can be written into the same local `mlruns/` folder (via the MLflow server).

### 3.2 Train the model locally (Dataset required locally)
```bash
python -m src.train
```

Training logs to MLflow:
- metrics (loss/accuracy)
- artifacts (loss curve, confusion matrix)
- model artifact

> Training is run locally because the full dataset is not committed into the Git repository.

---

## 4) Build & Run the API locally (optional)

```bash
uvicorn src.inference.app:app --host 0.0.0.0 --port 8000
```

API docs:
- http://127.0.0.1:8000/docs

---

## 5) Minikube (Local Kubernetes)

### 5.1 Start Minikube
```bash
minikube start --driver=docker
```

Verify:
```bash
kubectl get nodes
kubectl get pods
```

### 5.2 Deploy application manifests
```bash
kubectl apply -f k8s/
```

### 5.3 Access the application (Windows tunnel)
```bash
minikube service mlops-service
```

This prints a tunnel URL like:
- `http://127.0.0.1:<PORT>/docs`

Endpoints:
- `GET  /health`
- `POST /predict`
- `GET  /metrics`

---

## 6) Monitoring in Kubernetes (Prometheus + Grafana)

### 6.1 Deploy monitoring stack
```bash
kubectl apply -f k8s/monitoring/
```

Check:
```bash
kubectl get pods -n monitoring
kubectl get svc  -n monitoring
```

### 6.2 Access Prometheus and Grafana (Windows tunnel)
```bash
minikube service -n monitoring prometheus --url
minikube service -n monitoring grafana --url
```

Grafana default login:
- user: `admin`
- pass: `admin`

### 6.3 Example Prometheus query
In Prometheus UI, run:
- Total prediction requests:
  ```
  sum(request_count_total)
  ```
- Per endpoint:
  ```
  sum by (endpoint) (request_count_total)
  ```

---

## 7) Post-deploy Monitoring (CronJob → MLflow)

### 7.1 What it does
A Kubernetes **CronJob** runs every 5 minutes:
- calls the deployed `/predict` endpoint
- logs a metric to MLflow:
  - `post_deploy_accuracy` if labeled test images exist
  - else `post_deploy_predict_success = 1` (basic verification)

### 7.2 Important note (MLflow reachability)
CronJob logs to the **host MLflow server** using:
- `MLFLOW_TRACKING_URI=http://host.minikube.internal:5000`

This requires MLflow server to be running locally (Section 3.1).

### 7.3 Apply CronJob
```bash
kubectl apply -f k8s/monitoring/25-accuracy-script-configmap.yaml
kubectl apply -f k8s/monitoring/30-accuracy-cronjob.yaml
```

Run once immediately:
```bash
kubectl -n monitoring delete jobs --all
kubectl -n monitoring create job --from=cronjob/post-deploy-accuracy post-deploy-accuracy-now
kubectl -n monitoring logs job/post-deploy-accuracy-now
```

Then verify MLflow UI for new runs:
- http://127.0.0.1:5000

---

## 8) CI Pipeline (GitHub Actions)

CI runs on every push to any branch:
- installs dependencies
- runs tests (`pytest`)
- builds Docker image
- pushes to GitHub Container Registry (GHCR)

---

## 9) CD Pipeline (GitHub Actions + Self-hosted Runner)

CD runs on the self-hosted runner:
- deploys the latest image to Minikube
- waits for rollout
- performs smoke test (`/health` + `/predict`)

### 9.1 Configure self-hosted runner (Windows)
Follow GitHub’s official instructions:
- https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners

Start runner:
```bat
run.cmd
```

---

## 10) Troubleshooting quick notes

- If services don’t open using Minikube IP + NodePort on Windows, always use:
  - `minikube service <svc> --url`
- If Grafana “Total Requests” is empty, ensure dashboard uses:
  - `request_count_total` (not `request_count`)
- If Prometheus shows no metrics:
  - verify `/metrics` returns Prometheus format (text/plain)
  - verify Prometheus target is UP

---

## Repository Structure (high level)

- `src/` — training + inference application code
- `k8s/` — Kubernetes manifests for the app
- `k8s/monitoring/` — Prometheus + Grafana + CronJob manifests
- `monitoring/` — monitoring scripts & configs (Grafana dashboard JSON, Prometheus scrape config)
- `models/` — trained model artifacts used by inference
- `tests/` — unit tests + minimal smoke resources
