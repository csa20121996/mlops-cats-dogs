python -m venv venv  ## use stable version of python (v3.10.x)
venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

model training 
    python -m src.train

docker pull jenkins/jenkins:lts


docker run -d \
  -p 8080:8080 \
  -p 50000:50000 \
  --name jenkins \
  jenkins/jenkins:lts

http://localhost:8080

to get jenkins password
docker exec -it jenkins cat /var/jenkins_home/secrets/initialAdminPassword
b1c9deda6c434f078b75230f51b65b53
    created account there 
        username: chandrashekar
        pass: Admin@123
        fullname: CHANDRA SHEKAR A
        EmailAddress: 2024ab05298@wilp.bits-pilani.ac.in
            URL: http://localhost:8080/

docker compose up -d


minikube start --driver=docker

kubectl apply -f k8s-deployment.yaml
kubectl apply -f k8s-service.yaml


start minikube server
    minikube service mlops-service

start mlflow
    python -m mlflow ui


prometheus
    http://localhost:9090/query
grafana
    http://localhost:3000/

