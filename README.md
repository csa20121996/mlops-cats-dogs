python -m venv venv  ## use stable version of python (v3.10.x)
venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

model training 
    python -m src.train

