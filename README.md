<!-- Download Dataset Command -->
python backend/utils/download_dataset.py


<!-- Venv Creation -->
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch torch-geometric networkx numpy


<!-- Later, your React.js frontend can fetch from these endpoints like: -->

const response = await fetch("http://127.0.0.1:8000/anomalies?top_n=50");
const data = await response.json();
console.log(data);

<!-- Uvicorn -->
uvicorn backend.app.services.graph_api:app --reload --host 0.0.0.0 --port 8000
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/redoc


<!-- Requiremnet.txt run -->
pip install -r requirements.tx

<!-- And All necesary modules to Requirement.txt -->
pip freeze > requirements.txt
