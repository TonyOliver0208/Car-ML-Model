# Car-ML-Model

## 🛠️ Setup Instructions

### (Optional) Create and activate a virtual environment

```
python -m venv venv
source venv/bin/activate   # macOS/Linux
.\venv\Scripts\activate    # Windows
```

### Install required Python packages

```
pip install -r requirements.txt
```

### Start the ML server

```
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```
