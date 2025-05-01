### Step 1: Create a new virtual environment named 'venv'
python -m venv venv

### Step 2: Activate the environment
 On Windows:
 venv\Scripts\activate

### Step 3: Upgrade pip and install common packages
pip install --upgrade pip
pip install numpy pandas matplotlib jupyter
pip install fastapi uvicorn torch torchvision timm pillow ultralytics

### Step 4: Freeze dependencies
pip freeze > requirements.txt

### Step 5: Print VS Code instructions
echo "✅ Virtual environment 'venv' created."
echo "👉 Open Command Palette in VS Code (Ctrl+Shift+P), then run: Python: Select Interpreter"
echo "👉 Choose the one from './venv' folder"

## How to run
uvicorn main:app --reload

## How TO USE Postman
Endpoint: POST http://127.0.0.1:8000/predict/
Body type: form-data

Key: file (type: File)

Value: upload your image
