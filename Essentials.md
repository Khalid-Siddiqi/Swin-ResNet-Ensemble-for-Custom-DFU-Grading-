# Step 1: Create a new virtual environment named 'venv'
python -m venv venv

# Step 2: Activate the environment
# On Windows:
# venv\Scripts\activate

# Step 3: Upgrade pip and install common packages
pip install --upgrade pip
pip install numpy pandas matplotlib jupyter

# Step 4: Freeze dependencies
pip freeze > requirements.txt

# Step 5: Print VS Code instructions
echo "✅ Virtual environment 'venv' created."
echo "👉 Open Command Palette in VS Code (Ctrl+Shift+P), then run: Python: Select Interpreter"
echo "👉 Choose the one from './venv' folder"
