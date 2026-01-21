# Crime Rate Prediction Project

A simple web application built with Streamlit that predicts crime rates using machine learning.

## Features
- Interactive web interface
- City, crime type, and year selection
- Machine learning predictions using Random Forest
- Real-time prediction results
- Historical data analysis

## Installation Steps

### 1. Install Python (if not already installed)
- Download Python 3.8+ from https://python.org
- Make sure to check "Add Python to PATH" during installation

### 2. Install VS Code (if not already installed)
- Download from https://code.visualstudio.com/
- Install Python extension from VS Code marketplace

### 3. Set up the project
1. Create a new folder for your project
2. Open the folder in VS Code
3. Open terminal in VS Code (Terminal > New Terminal)

### 4. Create virtual environment
```bash
python -m venv crime_prediction_env

# Activate the virtual environment:
# On Windows:
crime_prediction_env\Scripts\activate
# On macOS/Linux:
source crime_prediction_env/bin/activate
```

### 5. Install required packages
```bash
pip install -r requirements.txt
```

### 6. Generate sample data
```bash
python generate_data.py
```

### 7. Run the application
```bash
streamlit run app.py
```

The app will open in your web browser at http://localhost:8501

## Project Structure
```
crime_prediction_project/
│
├── app.py                 # Main Streamlit application
├── generate_data.py       # Script to generate sample data
├── requirements.txt       # Python dependencies
├── sample_crime_data.csv  # Generated dataset
└── README.md             # This file
```

## Usage
1. Select a city from the dropdown
2. Choose crime type and year
3. Adjust the sliders for additional parameters
4. Click "Predict Crime Rate" to get the prediction
5. View the results and insights

## Model Information
- Algorithm: Random Forest Classifier
- Features: City, Year, Month, Population Density, Economic Index, Police Stations, Unemployment Rate
- Target: Crime Count

## Note
This project uses synthetic data for demonstration purposes. 
For real-world applications, use actual crime datasets from official sources.
