import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="Crime Rate Prediction",
    page_icon="🔍",
    layout="wide"
)

# Main title
st.title("🔍 Crime Rate Prediction System")
st.markdown("### Predict crime rates using Machine Learning")

# Load and cache data
@st.cache_data
def load_data():
    """Load the crime dataset"""
    try:
        # Try to load existing data
        data = pd.read_csv('sample_crime_data.csv')
        return data
    except FileNotFoundError:
        st.error("Dataset not found. Please upload sample_crime_data.csv file.")
        return None

# Train model function
@st.cache_resource
def train_model(data):
    """Train the machine learning model"""
    # Prepare features
    le_city = LabelEncoder()
    le_crime = LabelEncoder()

    data['City_encoded'] = le_city.fit_transform(data['City'])
    data['Crime_Type_encoded'] = le_crime.fit_transform(data['Crime_Type'])

    # Features for prediction
    features = ['City_encoded', 'Year', 'Month', 'Population_Density', 
                'Economic_Index', 'Police_Stations', 'Unemployment_Rate']
    target = 'Crime_Count'

    X = data[features]
    y = data[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, le_city, le_crime, accuracy

# Main app
def main():
    # Load data
    data = load_data()

    if data is not None:
        # Display data info
        st.sidebar.header("Dataset Information")
        st.sidebar.write(f"Total Records: {len(data)}")
        st.sidebar.write(f"Cities: {data['City'].nunique()}")
        st.sidebar.write(f"Crime Types: {data['Crime_Type'].nunique()}")

        # Train model
        with st.spinner("Training machine learning model..."):
            model, le_city, le_crime, accuracy = train_model(data)

        st.success(f"Model trained with {accuracy:.2%} accuracy!")

        # Create prediction interface
        st.header("🎯 Make a Prediction")

        col1, col2, col3 = st.columns(3)

        with col1:
            # City selection
            cities = data['City'].unique()
            selected_city = st.selectbox("Select City:", cities)

            # Year selection
            years = sorted(data['Year'].unique())
            selected_year = st.selectbox("Select Year:", years)

        with col2:
            # Crime type selection
            crime_types = data['Crime_Type'].unique()
            selected_crime_type = st.selectbox("Select Crime Type:", crime_types)

            # Month selection
            months = ['January', 'February', 'March', 'April', 'May', 'June',
                     'July', 'August', 'September', 'October', 'November', 'December']
            selected_month = st.selectbox("Select Month:", months)
            selected_month_num = months.index(selected_month) + 1

        with col3:
            # Additional features (with default values)
            population_density = st.slider("Population Density (per sq km):", 
                                         min_value=1000, max_value=10000, value=5000)
            economic_index = st.slider("Economic Index (0-1):", 
                                     min_value=0.0, max_value=1.0, value=0.6, step=0.1)
            police_stations = st.slider("Number of Police Stations:", 
                                      min_value=5, max_value=50, value=25)
            unemployment_rate = st.slider("Unemployment Rate (%):", 
                                        min_value=3.0, max_value=15.0, value=8.0)

        # Prediction button
        if st.button("🔮 Predict Crime Rate", type="primary"):
            try:
                # Prepare input data
                city_encoded = le_city.transform([selected_city])[0]

                input_data = np.array([[
                    city_encoded, selected_year, selected_month_num, 
                    population_density, economic_index, police_stations, unemployment_rate
                ]])

                # Make prediction
                prediction = model.predict(input_data)[0]

                # Display results
                st.header("📊 Prediction Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Selected City", selected_city)
                    st.metric("Selected Year", selected_year)

                with col2:
                    st.metric("Crime Type", selected_crime_type)
                    st.metric("Month", selected_month)

                with col3:
                    st.metric("Predicted Crime Count", f"{prediction}")

                    # Risk level
                    if prediction <= 5:
                        risk_level = "🟢 Low"
                    elif prediction <= 10:
                        risk_level = "🟡 Medium"
                    else:
                        risk_level = "🔴 High"

                    st.metric("Risk Level", risk_level)

                # Display additional insights
                st.header("📈 Additional Insights")

                # Filter data for the selected city and crime type
                city_data = data[
                    (data['City'] == selected_city) & 
                    (data['Crime_Type'] == selected_crime_type)
                ]

                if not city_data.empty:
                    avg_crime = city_data['Crime_Count'].mean()
                    st.write(f"Historical average for {selected_crime_type} in {selected_city}: {avg_crime:.1f}")

                    if prediction > avg_crime:
                        st.warning("⚠️ Predicted crime rate is above historical average!")
                    else:
                        st.success("✅ Predicted crime rate is below historical average.")

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

        # Display sample data
        if st.checkbox("Show Sample Data"):
            st.header("📋 Sample Dataset")
            st.dataframe(data.head(10))

            # Display basic statistics
            st.header("📊 Dataset Statistics")
            st.write(data.describe())

if __name__ == "__main__":
    main()
