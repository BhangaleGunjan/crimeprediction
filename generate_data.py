import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_crime_data(num_records=1000, filename='sample_crime_data.csv'):
    """Generate sample crime data for the project.
    
    Returns:
        pd.DataFrame: The generated crime data as a pandas DataFrame.
    """

    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Define data parameters
    cities = ['Mumbai', 'Delhi', 'Bangalore', 'Kolkata', 'Chennai', 
              'Hyderabad', 'Pune', 'Ahmedabad', 'Jaipur', 'Surat']

    crime_types = ['Theft', 'Assault', 'Burglary', 'Drug Offense', 
                   'Fraud', 'Vandalism', 'Robbery', 'Motor Vehicle Theft']

    # Generate data
    data = []
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 31)

    for _ in range(num_records):
        record = {
            'City': random.choice(cities),
            'Crime_Type': random.choice(crime_types),
            'Date': start_date + timedelta(days=random.randint(0, (end_date - start_date).days)),
            'Month': random.randint(1, 12),
            'Year': random.randint(2020, 2024),
            'Population_Density': random.randint(1000, 10000),
            'Economic_Index': round(random.uniform(0.3, 0.9), 3),
            'Police_Stations': random.randint(5, 50),
            'Unemployment_Rate': round(random.uniform(3.0, 15.0), 2),
            'Crime_Count': random.randint(1, 20)
        }
        data.append(record)

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f'Generated {num_records} records and saved to "sample_crime_data.csv"')
    return df

if __name__ == "__main__":
    generate_crime_data()
