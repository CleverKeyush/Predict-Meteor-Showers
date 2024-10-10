import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Function to load the data
def load_data(meteorite_file, weather_file):
    # Load meteorite data and weather data
    meteorite_data = pd.read_csv(r'C:\Users\keyus\OneDrive\Desktop\Predict-meteor-showers\meteorite-landings-with-dates.csv')
    weather_data = pd.read_csv(r'C:\Users\keyus\OneDrive\Desktop\Predict-meteor-showers\weather-with-dates.csv')
    return meteorite_data, weather_data

# Function to merge datasets on city
def merge_datasets(meteorite_data, weather_data):
    # Rename 'place' in meteorite data to 'city' (if applicable)
    if 'place' in meteorite_data.columns:
        meteorite_data.rename(columns={'place': 'city'}, inplace=True)

    # Check if 'city' columns exist in both datasets
    if 'city' not in meteorite_data.columns or 'Station.City' not in weather_data.columns:
        raise KeyError("City column not found in one of the datasets.")
    
    # Merge the datasets on 'city'
    merged_data = pd.merge(meteorite_data, weather_data, left_on='city', right_on='Station.City', how='inner')
    
    # Remove the debug print statement
    # print("Merged DataFrame columns:", merged_data.columns.tolist())
    
    return merged_data

# Function to preprocess the data and create a 'visible' column
def preprocess_data(merged_data):
    # Assume cloud cover < 20 is considered clear visibility
    merged_data['visible'] = merged_data['Data.Precipitation'].apply(lambda x: 1 if x < 20 else 0)
    return merged_data

# Function to convert dates to ordinal format for model training
def process_dates(data):
    # Convert the 'date_x' column to datetime
    data['date'] = pd.to_datetime(data['date_x'], errors='coerce')
    
    # Remove rows where date conversion failed
    data = data.dropna(subset=['date'])
    
    # Convert dates to ordinal format (necessary for machine learning models)
    data['date_ordinal'] = data['date'].apply(lambda x: x.toordinal())
    return data

# Function to train the model
def train_model(data):
    # Features are latitude, longitude, temperature, and date (encoded as ordinal)
    features = data[['reclat', 'reclong', 'Data.Temperature.Avg Temp', 'date_ordinal']]
    target = data['visible']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Initialize and train a RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    return clf

# Function to predict visibility for all cities
def predict_visibility(model, data):
    # Use the trained model to predict visibility
    features = data[['reclat', 'reclong', 'Data.Temperature.Avg Temp', 'date_ordinal']]
    predictions = model.predict(features)

    # Add predictions to the data as a new column
    data['prediction'] = predictions

    # Remove unnecessary columns
    columns_to_keep = ['Station.City', 'date_x', 'Data.Precipitation', 'prediction']
    data_filtered = data[columns_to_keep]

    # Filter for cities with clear visibility
    clear_cities = data_filtered[data_filtered['prediction'] == 1]

    # Remove duplicates based on Station.City and date_x
    clear_cities = clear_cities.drop_duplicates(subset=['Station.City', 'date_x'])

    # Sort the results by precipitation and take the top 20
    top_cities = clear_cities.sort_values(by='Data.Precipitation').head(20)

    # Output the top 20 cities with clear visibility
    print("\nTop 20 cities with clear visibility for the meteor shower:")
    print(top_cities)

# Main function to run all steps
def main():
    # Load the data
    meteorite_data, weather_data = load_data('meteorite-landings.csv', 'weather.csv')

    if meteorite_data is None or weather_data is None:
        print("Failed to load the data. Exiting.")
        return

    try:
        merged_data = merge_datasets(meteorite_data, weather_data)
    except KeyError as e:
        print(f"Error while merging datasets: {e}")
        return

    # Check for 'date_x' column in merged_data
    if 'date_x' not in merged_data.columns:
        print("Merged data does not contain 'date_x' column. Available columns are:", merged_data.columns.tolist())
        return

    processed_data = process_dates(merged_data)
    processed_data = preprocess_data(processed_data)

    model = train_model(processed_data)
    predict_visibility(model, processed_data)

# Run the main function
if __name__ == "__main__":
    main()
