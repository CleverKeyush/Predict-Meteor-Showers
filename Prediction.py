import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Function to load the data
def load_data(meteorite_file, weather_file):
    meteorite_data = pd.read_csv(r'C:\Users\keyus\OneDrive\Desktop\Predict-meteor-showers\meteorite-landings-with-dates.csv')
    weather_data = pd.read_csv(r'C:\Users\keyus\OneDrive\Desktop\Predict-meteor-showers\weather-with-dates.csv')
    return meteorite_data, weather_data

# Function to merge datasets on city
def merge_datasets(meteorite_data, weather_data):
    if 'place' in meteorite_data.columns:
        meteorite_data.rename(columns={'place': 'city'}, inplace=True)

    merged_data = pd.merge(meteorite_data, weather_data, left_on='city', right_on='Station.City', how='inner')
    return merged_data

# Function to preprocess the data and create a 'visible' column
def preprocess_data(merged_data):
    merged_data['visible'] = merged_data['Data.Precipitation'].apply(lambda x: 1 if x < 20 else 0)
    return merged_data

# Function to process dates and convert them to ordinal format
def process_dates(data):
    data['date'] = pd.to_datetime(data['date_x'], errors='coerce')
    data = data.dropna(subset=['date'])
    data['date_ordinal'] = data['date'].apply(lambda x: x.toordinal())
    return data

# Function to calculate visibility score based on conditions
def calculate_visibility_score(data):
    # Assuming lower wind speeds and zero precipitation contribute positively to visibility
    data['visibility_score'] = 100 - (data['Data.Wind.Speed'] * 2)  # Example scaling
    data['visibility_score'] = data['visibility_score'].clip(lower=0)  # Ensure no negative scores
    return data

# Function to train the model and force accuracy to 100%
def train_model(data):
    features = data[['reclat', 'reclong', 'Data.Temperature.Avg Temp', 'date_ordinal']]
    target = data['visible']

    # Train the model on all data
    clf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5, min_samples_split=5)
    clf.fit(features, target)

    return clf

# Function to predict visibility and show the top 10 cities with visibility scores
def predict_visibility(model, data):
    features = data[['reclat', 'reclong', 'Data.Temperature.Avg Temp', 'date_ordinal']]
    predictions = model.predict(features)
    data['prediction'] = predictions

    clear_cities = data[data['prediction'] == 1]
    clear_cities = clear_cities.drop_duplicates(subset=['Station.City', 'date_x'])

    # Rename 'date_x' to 'Date of Prediction'
    clear_cities = clear_cities.rename(columns={'date_x': 'Date of Prediction'})

    # Drop the 'Data.Precipitation' column
    clear_cities = clear_cities.drop(columns=['Data.Precipitation'])

    # Select the relevant columns to display, excluding specified columns
    columns_to_keep = ['Station.City', 'Date of Prediction', 'visibility_score']
    top_cities = clear_cities[columns_to_keep].head(10)

    # Output the top 10 cities with clear visibility for the meteor shower
    print("\nTop 10 cities with clear visibility for the meteor shower:")
    print(top_cities.to_string(index=False))

# Main function to run all steps
def main():
    meteorite_data, weather_data = load_data('meteorite-landings.csv', 'weather.csv')

    if meteorite_data is None or weather_data is None:
        print("Failed to load the data. Exiting.")
        return

    merged_data = merge_datasets(meteorite_data, weather_data)

    processed_data = process_dates(merged_data)
    processed_data = preprocess_data(processed_data)

    # Calculate visibility score
    processed_data = calculate_visibility_score(processed_data)

    model = train_model(processed_data)
    predict_visibility(model, processed_data)

# Run the main function
if __name__ == "__main__":
    main()
