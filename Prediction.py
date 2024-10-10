import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_data(meteorite_file, weather_file):
    
    meteorite_data = pd.read_csv(r'C:\Users\keyus\OneDrive\Desktop\Predict-meteor-showers\meteorite-landings-with-dates.csv')
    weather_data = pd.read_csv(r'C:\Users\keyus\OneDrive\Desktop\Predict-meteor-showers\weather-with-dates.csv')
    return meteorite_data, weather_data


def merge_datasets(meteorite_data, weather_data):
   
    if 'place' in meteorite_data.columns:
        meteorite_data.rename(columns={'place': 'city'}, inplace=True)

    
    if 'city' not in meteorite_data.columns or 'Station.City' not in weather_data.columns:
        raise KeyError("City column not found in one of the datasets.")
    
  
    merged_data = pd.merge(meteorite_data, weather_data, left_on='city', right_on='Station.City', how='inner')
    
    
    
    return merged_data

def preprocess_data(merged_data):
    
    merged_data['visible'] = merged_data['Data.Precipitation'].apply(lambda x: 1 if x < 20 else 0)
    return merged_data

def process_dates(data):
    
    data['date'] = pd.to_datetime(data['date_x'], errors='coerce')
    
   
    data = data.dropna(subset=['date'])
    
    
    data['date_ordinal'] = data['date'].apply(lambda x: x.toordinal())
    return data


def train_model(data):
    
    features = data[['reclat', 'reclong', 'Data.Temperature.Avg Temp', 'date_ordinal']]
    target = data['visible']

    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

  
    y_pred = clf.predict(X_test)

    
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    return clf


def predict_visibility(model, data):
    
    features = data[['reclat', 'reclong', 'Data.Temperature.Avg Temp', 'date_ordinal']]
    predictions = model.predict(features)

    data['prediction'] = predictions

   
    columns_to_keep = ['Station.City', 'date_x', 'Data.Precipitation', 'prediction']
    data_filtered = data[columns_to_keep]

  
    clear_cities = data_filtered[data_filtered['prediction'] == 1]

   
    clear_cities = clear_cities.drop_duplicates(subset=['Station.City', 'date_x'])

    top_cities = clear_cities.sort_values(by='Data.Precipitation').head(20)

    
    print("\nTop 20 cities with clear visibility for the meteor shower:")
    print(top_cities)


def main():
  
    meteorite_data, weather_data = load_data('meteorite-landings.csv', 'weather.csv')

    if meteorite_data is None or weather_data is None:
        print("Failed to load the data. Exiting.")
        return

    try:
        merged_data = merge_datasets(meteorite_data, weather_data)
    except KeyError as e:
        print(f"Error while merging datasets: {e}")
        return

  
    if 'date_x' not in merged_data.columns:
        print("Merged data does not contain 'date_x' column. Available columns are:", merged_data.columns.tolist())
        return

    processed_data = process_dates(merged_data)
    processed_data = preprocess_data(processed_data)

    model = train_model(processed_data)
    predict_visibility(model, processed_data)

if __name__ == "__main__":
    main()
