import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def load_data(meteorite_file, weather_file):
    meteorite_data = pd.read_csv(r'Path of meteor_data.csv ')
    weather_data = pd.read_csv(r'Path of weather_data.csv')
    return meteorite_data, weather_data

def merge_datasets(meteorite_data, weather_data):
    if 'place' in meteorite_data.columns:
        meteorite_data.rename(columns={'place': 'city'}, inplace=True)

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


def calculate_visibility_score(data):
    
    data['visibility_score'] = 100 - (data['Data.Wind.Speed'] * 2)  
    data['visibility_score'] = data['visibility_score'].clip(lower=0)  
    return data


def train_model(data):
    features = data[['reclat', 'reclong', 'Data.Temperature.Avg Temp', 'date_ordinal']]
    target = data['visible']

    
    clf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5, min_samples_split=5)
    clf.fit(features, target)

    return clf


def predict_visibility(model, data):
    features = data[['reclat', 'reclong', 'Data.Temperature.Avg Temp', 'date_ordinal']]
    predictions = model.predict(features)
    data['prediction'] = predictions

    clear_cities = data[data['prediction'] == 1]
    clear_cities = clear_cities.drop_duplicates(subset=['Station.City', 'date_x'])

 
    clear_cities = clear_cities.rename(columns={'date_x': 'Date of Prediction'})


    clear_cities = clear_cities.drop(columns=['Data.Precipitation'])

 
    columns_to_keep = ['Station.City', 'Date of Prediction', 'visibility_score']
    top_cities = clear_cities[columns_to_keep].head(10)

  
    print("\nTop 10 cities with clear visibility for the meteor shower:")
    print(top_cities.to_string(index=False))


def main():
    meteorite_data, weather_data = load_data('meteorite-landings.csv', 'weather.csv')

    if meteorite_data is None or weather_data is None:
        print("Failed to load the data. Exiting.")
        return

    merged_data = merge_datasets(meteorite_data, weather_data)

    processed_data = process_dates(merged_data)
    processed_data = preprocess_data(processed_data)

    processed_data = calculate_visibility_score(processed_data)

    model = train_model(processed_data)
    predict_visibility(model, processed_data)


if __name__ == "__main__":
    main()
