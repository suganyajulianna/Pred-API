import requests
import pandas as pd
import time
import os

# Flask API URL
api_url = 'http://127.0.0.1:5001/predict'

import pandas as pd
import os

def read_new_data(file_path):
    """Read the latest one second of data from the CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    data = pd.read_csv(file_path, delimiter=',')  # Adjust delimiter if needed
    
    # Check if data is available
    if data.empty:
        return pd.DataFrame()  # Return an empty DataFrame if no data is present

    # Get the latest timestamp
    latest_timestamp = pd.to_datetime(data['Timestamp']).max()

    # Get the data for the last second
    one_minute_ago = latest_timestamp - pd.Timedelta(minutes=1)
    new_data = data[pd.to_datetime(data['Timestamp']) > one_minute_ago]
    
    return new_data


def send_data_to_flask(data):
    try:
        data_csv = data.to_csv(index=False, sep=',')
        response = requests.post(api_url, files={'file': ('new_data.csv', data_csv, 'text/csv')})

        # Print full response for debugging
        print("Response Status Code:", response.status_code)
        print("Response Content:", response.content.decode('utf-8'))

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code} - {response.content.decode('utf-8')}")
            return {'error': 'Failed to get response from API'}
    except Exception as e:
        print(f"Exception occurred: {e}")
        return {'error': 'Exception occurred'}


file_path = r"C:\Users\Jsuga\Desktop\live_data.txt"

try:
    while True:
        new_data = read_new_data(file_path)
        
        if not new_data.empty:
            new_data = new_data.dropna()  # Remove rows with NaN values
            
            if not new_data.empty:
                result = send_data_to_flask(new_data)

                if 'error' in result:
                    print(f"Error: {result['error']}")
                else:
                    # Print the prediction along with the timestamps
                    for index, row in new_data.iterrows():
                        print(f"Timestamp: {row['Timestamp']}, Prediction: {result['Predictions'][index]}")
            else:
                print("No new data after dropping NaNs.")
        else:
            print("No new data.")
        
        time.sleep(59)
except KeyboardInterrupt:
    print("Stopped by user")
except Exception as e:
    print(f"An error occurred: {e}")
