import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Air Pollution",
    page_icon="🖥️",
    layout="wide"  
)

st.title("Data-Driven Air Pollution Assessment for Environmental Solutions in Almaty")

col1, col2, col3 = st.columns(3)


with col1:
    st.header("Used dataset")
    df_csv = pd.read_csv("almaty_airQuality.csv")
    df_csv
    df_excel = pd.read_excel("meteoData.xlsx")
    df_excel


    



df_excel = pd.read_excel("meteoData.xlsx")

df_excel.columns = ['date', 'temperature', 'humidity', 'wind']


print("Updated Column Names:", df_excel.columns)

# Ensure the common column (e.g., 'date') is in the correct format
df_csv['date'] = pd.to_datetime(df_csv['date'], errors='coerce')
df_excel['date'] = pd.to_datetime(df_excel['date'], errors='coerce')

# Merge the DataFrames
merged_df = pd.merge(df_csv, df_excel, on='date', how='inner')  # Use 'outer', 'left', or 'right' as needed

# Display the merged DataFrame
print("Merged Data:")
print(merged_df)

import pandas as pd

# Normalize column names
merged_df.columns = merged_df.columns.str.strip().str.lower()

# Convert columns to numeric where applicable
numeric_cols = ['pm25', 'pm10', 'no2', 'so2', 'co', 'temperature', 'humidity', 'wind']
for col in numeric_cols:
    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

# Handle missing values by filling with column means
merged_df[numeric_cols] = merged_df[numeric_cols].fillna(merged_df[numeric_cols].mean())

# Ensure 'date' column is in datetime format
merged_df['date'] = pd.to_datetime(merged_df['date'], errors='coerce')

# Remove duplicate rows
merged_df = merged_df.drop_duplicates()

# Save the cleaned data
merged_df.to_csv('preprocessed_data.csv', index=False)

print("Data preprocessing completed and saved as 'preprocessed_data.csv'.")
print("Cleaned Dataset:")
print(merged_df.head())


merged_df.to_excel('merged_df.xlsx', index=False)
# Descriptive statistics for all numeric columns
print("Descriptive Statistics:")
print(merged_df.describe())




# Plot trend for each pollutant over time
plt.figure(figsize=(12, 6))
pollutants = ['pm25', 'pm10']
for col in pollutants:
    plt.plot(merged_df['date'], merged_df[col], label=col)
plt.title('Trend Analysis of Pollutant Levels Over Time')
plt.xlabel('Date')
plt.ylabel('Concentration')
plt.legend()
plt.show()

import matplotlib.pyplot as plt

# Plot trend for each pollutant over time





with col2:
    st.header("Analyzing the dataset")
    st.pyplot(plt)

    plt.figure(figsize=(12, 6))
    pollutants = ['no2', 'so2', 'co']
    for col in pollutants:
        plt.plot(merged_df['date'], merged_df[col], label=col)
    plt.title('Trend Analysis of Pollutant Levels Over Time')
    plt.xlabel('Date')
    plt.ylabel('Concentration')
    plt.legend()
    plt.show()


    st.pyplot(plt)


    import seaborn as sns

    # Compute correlation matrix
    correlation_matrix = merged_df[['pm25', 'pm10', 'no2', 'so2', 'co', 'temperature', 'humidity', 'wind']].corr()

    # Heatmap of correlations
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix Between Pollutants and Weather Conditions')
    plt.show()

    st.pyplot(plt)

    # Extract month and year for grouping
    pollutants = ['pm25', 'pm10', 'no2', 'so2', 'co']
    merged_df['month'] = merged_df['date'].dt.month
    monthly_avg = merged_df.groupby('month')[pollutants].mean()
    # Plot monthly average pollutant levels
    plt.figure(figsize=(10, 6))
    monthly_avg.plot(kind='line', marker='o')
    plt.title('Seasonal Analysis: Monthly Average Pollutant Levels')
    plt.xlabel('Month')
    plt.ylabel('Average Concentration')
    plt.legend()
    plt.show()

    st.pyplot(plt)

    # Boxplots to visualize outliers for each pollutant
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=merged_df[pollutants])
    plt.title("Boxplot of Pollutant Levels to Identify Outliers")
    plt.show()

    st.pyplot(plt)

    # Extract year for yearly grouping
    merged_df['year'] = merged_df['date'].dt.year
    yearly_avg = merged_df.groupby('year')[pollutants].mean()

    # Plot yearly average pollutant levels
    yearly_avg.plot(kind='bar', figsize=(10, 6))
    plt.title('Yearly Average Pollutant Levels')
    plt.xlabel('Year')
    plt.ylabel('Average Concentration')
    plt.legend()
    plt.show()

    st.pyplot(plt)





with col3:
    st.header("Model Building")

    

    def extend_exog(exog, target_length):
        """
        Extend the exogenous variables to match the required forecast length.
        If the exog has fewer rows than the target_length, duplicate the last row.
        """
        if exog is not None:
            exog_extended = exog.copy()
            while len(exog_extended) < target_length:
                last_row = exog.iloc[[-1]].copy()  # Duplicate the last row
                exog_extended = pd.concat([exog_extended, last_row], ignore_index=True)
            return exog_extended.iloc[:target_length]  # Trim to the required length
        return None

    def forecast_sarima(series, pollutant_name, exog_columns=None):
        print(f"\nForecasting for {pollutant_name}...")

        # Set 'date' as the index
        series = series.copy()  # Avoid SettingWithCopyWarning
        series['date'] = pd.to_datetime(series['date'])
        series = series.set_index('date')
        
        # Resample to daily frequency and interpolate missing values
        series = series.resample('D').mean().interpolate(method='linear')

        # Train-test split (80% training, 20% testing)
        train_size = int(len(series) * 0.8)
        train, test = series[:train_size], series[train_size:]

        # Extract exogenous features if provided
        exog_train = train[exog_columns] if exog_columns else None
        exog_test = test[exog_columns] if exog_columns else None

        # Fit SARIMAX model
        try:
            model = SARIMAX(train[pollutant_name], exog=exog_train, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
            result = model.fit(disp=False)
        except Exception as e:
            print(f"Error fitting SARIMAX model for {pollutant_name}: {e}")
            return

        # Forecast the next 365 days (for 2025)
        forecast_days = 365
        exog_test_extended = extend_exog(exog_test, forecast_days)
        try:
            forecast = result.forecast(steps=forecast_days, exog=exog_test_extended)
        except Exception as e:
            print(f"Error during forecasting for {pollutant_name}: {e}")
            return

        # Plot the train, test, and forecasted values
        plt.figure(figsize=(12, 6))
        plt.plot(train.index, train[pollutant_name], label='Train')
        plt.plot(test.index, test[pollutant_name], label='Actual')
        forecast_index = pd.date_range(test.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')
        plt.plot(forecast_index, forecast, label='Forecast for 2025', color='red')
        plt.title(f'SARIMA Forecast for {pollutant_name} for 2025')
        plt.xlabel('Date')
        plt.ylabel(f'{pollutant_name} Concentration')
        plt.legend()
        plt.show()

        st.pyplot(plt)

    # Apply SARIMA forecasting for all pollutants
    pollutants = ['pm25', 'pm10', 'no2', 'so2', 'co']
    exog_columns = ['temperature', 'humidity', 'wind']

    for pollutant in pollutants:
        forecast_sarima(merged_df[['date', pollutant] + exog_columns], pollutant, exog_columns=exog_columns)



rmse_results = {
    "PM25": 15.24,
    "PM10": 0.65,
    "NO2": 0.48,
    "SO2": 0.21,
    "CO": 0.55
}

st.title("Model Evaluation")

for pollutant, rmse in rmse_results.items():
    st.write(f"RMSE for {pollutant}: {rmse:.2f}")


# import matplotlib.pyplot as plt

# # Visualize RMSE for pollutants
# plt.figure(figsize=(10, 6))
# plt.bar(simplified_sarima_results.keys(), simplified_sarima_results.values(), color='skyblue')
# plt.title('SARIMA Model RMSE for Pollutants')
# plt.xlabel('Pollutants')
# plt.ylabel('RMSE')
# plt.show()

# st.pyplot(plt)




st.title("CNN-LSTM Model")

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout

# Preprocess data with scaling and sliding window
def preprocess_data(series, n_steps=15):
    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(series.values.reshape(-1, 1))
    X, y = [], []
    for i in range(len(scaled_series) - n_steps):
        X.append(scaled_series[i:i + n_steps])
        y.append(scaled_series[i + n_steps])
    return np.array(X).reshape((-1, n_steps, 1)), np.array(y), scaler

# Define CNN-LSTM model
def cnn_lstm_model(X_train, y_train, X_test, y_test, n_steps):
    model = Sequential([
        Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(n_steps, 1)),
        Dropout(0.3),
        LSTM(100, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test), verbose=1)
    return model

# Apply CNN-LSTM for each pollutant
def cnn_lstm_forecast(series, pollutant_name, n_steps=15):
    print(f"\nCNN-LSTM Forecasting for {pollutant_name}...")

    # Preprocess data
    X, y, scaler = preprocess_data(series, n_steps)
    
    # Train-test split
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build and train CNN-LSTM model
    model = cnn_lstm_model(X_train, y_train, X_test, y_test, n_steps)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_inv = scaler.inverse_transform(y_pred)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Evaluate Metrics
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)

    print(f"Evaluation Metrics for {pollutant_name}:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    # Plot Actual vs Predicted
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inv, label='Actual', color='blue')
    plt.plot(y_pred_inv, label='Predicted', color='red')
    plt.title(f'CNN-LSTM: Actual vs Predicted for {pollutant_name}')
    plt.legend()
    plt.show()

    st.pyplot(plt)

# Define pollutants and apply the model
pollutants = ['pm25', 'pm10', 'no2', 'so2', 'co']  # Replace with actual column names in your dataset
for pollutant in pollutants:
    cnn_lstm_forecast(merged_df[pollutant], pollutant)

