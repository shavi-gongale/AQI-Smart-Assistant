# ==========================================
# AQI ANALYSIS AND PREDICTION PROJECT
# ==========================================
import re
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

def get_category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 200: return "Unhealthy"
    elif aqi <= 300: return "Poor"
    else: return "Hazardous"


def get_risk(aqi):
    if aqi <= 100: return "LOW"
    elif aqi <= 200: return "MEDIUM"
    else: return "HIGH"


# ==========================================
# LOAD DATASET
# ==========================================

print("\nLoading Dataset...")

df = pd.read_csv("aqi_india_work.csv")

print("Dataset Shape:", df.shape)
print(df.head())


# ==========================================
# SAMPLE DATA 
# ==========================================

df = df.sample(100000, random_state=42)

print("Sampled Dataset Shape:", df.shape)


# ==========================================
# FEATURE IMPORTANCE ANALYSIS
# ==========================================

print("\nFeature Importance Analysis")

X_temp = df.drop("us_aqi", axis=1)
y_temp = df["us_aqi"]

rf_temp = RandomForestRegressor(n_estimators=50)

rf_temp.fit(X_temp, y_temp)

importance = pd.Series(rf_temp.feature_importances_, index=X_temp.columns)

importance.sort_values().plot(kind="barh", figsize=(10,6))

plt.title("Feature Importance")

plt.show()


# ==========================================
# CITY POLLUTION RANKING
# ==========================================

print("\nCity Pollution Ranking")

city_columns = [
'ahmedabad','bengaluru','chandigarh','chennai','delhi','guwahati',
'hyderabad','jaipur','kolkata','lucknow','mumbai','raipur',
'ranchi','shimla','visakhapatnam'
]

city_avg = {}

for city in city_columns:

    city_data = df[df[city] == 1]

    if len(city_data) > 0:
        city_avg[city] = city_data["us_aqi"].mean()
    else:
        city_avg[city] = 0

city_avg = pd.Series(city_avg)

city_avg = city_avg.sort_values(ascending=True)

print(city_avg)

plt.figure(figsize=(10,5))

city_avg.plot(kind="bar", color="skyblue")

plt.title("City-wise AQI Comparison")

plt.ylabel("Average AQI")

plt.xlabel("City")

plt.xticks(rotation=45)

plt.show()


# ==========================================
# SEASONAL POLLUTION ANALYSIS
# ==========================================

print("\nSeasonal Pollution Analysis")

season_names = {

0: "Post-Monsoon",
1: "Summer",
2: "Monsoon",
3: "Winter"

}

season_avg = df.groupby("season")["us_aqi"].mean()

season_avg.index = season_avg.index.map(season_names)

plt.figure(figsize=(8,5))

season_avg.plot(kind="bar", color="orange")

plt.title("Seasonal AQI")

plt.ylabel("Average AQI")

plt.xlabel("Season")

plt.show()


# ==========================================
# CORRELATION HEATMAP
# ==========================================

print("\nCorrelation Heatmap")

plt.figure(figsize=(12,8))

sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")

plt.title("Pollution Correlation Heatmap")

plt.show()


# ==========================================
# AQI CATEGORY
# ==========================================

def category(aqi):

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Moderate"

    elif aqi <= 150:
        return "Unhealthy"

    else:
        return "Hazardous"

df["AQI_Category"] = df["us_aqi"].apply(category)

print("\nAQI Category Counts")

print(df["AQI_Category"].value_counts())


# ==========================================
# TIME PATTERN ANALYSIS
# ==========================================

hourly = df.groupby("Hour")["us_aqi"].mean()

hourly.plot()

plt.title("AQI Time Pattern")

plt.xlabel("Hour")

plt.ylabel("AQI")

plt.show()


# ==========================================
# AQI PREDICTION FEATURES
# ==========================================

features = [

'pm2_5_ugm3',
'pm10_ugm3',
'co_ugm3',
'no2_ugm3',
'so2_ugm3',
'o3_ugm3',
'dust_ugm3',
'aod',
'humidity_percent',
'dew_point_c',
'wind_gusts_kmh',
'pressure_msl_hpa',
'cloud_cover_percent'

]

X = df[features]

y = df["us_aqi"]


# ==========================================
# NORMALIZATION
# ==========================================

scaler = StandardScaler()

scaler.fit(X)
X_scaled = scaler.transform(X)


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

print("\nSplitting Dataset")

X_train, X_test, y_train, y_test = train_test_split(

X_scaled, y,

test_size=0.3,

random_state=42

)

print("Training Samples:", X_train.shape[0])

print("Testing Samples:", X_test.shape[0])


# ==========================================
# MULTIPLE LINEAR REGRESSION
# ==========================================

print("\nTraining Multiple Linear Regression")

lr = LinearRegression()

lr.fit(X_train, y_train)

pred_lr = lr.predict(X_test)

print("Linear Regression R2:", r2_score(y_test, pred_lr))

print("Linear Regression RMSE:", np.sqrt(mean_squared_error(y_test, pred_lr)))


# ==========================================
# DECISION TREE
# ==========================================

print("\nTraining Decision Tree")

dt = DecisionTreeRegressor(max_depth=15)

dt.fit(X_train, y_train)

pred_dt = dt.predict(X_test)

print("Decision Tree R2:", r2_score(y_test, pred_dt))

print("Decision Tree RMSE:", np.sqrt(mean_squared_error(y_test, pred_dt)))


# ==========================================
# RANDOM FOREST
# ==========================================

print("\nTraining Random Forest")

rf = RandomForestRegressor(

n_estimators=200,

max_depth=20,

min_samples_split=5,

random_state=42

)

# ==========================================
# SMART MODEL LOADING (NO RE-TRAINING)
# ==========================================

if os.path.exists("aqi_model.pkl") and os.path.exists("scaler.pkl"):
    print("\nLoading saved model...")

    rf = joblib.load("aqi_model.pkl")
    scaler = joblib.load("scaler.pkl")

    # ✅ FIX: Generate predictions when loading model
    pred_rf = rf.predict(X_test)

    print("Random Forest R2:", r2_score(y_test, pred_rf))
    print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, pred_rf)))

else:
    print("\nTraining Random Forest (First Time Only)...")

    rf.fit(X_train, y_train)

    pred_rf = rf.predict(X_test)

    print("Random Forest R2:", r2_score(y_test, pred_rf))
    print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, pred_rf)))

    joblib.dump(rf, "aqi_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    print("Model saved successfully")

    print("Random Forest R2:", r2_score(y_test, pred_rf))
    print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, pred_rf)))

    joblib.dump(rf, "aqi_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    print("Model saved successfully")

# ==========================================
# MODEL COMPARISON
# ==========================================



models = ["Linear Regression","Decision Tree","Random Forest"]

scores = []

# Linear Regression score
scores.append(r2_score(y_test, pred_lr))

# Decision Tree score
scores.append(r2_score(y_test, pred_dt))

# Random Forest score 
try:
    scores.append(r2_score(y_test, pred_rf))
except:
    scores.append(0)  


plt.bar(models, scores)

plt.title("Model Comparison")

plt.ylabel("R2 Score")

plt.show()


# ==========================================
# POLLUTION RISK INDICATOR
# ==========================================

df["Pollution_Risk"] = df["us_aqi"].apply(

lambda x: "High Risk" if x > 150 else "Low Risk"

)

print("\nPollution Risk Count")

print(df["Pollution_Risk"].value_counts())


# ==========================================
# PM2.5 IMPACT GRAPH
# ==========================================

plt.scatter(df["pm2_5_ugm3"], df["us_aqi"])

plt.xlabel("PM2.5")

plt.ylabel("AQI")

plt.title("Impact of PM2.5 on AQI")

plt.show()


# ==========================================
# USER AQI PREDICTION SYSTEM
# ==========================================

import numpy as np

print("\n========== AQI PREDICTION SYSTEM ==========")
print("Enter pollution values to predict AQI")


# function to safely take float input
def get_value(name):
    while True:
        try:
            return float(input(name))
        except ValueError:
            print("Please enter a valid number")


pm25 = get_value("PM2.5: ")
pm10 = get_value("PM10: ")
co = get_value("CO: ")
no2 = get_value("NO2: ")
so2 = get_value("SO2: ")
o3 = get_value("O3: ")
dust = get_value("Dust: ")
aod = get_value("AOD: ")

humidity = get_value("Humidity: ")
dew = get_value("Dew Point: ")
wind = get_value("Wind Gust: ")
pressure = get_value("Pressure: ")
cloud = get_value("Cloud Cover: ")


import pandas as pd

input_data = pd.DataFrame([{
    'pm2_5_ugm3': pm25,
    'pm10_ugm3': pm10,
    'co_ugm3': co,
    'no2_ugm3': no2,
    'so2_ugm3': so2,
    'o3_ugm3': o3,
    'dust_ugm3': dust,
    'aod': aod,
    'humidity_percent': humidity,
    'dew_point_c': dew,
    'wind_gusts_kmh': wind,
    'pressure_msl_hpa': pressure,
    'cloud_cover_percent': cloud
}])

input_scaled = scaler.transform(input_data)

predicted_aqi = rf.predict(input_scaled)

print("\nPredicted AQI:", predicted_aqi[0])


# ==========================================
# POLLUTION ALERT
# ==========================================

if predicted_aqi[0] > 200:
    print("Severe Pollution Alert")

elif predicted_aqi[0] > 150:
    print("High Pollution Alert")

else:
    print("Air Quality Normal")


from forecast import forecast_next_7_days

inputs = [
pm25, pm10, co, no2, so2, o3, dust, aod,
humidity, dew, wind, pressure, cloud
]

forecast = forecast_next_7_days(inputs)

forecast_clean = [float(x) for x in forecast]
print("\n7-Day Forecast:", forecast_clean)

# GRAPH
import matplotlib.pyplot as plt

days = ["Day1","Day2","Day3","Day4","Day5","Day6","Day7"]

plt.plot(days, forecast, marker='o')
plt.title("7-Day AQI Forecast")
plt.xlabel("Days")
plt.ylabel("AQI")
plt.show()    

# ===============================
#  CHATGPT-STYLE CHATBOT
# ===============================

from python_bridge import ask_prolog_chat
import re

# ===============================
# HELPER FUNCTIONS
# ===============================

def get_category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 200: return "Unhealthy"
    elif aqi <= 300: return "Poor"
    else: return "Hazardous"

def get_risk(aqi):
    if aqi <= 100: return "LOW"
    elif aqi <= 200: return "MEDIUM"
    else: return "HIGH"

# ===============================
# MEMORY + PROFILE
# ===============================

last_aqi = predicted_aqi[0]
user_profile = {"conditions": []}
unknown_questions = []

print("\n===== AQI CHATBOT =====")

while True:
    user_q = input("Ask something (or type 'exit'): ").lower()

    if user_q == "exit":
        break

    responses = []

    # ===============================
    # ACTIVE AQI (MAIN LOGIC)
    # ===============================
    match = re.search(r'(\d+)\s*day', user_q)

    if match:
        day = int(match.group(1))

        if 1 <= day <= 7:
            active_aqi = forecast[day-1]
            context = f"Day {day}"
        else:
            active_aqi = predicted_aqi[0]
            context = "Today"
    else:
        active_aqi = predicted_aqi[0]
        context = "Today"

    cat = get_category(active_aqi)
    risk = get_risk(active_aqi)

    # ===============================
    # MEMORY (TOMORROW)
    # ===============================
    if "tomorrow" in user_q:
        active_aqi = forecast[0]
        cat = get_category(active_aqi)

        response = ask_prolog_chat(active_aqi, user_q)
        print(f"Bot (Tomorrow AQI={active_aqi}, {cat}): {response}")
        continue

    # ===============================
    # PERSONALIZATION
    # ===============================
    if "asthma" in user_q:
        user_profile["conditions"].append("asthma")
        responses.append("Asthma noted. High risk.")

    if "heart" in user_q:
        user_profile["conditions"].append("heart")
        responses.append("Heart condition noted.")

    # ===============================
    # MULTI-QUESTION HANDLING
    # ===============================
    parts = re.split(r' and |,', user_q)

    for part in parts:

      response = ask_prolog_chat(active_aqi, part)

    # ✅ SMART FALLBACK (IMPORTANT)
    if not response or response.strip() == "":
        
        if "why" in part:
            response = f"{context} AQI is high due to traffic, pollution, weather conditions, and industrial activities."

        elif "go outside" in part:
            if active_aqi < 100:
                response = "It is safe to go outside."
            elif active_aqi < 200:
                response = "Limit outdoor activities."
            else:
                response = "Avoid going outside."

        elif "aqi" in part:
            response = f"Current AQI is {active_aqi:.2f}, which is {cat}."

        else:
            response = "Air quality depends on pollution, weather, and human activities."

    responses.append(response)

    # ===============================
    # TREND ANALYSIS
    # ===============================
    if "trend" in user_q or "increase" in user_q:
        if forecast[-1] > forecast[0]:
            responses.append("AQI is increasing in coming days")
        else:
            responses.append("AQI is decreasing")

    # ===============================
    # COMPARISON SYSTEM
    # ===============================
    if "compare" in user_q:
        responses.append(
            f"Today AQI={predicted_aqi[0]} vs Day3 AQI={forecast[2]}"
        )

    # ===============================
    # LOCATION SYSTEM
    # ===============================
    if "delhi" in user_q:
        responses.append("Delhi AQI is high due to traffic and pollution.")

    if "mumbai" in user_q:
        responses.append("Mumbai AQI moderate but weather-dependent.")

    # ===============================
    # HEALTH SYSTEM
    # ===============================
    if "asthma" in user_profile["conditions"]:
        responses.append("Avoid outdoor exposure completely.")

    if "heart" in user_profile["conditions"]:
        responses.append("Avoid stress and polluted areas.")

    # ===============================
    # WHY ENGINE (FIXED)
    # ===============================
    if "why" in user_q:
        responses.append(
            f"{context} ({cat}) → AQI may be high due to traffic, weather, and pollution buildup."
        )

    # ===============================
    # ALERT SYSTEM (FIXED)
    # ===============================
    if active_aqi > 250:
        responses.append(f"{context}: AQI is hazardous!")

    # ===============================
    # SUMMARY SYSTEM
    # ===============================
    if "summary" in user_q:
        trend = "Increasing" if forecast[-1] > forecast[0] else "Decreasing"
        responses.append(
            f"Summary → AQI={active_aqi} ({cat}), Risk={risk}, Trend={trend}"
        )

    # ===============================
    # LEARNING SYSTEM
    # ===============================
    if len(responses) == 0:
        unknown_questions.append(user_q)
        responses.append("I will learn this question in future.")

    # ===============================
    # OUTPUT (FINAL FIX)
    # ===============================
    for r in responses:
        print(f"Bot ({context} AQI={active_aqi}, {cat}, Risk={risk}): {r}")