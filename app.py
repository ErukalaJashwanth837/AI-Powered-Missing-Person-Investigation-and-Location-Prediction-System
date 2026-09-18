import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CASEFILE AI",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 CASEFILE")
st.subheader(
    "AI-Powered Missing Person Investigation and "
    "Probable Location Prediction System"
)

st.info(
    "Academic simulation only. All case identities and generated "
    "case information are fictional. Predictions are probabilistic "
    "and must not be used as real-world decisions."
)

# ============================================================
# SYNTHETIC GPS DATA GENERATION
# ============================================================

@st.cache_data
def generate_gps_data(n=3000, seed=42):

    np.random.seed(seed)

    # Synthetic area centers
    centers = np.array([
        [22.7196, 75.8577],
        [22.7350, 75.8500],
        [22.7000, 75.8700],
        [22.7500, 75.8800],
        [22.6900, 75.8400]
    ])

    records = []

    for i in range(n):

        area = np.random.randint(0, len(centers))

        lat = centers[area][0] + np.random.normal(0, 0.005)
        lon = centers[area][1] + np.random.normal(0, 0.005)

        hour = np.random.randint(6, 23)

        speed = np.random.uniform(5, 50)

        distance = speed * np.random.uniform(0.1, 1.5)

        records.append({
            "User_ID": np.random.randint(1, 21),
            "Latitude": lat,
            "Longitude": lon,
            "Hour": hour,
            "Day": np.random.randint(0, 7),
            "Month": np.random.randint(1, 13),
            "Speed": speed,
            "Distance": distance,
            "Area": area
        })

    df = pd.DataFrame(records)

    return df


df = generate_gps_data()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Controls")

case_id = st.sidebar.text_input(
    "Case ID",
    "MP-2026-017"
)

age_group = st.sidebar.selectbox(
    "Age Group",
    ["Under 18", "18–25", "26–40", "41–60", "60+"]
)

weather = st.sidebar.selectbox(
    "Weather",
    ["Clear", "Cloudy", "Rain", "Hot", "Cold"]
)

last_hour = st.sidebar.slider(
    "Last Seen Hour",
    0,
    23,
    18
)

time_since_seen = st.sidebar.slider(
    "Time Since Last Seen (hours)",
    1,
    24,
    2
)

last_lat = st.sidebar.number_input(
    "Last Known Latitude",
    value=22.7196,
    format="%.6f"
)

last_lon = st.sidebar.number_input(
    "Last Known Longitude",
    value=75.8577,
    format="%.6f"
)

# ============================================================
# CASE INFORMATION
# ============================================================

st.header("📋 Case Information")

case_col1, case_col2, case_col3, case_col4 = st.columns(4)

case_col1.metric("Case ID", case_id)
case_col2.metric("Age Group", age_group)
case_col3.metric("Last Seen", f"{last_hour}:00")
case_col4.metric("Weather", weather)

# ============================================================
# DATA OVERVIEW
# ============================================================

st.header("📊 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("GPS Records", len(df))
c2.metric("Users", df["User_ID"].nunique())
c3.metric("Average Speed", f"{df['Speed'].mean():.2f}")
c4.metric("Average Distance", f"{df['Distance'].mean():.2f}")

with st.expander("View GPS Dataset"):
    st.dataframe(df.head(100), use_container_width=True)

# ============================================================
# DATA PREPROCESSING
# ============================================================

st.header("🧹 Data Preprocessing")

# Remove missing values
df = df.dropna()

# Remove duplicates
df = df.drop_duplicates()

# Validate GPS coordinates
df = df[
    (df["Latitude"].between(-90, 90)) &
    (df["Longitude"].between(-180, 180))
]

st.success(
    f"Data cleaned successfully. Remaining records: {len(df)}"
)

# ============================================================
# FEATURE ENGINEERING
# ============================================================

st.header("⚙️ Feature Engineering")

df["Is_Weekend"] = df["Day"] >= 5

df["Movement_Intensity"] = (
    df["Speed"] * df["Distance"]
)

df["Hour_Sin"] = np.sin(
    2 * np.pi * df["Hour"] / 24
)

df["Hour_Cos"] = np.cos(
    2 * np.pi * df["Hour"] / 24
)

st.write("Generated movement-related features:")

st.write([
    "Hour",
    "Day",
    "Month",
    "Speed",
    "Distance",
    "Movement_Intensity",
    "Is_Weekend",
    "Hour_Sin",
    "Hour_Cos"
])

# ============================================================
# MOVEMENT CLUSTERING - K MEANS
# ============================================================

st.header("📍 1. Movement Clustering — K-Means")

coordinates = df[
    ["Latitude", "Longitude"]
]

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(
    coordinates
)

cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

cluster_table = pd.DataFrame({
    "Area": [
        f"Area {chr(65+i)}"
        for i in range(len(cluster_counts))
    ],
    "Visits": cluster_counts.values
})

st.dataframe(
    cluster_table,
    use_container_width=True
)

# ============================================================
# FREQUENT LOCATIONS
# ============================================================

st.header("📌 Frequently Visited Areas")

frequent_area = (
    df["Cluster"]
    .value_counts()
    .idxmax()
)

st.success(
    f"Most frequently visited cluster: "
    f"Area {chr(65 + frequent_area)}"
)

# ============================================================
# ANOMALY DETECTION
# ============================================================

st.header("🚨 2. Anomaly Detection — Isolation Forest")

anomaly_features = df[
    [
        "Latitude",
        "Longitude",
        "Speed",
        "Distance"
    ]
].copy()

isolation_model = IsolationForest(
    contamination=0.05,
    random_state=42
)

df["Anomaly"] = isolation_model.fit_predict(
    anomaly_features
)

df["Anomaly_Label"] = np.where(
    df["Anomaly"] == -1,
    "Anomaly",
    "Normal"
)

normal_count = (
    df["Anomaly_Label"] == "Normal"
).sum()

anomaly_count = (
    df["Anomaly_Label"] == "Anomaly"
).sum()

a1, a2 = st.columns(2)

a1.metric(
    "Normal Movements",
    normal_count
)

a2.metric(
    "Detected Anomalies",
    anomaly_count
)

st.warning(
    "An anomaly represents a movement pattern that differs "
    "from the learned normal pattern. It does not prove "
    "suspicious or criminal behavior."
)

# ============================================================
# CREATE SYNTHETIC TARGET
# ============================================================

st.header("🎯 Synthetic Case Training Data")

# Target area is generated from movement patterns
# for academic simulation.

df["Target_Area"] = (
    df["Cluster"].astype(str)
)

# ============================================================
# LOCATION PREDICTION - RANDOM FOREST
# ============================================================

st.header("🤖 3. Location Prediction — Random Forest")

feature_columns = [
    "Hour",
    "Day",
    "Speed",
    "Distance",
    "Latitude",
    "Longitude"
]

X = df[feature_columns]
y = df["Target_Area"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

y_pred = rf_model.predict(X_test)

# ============================================================
# MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Accuracy",
    f"{accuracy * 100:.2f}%"
)

m2.metric(
    "Precision",
    f"{precision * 100:.2f}%"
)

m3.metric(
    "Recall",
    f"{recall * 100:.2f}%"
)

m4.metric(
    "F1 Score",
    f"{f1 * 100:.2f}%"
)

# ============================================================
# CURRENT CASE PREDICTION
# ============================================================

case_avg_speed = float(
    df["Speed"].mean()
)

case_avg_distance = float(
    df["Distance"].mean()
)

case_data = pd.DataFrame([{
    "Hour": last_hour,
    "Day": 4,
    "Speed": case_avg_speed,
    "Distance": case_avg_distance,
    "Latitude": last_lat,
    "Longitude": last_lon
}])

probabilities = rf_model.predict_proba(
    case_data
)[0]

classes = rf_model.classes_

prediction_df = pd.DataFrame({
    "Area": classes,
    "Probability": probabilities
})

prediction_df = prediction_df.sort_values(
    "Probability",
    ascending=False
).reset_index(drop=True)

prediction_df["Probability"] *= 100

prediction_df["Probability"] = (
    prediction_df["Probability"].round(2)
)

prediction_df["Priority"] = pd.cut(
    prediction_df["Probability"],
    bins=[-1, 10, 20, 30, 101],
    labels=[
        "Low",
        "Medium",
        "High",
        "Very High"
    ]
)

st.subheader("📍 Probable Location Predictions")

st.dataframe(
    prediction_df,
    use_container_width=True
)

# ============================================================
# TOP 1 / TOP 3 / TOP 5
# ============================================================

st.subheader("🎯 Prediction Coverage")

top1 = prediction_df.head(1)
top3 = prediction_df.head(3)
top5 = prediction_df.head(5)

c1, c2, c3 = st.columns(3)

c1.metric(
    "Top-1 Area",
    top1.iloc[0]["Area"]
)

c2.metric(
    "Top-3 Areas",
    len(top3)
)

c3.metric(
    "Top-5 Areas",
    len(top5)
)

# ============================================================
# MARKOV CHAIN ROUTE PREDICTION
# ============================================================

st.header("🛣️ 4. Route Prediction — Markov Chain")

# Create synthetic sequential movement data
route_df = df.sort_values(
    ["User_ID", "Hour"]
).copy()

route_df["Next_Area"] = (
    route_df
    .groupby("User_ID")["Cluster"]
    .shift(-1)
)

transitions = route_df.dropna(
    subset=["Next_Area"]
)

transition_matrix = pd.crosstab(
    transitions["Cluster"],
    transitions["Next_Area"],
    normalize="index"
)

st.subheader("Transition Probability Matrix")

st.dataframe(
    transition_matrix.round(3),
    use_container_width=True
)

# Determine current area using nearest cluster
cluster_centers = kmeans.cluster_centers_

distances = np.sqrt(
    (
        cluster_centers[:, 0] - last_lat
    ) ** 2
    +
    (
        cluster_centers[:, 1] - last_lon
    ) ** 2
)

current_cluster = int(
    np.argmin(distances)
)

probable_route = [
    current_cluster
]

for _ in range(3):

    current = probable_route[-1]

    if current in transition_matrix.index:

        next_probs = (
            transition_matrix.loc[current]
        )

        next_area = int(
            next_probs.idxmax()
        )

        probable_route.append(
            next_area
        )

    else:
        break

route_names = [
    f"Area {chr(65 + int(x))}"
    for x in probable_route
]

st.success(
    " → ".join(route_names)
)

# ============================================================
# SEARCH PRIORITY SCORE
# ============================================================

st.header("📊 Search Priority Score")

priority_df = prediction_df.copy()

# Historical visit frequency
visit_counts = (
    df["Cluster"]
    .value_counts(normalize=True)
    * 100
)

priority_df["Visit_Frequency"] = (
    priority_df["Area"]
    .apply(
        lambda x:
        visit_counts.get(
            ord(x[-1]) - 65,
            0
        )
    )
)

# Normalize visit frequency
if priority_df["Visit_Frequency"].max() > 0:
    priority_df["Visit_Score"] = (
        priority_df["Visit_Frequency"]
        / priority_df["Visit_Frequency"].max()
        * 100
    )
else:
    priority_df["Visit_Score"] = 0

# Prediction score
priority_df["Prediction_Score"] = (
    priority_df["Probability"]
)

# Route similarity
route_set = set(probable_route)

priority_df["Route_Score"] = (
    priority_df["Area"]
    .apply(
        lambda x:
        100
        if ord(x[-1]) - 65 in route_set
        else 30
    )
)

# Distance relevance
priority_df["Distance_Score"] = 70

# Time relevance
priority_df["Time_Score"] = 70

# Anomaly evidence
priority_df["Anomaly_Score"] = (
    100
    if anomaly_count > 0
    else 20
)

# ============================================================
# WEIGHTED SCORE
# ============================================================

priority_df["Search_Priority_Score"] = (

    priority_df["Prediction_Score"] * 0.30

    + priority_df["Visit_Score"] * 0.20

    + priority_df["Route_Score"] * 0.15

    + priority_df["Distance_Score"] * 0.15

    + priority_df["Time_Score"] * 0.10

    + priority_df["Anomaly_Score"] * 0.10
)

priority_df["Search_Priority_Score"] = (
    priority_df["Search_Priority_Score"]
    .clip(0, 100)
    .round(2)
)

def priority_label(score):

    if score <= 30:
        return "Low"

    elif score <= 60:
        return "Medium"

    elif score <= 80:
        return "High"

    return "Very High"


priority_df["Search Priority"] = (
    priority_df["Search_Priority_Score"]
    .apply(priority_label)
)

priority_df = priority_df.sort_values(
    "Search_Priority_Score",
    ascending=False
).reset_index(drop=True)

st.dataframe(
    priority_df[
        [
            "Area",
            "Probability",
            "Search_Priority_Score",
            "Search Priority"
        ]
    ],
    use_container_width=True
)

# ============================================================
# EXPLAINABLE AI
# ============================================================

st.header("💡 Explainable AI")

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": rf_model.feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

st.dataframe(
    importance_df,
    use_container_width=True
)

top_area = prediction_df.iloc[0]["Area"]

st.info(
    f"""
    **Prediction explanation for {top_area}:**

    • The Random Forest model uses historical movement features.

    • Time of movement contributes to the prediction.

    • Previous GPS coordinates provide geographical context.

    • Historical speed and distance represent movement behavior.

    • Frequently observed movement clusters influence the predicted area.

    • The result is a probability, not a confirmed location.
    """
)

# ============================================================
# INTERACTIVE MAP
# ============================================================

st.header("🗺️ Interactive Investigation Map")

map_center = [
    last_lat,
    last_lon
]

m = folium.Map(
    location=map_center,
    zoom_start=12
)

# Last known location
folium.Marker(
    [last_lat, last_lon],
    popup="Last Known Location",
    tooltip="Last Known Location",
    icon=folium.Icon(
        color="red",
        icon="info-sign"
    )
).add_to(m)

# Frequently visited cluster centers
for i, center in enumerate(cluster_centers):

    area_name = (
        f"Area {chr(65 + i)}"
    )

    folium.CircleMarker(
        location=[
            center[0],
            center[1]
        ],
        radius=8,
        popup=area_name,
        tooltip=area_name,
        fill=True
    ).add_to(m)

# Anomalous locations
anomalies = df[
    df["Anomaly"] == -1
].sample(
    min(100, anomaly_count),
    random_state=42
)

for _, row in anomalies.iterrows():

    folium.CircleMarker(
        location=[
            row["Latitude"],
            row["Longitude"]
        ],
        radius=4,
        popup="Anomalous Movement",
        fill=True
    ).add_to(m)

# Heatmap
heat_data = df[
    ["Latitude", "Longitude"]
].values.tolist()

HeatMap(
    heat_data,
    radius=10
).add_to(m)

# Probable route
route_coordinates = []

for area in probable_route:

    route_coordinates.append(
        [
            cluster_centers[
                int(area)
            ][0],

            cluster_centers[
                int(area)
            ][1]
        ]
    )

if len(route_coordinates) >= 2:

    folium.PolyLine(
        route_coordinates,
        weight=5,
        popup="Probable Route"
    ).add_to(m)

st_folium(
    m,
    width=1200,
    height=600
)

# ============================================================
# MOVEMENT ANALYSIS
# ============================================================

st.header("📈 Movement Analysis")

mc1, mc2, mc3, mc4 = st.columns(4)

mc1.metric(
    "Average Speed",
    f"{df['Speed'].mean():.2f}"
)

mc2.metric(
    "Maximum Speed",
    f"{df['Speed'].max():.2f}"
)

mc3.metric(
    "Average Distance",
    f"{df['Distance'].mean():.2f}"
)

mc4.metric(
    "Visited Areas",
    df["Cluster"].nunique()
)

# ============================================================
# CASE SUMMARY
# ============================================================

st.header("📝 Investigation Summary")

best_area = prediction_df.iloc[0]["Area"]
best_probability = prediction_df.iloc[0]["Probability"]

best_priority_row = priority_df[
    priority_df["Area"] == best_area
].iloc[0]

st.write(
    f"""
    **Case:** {case_id}

    **Last known coordinates:** 
    {last_lat:.6f}, {last_lon:.6f}

    **Last seen hour:** {last_hour}:00

    **Most probable area:** {best_area}

    **Model probability:** {best_probability:.2f}%

    **Search-priority score:** 
    {best_priority_row["Search_Priority_Score"]:.2f}

    **Priority category:** 
    {best_priority_row["Search Priority"]}

    **Probable route:** 
    {" → ".join(route_names)}
    """
)

# ============================================================
# ETHICAL DISCLAIMER
# ============================================================

st.header("⚠️ Ethical Considerations")

st.warning(
    """
    This application is an academic simulation.

    • Case identities are fictional.
    • No private missing-person records are used.
    • Predictions are probabilistic.
    • Anomalies do not prove suspicious or criminal behavior.
    • A predicted area does not prove a person's location.
    • Model bias and false positives are possible.
    • The system must not be used to make real-world decisions
      concerning missing persons.
    """
)

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CASEFILE AI | Advanced Machine Learning Academic Project"
)
