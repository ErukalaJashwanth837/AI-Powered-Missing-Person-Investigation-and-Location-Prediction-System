import math
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CASEFILE AI",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ CASEFILE AI")
st.subheader(
    "AI-Powered Missing Person Investigation & Probable Location Prediction"
)

st.info(
    "Academic simulation only. All predictions are probabilistic and "
    "must not be used for real-world missing-person decisions."
)


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine(lat1, lon1, lat2, lon2):
    radius = 6371.0

    lat1 = math.radians(float(lat1))
    lat2 = math.radians(float(lat2))

    dlat = math.radians(float(lat2) - float(lat1))
    dlon = math.radians(float(lon2) - float(lon1))

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    a = min(1.0, max(0.0, a))

    return 2 * radius * math.asin(math.sqrt(a))


# ============================================================
# SYNTHETIC DATASET
# ============================================================

def create_synthetic_dataset(n=1000, seed=42):

    rng = np.random.default_rng(seed)

    centers = np.array([
        [22.7196, 75.8577],
        [22.7250, 75.8650],
        [22.7100, 75.8500],
        [22.7350, 75.8450],
        [22.7000, 75.8750]
    ])

    rows = []

    start_time = pd.Timestamp("2026-01-01 06:00:00")

    for i in range(n):

        user_id = f"U{(i % 15) + 1:02d}"

        area_id = rng.choice(
            5,
            p=[0.30, 0.25, 0.20, 0.15, 0.10]
        )

        lat, lon = centers[area_id]

        lat += rng.normal(0, 0.002)
        lon += rng.normal(0, 0.002)

        # Occasional unusual movement
        if rng.random() < 0.05:
            lat += rng.normal(0, 0.015)
            lon += rng.normal(0, 0.015)

        timestamp = (
            start_time
            + pd.Timedelta(minutes=i * 10)
        )

        rows.append([
            user_id,
            lat,
            lon,
            timestamp
        ])

    return pd.DataFrame(
        rows,
        columns=[
            "User_ID",
            "Latitude",
            "Longitude",
            "Timestamp"
        ]
    )


# ============================================================
# DATA CLEANING
# ============================================================

def clean_data(df):

    df = df.copy()

    if df.empty:
        raise ValueError("The dataset is empty.")

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    rename_map = {}

    if "user_id" not in df.columns:
        for col in ["userid", "user", "id"]:
            if col in df.columns:
                rename_map[col] = "user_id"
                break

    if "latitude" not in df.columns:
        for col in ["lat"]:
            if col in df.columns:
                rename_map[col] = "latitude"
                break

    if "longitude" not in df.columns:
        for col in ["lon", "lng"]:
            if col in df.columns:
                rename_map[col] = "longitude"
                break

    if "timestamp" not in df.columns:
        for col in ["datetime", "date_time", "time", "date"]:
            if col in df.columns:
                rename_map[col] = "timestamp"
                break

    df.rename(columns=rename_map, inplace=True)

    if "latitude" not in df.columns or "longitude" not in df.columns:
        raise ValueError(
            "CSV must contain Latitude and Longitude columns."
        )

    # Create user ID if missing
    if "user_id" not in df.columns:
        df["user_id"] = "SINGLE_USER"

    # Convert coordinates to numeric
    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

    # Remove missing GPS values
    df.dropna(
        subset=["latitude", "longitude"],
        inplace=True
    )

    # Validate GPS coordinates
    df = df[
        df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
    ]

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Timestamp processing
    if "timestamp" in df.columns:

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

    else:

        df["timestamp"] = pd.date_range(
            start="2026-01-01 08:00",
            periods=len(df),
            freq="10min"
        )

    # Remove invalid timestamps
    df.dropna(
        subset=["timestamp"],
        inplace=True
    )

    df.sort_values(
        ["user_id", "timestamp"],
        inplace=True
    )

    df.reset_index(
        drop=True,
        inplace=True
    )

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def feature_engineering(df):

    df = df.copy()

    # Previous location for each user
    df["previous_latitude"] = (
        df.groupby("user_id")["latitude"]
        .shift(1)
    )

    df["previous_longitude"] = (
        df.groupby("user_id")["longitude"]
        .shift(1)
    )

    df["previous_timestamp"] = (
        df.groupby("user_id")["timestamp"]
        .shift(1)
    )

    # Distance calculation
    distances = []

    for _, row in df.iterrows():

        if pd.isna(row["previous_latitude"]):

            distances.append(0.0)

        else:

            distances.append(
                haversine(
                    row["previous_latitude"],
                    row["previous_longitude"],
                    row["latitude"],
                    row["longitude"]
                )
            )

    df["distance_km"] = distances

    # Time gap
    df["time_gap_minutes"] = (
        (
            df["timestamp"]
            - df["previous_timestamp"]
        )
        .dt.total_seconds()
        .div(60)
    )

    df["time_gap_minutes"] = (
        df["time_gap_minutes"]
        .fillna(10)
        .clip(lower=0.1)
    )

    # Speed
    df["speed_kmh"] = (
        df["distance_km"]
        / (df["time_gap_minutes"] / 60)
    )

    df["speed_kmh"] = (
        df["speed_kmh"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
        .clip(upper=300)
    )

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["weekday"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    df["is_weekend"] = (
        df["weekday"] >= 5
    ).astype(int)

    # Geographic area
    cell = 0.01

    lat_cell = (
        (df["latitude"] / cell)
        .round()
        * cell
    ).round(4)

    lon_cell = (
        (df["longitude"] / cell)
        .round()
        * cell
    ).round(4)

    df["area"] = (
        lat_cell.astype(str)
        + "_"
        + lon_cell.astype(str)
    )

    return df


# ============================================================
# K-MEANS CLUSTERING
# ============================================================

def perform_clustering(df, n_clusters):

    coordinates = df[
        ["latitude", "longitude"]
    ].copy()

    actual_clusters = min(
        int(n_clusters),
        len(coordinates)
    )

    if actual_clusters < 2:
        raise ValueError(
            "At least 2 GPS records are required for clustering."
        )

    model = KMeans(
        n_clusters=actual_clusters,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(
        coordinates
    )

    result = df.copy()

    result["movement_cluster"] = labels

    return model, result


# ============================================================
# ISOLATION FOREST
# ============================================================

def perform_anomaly_detection(df, contamination):

    features = df[
        [
            "latitude",
            "longitude",
            "speed_kmh",
            "distance_km",
            "hour"
        ]
    ].fillna(0)

    scaler = StandardScaler()

    scaled = scaler.fit_transform(
        features
    )

    model = IsolationForest(
        n_estimators=250,
        contamination=float(contamination),
        random_state=42
    )

    predictions = model.fit_predict(
        scaled
    )

    result = df.copy()

    result["anomaly"] = (
        predictions == -1
    ).astype(int)

    result["anomaly_score"] = (
        -model.score_samples(scaled)
    )

    return model, scaler, result


# ============================================================
# LOCATION PREDICTION DATA
# ============================================================

def prepare_prediction_data(df):

    data = df.copy()

    data["average_speed"] = (
        data.groupby("user_id")["speed_kmh"]
        .transform("mean")
    )

    data["average_distance"] = (
        data.groupby("user_id")["distance_km"]
        .transform("mean")
    )

    data["visit_frequency"] = (
        data.groupby(
            ["user_id", "area"]
        )["area"]
        .transform("count")
    )

    data["previous_area"] = (
        data.groupby("user_id")["area"]
        .shift(1)
        .fillna("UNKNOWN")
    )

    feature_columns = [
        "latitude",
        "longitude",
        "hour",
        "weekday",
        "month",
        "is_weekend",
        "speed_kmh",
        "distance_km",
        "average_speed",
        "average_distance",
        "visit_frequency",
        "previous_area"
    ]

    data = data.dropna(
        subset=["area"]
    )

    X = pd.get_dummies(
        data[feature_columns],
        dtype=float
    )

    y = data["area"].astype(str)

    # Remove very rare target classes
    # At least 5 observations are required
    # so train/test splitting remains reliable.
    counts = y.value_counts()

    valid_classes = counts[
        counts >= 5
    ].index

    valid = y.isin(
        valid_classes
    )

    X = X.loc[valid].copy()
    y = y.loc[valid].copy()
    valid_data = data.loc[valid].copy()

    if len(X) < 20:
        raise ValueError(
            "Not enough valid records for location prediction. "
            "Use a larger dataset."
        )

    if y.nunique() < 2:
        raise ValueError(
            "At least two location classes are required."
        )

    return valid_data, X, y


# ============================================================
# RANDOM FOREST
# ============================================================

def train_random_forest(X, y):

    if y.nunique() < 2:
        raise ValueError(
            "At least two location classes are required."
        )

    class_counts = y.value_counts()

    # Every class has at least 5 records because of
    # prepare_prediction_data().
    min_class_count = int(
        class_counts.min()
    )

    test_size = max(
        0.25,
        y.nunique() / len(y) + 0.01
    )

    test_size = min(
        test_size,
        0.40
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "precision": precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "recall": recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "f1": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=model.classes_
        )
    }

    return (
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        metrics
    )


# ============================================================
# TOP-K ACCURACY
# ============================================================

def top_k_accuracy(model, X_test, y_test, k):

    probabilities = model.predict_proba(
        X_test
    )

    classes = np.array(
        model.classes_
    )

    k = min(
        int(k),
        len(classes)
    )

    top_indices = np.argsort(
        probabilities,
        axis=1
    )[:, -k:]

    top_classes = classes[
        top_indices
    ]

    return np.mean([
        actual in predicted
        for actual, predicted
        in zip(y_test, top_classes)
    ])


# ============================================================
# MARKOV CHAIN
# ============================================================

def create_markov_chain(df):

    transitions = defaultdict(Counter)

    groups = (
        df.sort_values("timestamp")
        .groupby("user_id")
    )

    for _, group in groups:

        areas = (
            group["area"]
            .astype(str)
            .tolist()
        )

        for current, next_area in zip(
            areas[:-1],
            areas[1:]
        ):

            if current != next_area:

                transitions[
                    current
                ][next_area] += 1

    probabilities = {}

    for current, counter in transitions.items():

        total = sum(
            counter.values()
        )

        if total > 0:

            probabilities[current] = {
                nxt: count / total
                for nxt, count
                in counter.items()
            }

    return probabilities


def predict_route(
    markov,
    start_area,
    steps=4
):

    route = [start_area]

    current = start_area

    visited = {start_area}

    for _ in range(steps):

        if current not in markov:
            break

        available = {
            area: probability
            for area, probability
            in markov[current].items()
            if area not in visited
        }

        if not available:
            break

        next_area = max(
            available,
            key=available.get
        )

        route.append(
            next_area
        )

        visited.add(
            next_area
        )

        current = next_area

    return route


# ============================================================
# PRIORITY SCORE
# ============================================================

def calculate_priority(
    ml_probability,
    visit_frequency,
    route_similarity,
    distance_relevance,
    time_relevance,
    anomaly_evidence
):

    score = (
        ml_probability * 0.30
        + visit_frequency * 0.20
        + route_similarity * 0.15
        + distance_relevance * 0.15
        + time_relevance * 0.10
        + anomaly_evidence * 0.10
    )

    return float(score)


def priority_category(score):

    if score <= 30:
        return "Low"

    if score <= 60:
        return "Medium"

    if score <= 80:
        return "High"

    return "Very High"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Project Controls"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload GPS CSV",
    type=["csv"]
)

use_synthetic = st.sidebar.checkbox(
    "Use synthetic dataset",
    value=uploaded_file is None
)

number_clusters = st.sidebar.slider(
    "Number of K-Means clusters",
    min_value=2,
    max_value=10,
    value=5
)

contamination = st.sidebar.slider(
    "Anomaly contamination",
    min_value=0.01,
    max_value=0.20,
    value=0.05,
    step=0.01
)


# ============================================================
# LOAD DATA
# ============================================================

if uploaded_file is not None and not use_synthetic:

    try:

        raw_data = pd.read_csv(
            uploaded_file
        )

        source_name = uploaded_file.name

    except Exception as error:

        st.error(
            f"CSV loading error: {error}"
        )

        st.stop()

else:

    raw_data = create_synthetic_dataset()

    source_name = (
        "Built-in synthetic dataset"
    )


# ============================================================
# PROCESS DATA
# ============================================================

try:

    cleaned_data = clean_data(
        raw_data
    )

    data = feature_engineering(
        cleaned_data
    )

except Exception as error:

    st.error(
        f"Processing error: {error}"
    )

    st.stop()


# ============================================================
# MODELS
# ============================================================

try:

    cluster_model, data = perform_clustering(
        data,
        number_clusters
    )

except Exception as error:

    st.error(
        f"Clustering error: {error}"
    )

    st.stop()


try:

    anomaly_model, anomaly_scaler, data = (
        perform_anomaly_detection(
            data,
            contamination
        )
    )

except Exception as error:

    st.error(
        f"Anomaly detection error: {error}"
    )

    st.stop()


try:

    prediction_data, X, y = (
        prepare_prediction_data(data)
    )

    (
        location_model,
        X_train,
        X_test,
        y_train,
        y_test,
        model_metrics
    ) = train_random_forest(
        X,
        y
    )

except Exception as error:

    st.error(
        f"Location model error: {error}"
    )

    st.info(
        "Try using the built-in synthetic dataset "
        "or upload a larger GPS dataset."
    )

    st.stop()


markov = create_markov_chain(
    data
)


# ============================================================
# FICTIONAL CASE
# ============================================================

st.header(
    "📁 Fictional Case Information"
)

case_col1, case_col2, case_col3, case_col4 = (
    st.columns(4)
)

with case_col1:

    case_id = st.text_input(
        "Case ID",
        "MP-2026-017"
    )

with case_col2:

    age_group = st.selectbox(
        "Age Group",
        [
            "18-25",
            "26-35",
            "36-50",
            "50+"
        ]
    )

with case_col3:

    case_time = st.time_input(
        "Last Seen Time",
        pd.Timestamp(
            "2026-01-01 18:45"
        ).time()
    )

with case_col4:

    weather = st.selectbox(
        "Weather",
        [
            "Clear",
            "Cloudy",
            "Rain",
            "Hot",
            "Windy"
        ]
    )


area_list = sorted(
    data["area"]
    .astype(str)
    .unique()
)

last_known_area = st.selectbox(
    "Last Known Area",
    area_list
)

previous_area = st.selectbox(
    "Previous Area",
    ["UNKNOWN"] + area_list
)


# ============================================================
# LAST KNOWN LOCATION
# ============================================================

last_area_data = data[
    data["area"].astype(str)
    == last_known_area
]

if len(last_area_data) == 0:

    last_area_data = data.head(1)

last_latitude = float(
    last_area_data["latitude"].mean()
)

last_longitude = float(
    last_area_data["longitude"].mean()
)

case_hour = case_time.hour


# ============================================================
# TABS
# ============================================================

tabs = st.tabs([
    "📊 Overview",
    "🧹 Preprocessing",
    "🔵 Clustering",
    "🚨 Anomaly Detection",
    "🎯 Location Prediction",
    "➡️ Route Prediction",
    "📈 Priority Score",
    "🔍 Explainable AI",
    "🗺️ Interactive Map"
])


# ============================================================
# OVERVIEW
# ============================================================

with tabs[0]:

    st.header(
        "Dataset Overview"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "GPS Records",
        f"{len(data):,}"
    )

    c2.metric(
        "Unique Users",
        data["user_id"].nunique()
    )

    c3.metric(
        "Areas",
        data["area"].nunique()
    )

    c4.metric(
        "Detected Anomalies",
        int(data["anomaly"].sum())
    )

    st.write(
        f"**Dataset:** {source_name}"
    )

    st.subheader(
        "Movement Statistics"
    )

    statistics = pd.DataFrame({
        "Metric": [
            "Average Speed (km/h)",
            "Maximum Speed (km/h)",
            "Average Distance (km)",
            "Total Distance (km)",
            "Average Time Gap (minutes)"
        ],

        "Value": [
            data["speed_kmh"].mean(),
            data["speed_kmh"].max(),
            data["distance_km"].mean(),
            data["distance_km"].sum(),
            data["time_gap_minutes"].mean()
        ]
    })

    st.dataframe(
        statistics,
        hide_index=True,
        use_container_width=True
    )

    st.subheader(
        "Movement by Hour"
    )

    hourly = (
        data["hour"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(
        hourly
    )


# ============================================================
# PREPROCESSING
# ============================================================

with tabs[1]:

    st.header(
        "🧹 Data Preprocessing"
    )

    st.markdown("""
    The preprocessing pipeline performs:

    - Missing-value handling
    - Duplicate removal
    - Invalid GPS-coordinate removal
    - Timestamp conversion
    - Distance calculation
    - Speed calculation
    - Hour/day/month extraction
    - Weekend detection
    - Geographical area creation
    """)

    p1, p2 = st.columns(2)

    p1.metric(
        "Original Rows",
        len(raw_data)
    )

    p2.metric(
        "Cleaned Rows",
        len(data)
    )

    st.dataframe(
        data.head(50),
        use_container_width=True
    )


# ============================================================
# CLUSTERING
# ============================================================

with tabs[2]:

    st.header(
        "🔵 K-Means Movement Clustering"
    )

    st.write(
        "K-Means groups GPS observations into geographical "
        "movement patterns."
    )

    cluster_counts = (
        data["movement_cluster"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(
        cluster_counts
    )

    centers = pd.DataFrame(
        cluster_model.cluster_centers_,
        columns=[
            "Latitude",
            "Longitude"
        ]
    )

    centers.index.name = "Cluster"

    st.dataframe(
        centers,
        use_container_width=True
    )


# ============================================================
# ANOMALY DETECTION
# ============================================================

with tabs[3]:

    st.header(
        "🚨 Isolation Forest Anomaly Detection"
    )

    normal_count = int(
        (data["anomaly"] == 0).sum()
    )

    anomaly_count = int(
        (data["anomaly"] == 1).sum()
    )

    a1, a2 = st.columns(2)

    a1.metric(
        "Normal Points",
        normal_count
    )

    a2.metric(
        "Anomalous Points",
        anomaly_count
    )

    anomaly_chart = (
        data["anomaly"]
        .value_counts()
        .rename({
            0: "Normal",
            1: "Anomaly"
        })
    )

    st.bar_chart(
        anomaly_chart
    )

    anomalies = (
        data[data["anomaly"] == 1]
        .sort_values(
            "anomaly_score",
            ascending=False
        )
    )

    st.dataframe(
        anomalies[
            [
                "latitude",
                "longitude",
                "timestamp",
                "speed_kmh",
                "distance_km",
                "area",
                "anomaly_score"
            ]
        ].head(100),
        use_container_width=True
    )

    st.warning(
        "An anomaly indicates a movement pattern different "
        "from the learned data. It does not prove suspicious "
        "or criminal behavior."
    )


# ============================================================
# LOCATION PREDICTION
# ============================================================

with tabs[4]:

    st.header(
        "🎯 Random Forest Location Prediction"
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Accuracy",
        f"{model_metrics['accuracy']:.3f}"
    )

    m2.metric(
        "Precision",
        f"{model_metrics['precision']:.3f}"
    )

    m3.metric(
        "Recall",
        f"{model_metrics['recall']:.3f}"
    )

    m4.metric(
        "F1 Score",
        f"{model_metrics['f1']:.3f}"
    )

    t1, t2, t3 = st.columns(3)

    t1.metric(
        "Top-1 Accuracy",
        f"{top_k_accuracy(location_model, X_test, y_test, 1):.3f}"
    )

    t2.metric(
        "Top-3 Accuracy",
        f"{top_k_accuracy(location_model, X_test, y_test, 3):.3f}"
    )

    t3.metric(
        "Top-5 Accuracy",
        f"{top_k_accuracy(location_model, X_test, y_test, 5):.3f}"
    )

    # ========================================================
    # BUILD CASE PREDICTION ROW
    # ========================================================

    template = prediction_data.iloc[0].copy()

    template["latitude"] = last_latitude
    template["longitude"] = last_longitude
    template["hour"] = case_hour
    template["previous_area"] = previous_area

    template["speed_kmh"] = (
        data["speed_kmh"].median()
    )

    template["distance_km"] = (
        data["distance_km"].median()
    )

    template["average_speed"] = (
        data["speed_kmh"].mean()
    )

    template["average_distance"] = (
        data["distance_km"].mean()
    )

    template["visit_frequency"] = (
        data["area"]
        .value_counts()
        .median()
    )

    prediction_columns = [
        "latitude",
        "longitude",
        "hour",
        "weekday",
        "month",
        "is_weekend",
        "speed_kmh",
        "distance_km",
        "average_speed",
        "average_distance",
        "visit_frequency",
        "previous_area"
    ]

    case_df = pd.DataFrame([
        {
            column: template[column]
            for column in prediction_columns
        }
    ])

    case_X = pd.get_dummies(
        case_df,
        dtype=float
    )

    # Match training columns exactly
    case_X = case_X.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    probabilities = (
        location_model
        .predict_proba(case_X)[0]
    )

    classes = np.array(
        location_model.classes_
    )

    predictions = pd.DataFrame({
        "Area": classes,
        "Probability (%)":
            probabilities * 100
    })

    predictions.sort_values(
        "Probability (%)",
        ascending=False,
        inplace=True
    )

    predictions.reset_index(
        drop=True,
        inplace=True
    )

    predictions.insert(
        0,
        "Rank",
        range(
            1,
            len(predictions) + 1
        )
    )

    predictions["Probability (%)"] = (
        predictions["Probability (%)"]
        .round(2)
    )

    st.subheader(
        "Probable Areas"
    )

    st.dataframe(
        predictions,
        hide_index=True,
        use_container_width=True
    )

    st.subheader(
        "Confusion Matrix"
    )

    cm = pd.DataFrame(
        model_metrics["confusion_matrix"],
        index=location_model.classes_,
        columns=location_model.classes_
    )

    st.dataframe(
        cm,
        use_container_width=True
    )


# ============================================================
# ROUTE PREDICTION
# ============================================================

with tabs[5]:

    st.header(
        "➡️ Markov Chain Route Prediction"
    )

    st.write(
        "The Markov Chain estimates the next area from "
        "historical area-to-area transitions."
    )

    if last_known_area in markov:

        route_table = []

        for next_area, probability in sorted(
            markov[last_known_area].items(),
            key=lambda x: x[1],
            reverse=True
        ):

            route_table.append({
                "From": last_known_area,
                "To": next_area,
                "Transition Probability (%)":
                    round(probability * 100, 2)
            })

        st.dataframe(
            pd.DataFrame(route_table),
            hide_index=True,
            use_container_width=True
        )

        route = predict_route(
            markov,
            last_known_area,
            steps=4
        )

        st.success(
            "Probable route: "
            + " → ".join(route)
        )

    else:

        st.info(
            "No learned transition exists "
            "from this area."
        )


# ============================================================
# PRIORITY SCORE
# ============================================================

with tabs[6]:

    st.header(
        "📈 Search Priority Score"
    )

    st.write(
        "Suggested project weights:"
    )

    weights = pd.DataFrame({
        "Factor": [
            "ML Prediction Probability",
            "Historical Visit Frequency",
            "Route Similarity",
            "Distance Relevance",
            "Time Relevance",
            "Anomaly Evidence"
        ],

        "Weight (%)": [
            30,
            20,
            15,
            15,
            10,
            10
        ]
    })

    st.dataframe(
        weights,
        hide_index=True,
        use_container_width=True
    )

    frequency = (
        data["area"]
        .value_counts(
            normalize=True
        ) * 100
    )

    anomaly_by_area = (
        data.groupby("area")["anomaly"]
        .mean() * 100
    )

    priority_rows = []

    for _, prediction in predictions.iterrows():

        area = prediction["Area"]

        ml_probability = float(
            prediction["Probability (%)"]
        )

        visit_frequency = min(
            float(
                frequency.get(
                    area,
                    0
                )
            ),
            100
        )

        route_similarity = (
            markov
            .get(last_known_area, {})
            .get(area, 0)
            * 100
        )

        area_rows = data[
            data["area"].astype(str)
            == area
        ]

        if area_rows.empty:
            continue

        area_lat = area_rows[
            "latitude"
        ].mean()

        area_lon = area_rows[
            "longitude"
        ].mean()

        distance = haversine(
            last_latitude,
            last_longitude,
            area_lat,
            area_lon
        )

        distance_relevance = (
            100 * math.exp(
                -distance / 5
            )
        )

        typical_hour = int(
            area_rows["hour"].median()
        )

        time_difference = abs(
            case_hour - typical_hour
        )

        # Handle circular clock distance
        time_difference = min(
            time_difference,
            24 - time_difference
        )

        time_relevance = (
            100
            * (
                1
                - min(
                    time_difference / 12,
                    1
                )
            )
        )

        anomaly_evidence = float(
            anomaly_by_area.get(
                area,
                0
            )
        )

        score = calculate_priority(
            ml_probability,
            visit_frequency,
            route_similarity,
            distance_relevance,
            time_relevance,
            anomaly_evidence
        )

        priority_rows.append({
            "Area": area,
            "ML Probability": ml_probability,
            "Visit Frequency": visit_frequency,
            "Route Similarity": route_similarity,
            "Distance Relevance": distance_relevance,
            "Time Relevance": time_relevance,
            "Anomaly Evidence": anomaly_evidence,
            "Priority Score": score,
            "Priority": priority_category(score)
        })

    priority_df = pd.DataFrame(
        priority_rows
    )

    if not priority_df.empty:

        priority_df.sort_values(
            "Priority Score",
            ascending=False,
            inplace=True
        )

        priority_df["Priority Score"] = (
            priority_df["Priority Score"]
            .round(2)
        )

        st.dataframe(
            priority_df,
            hide_index=True,
            use_container_width=True
        )

    else:

        st.info(
            "Priority scores could not be calculated."
        )


# ============================================================
# EXPLAINABLE AI
# ============================================================

with tabs[7]:

    st.header(
        "🔍 Explainable AI"
    )

    st.write(
        "Random Forest feature importance is used to "
        "show which input features contributed most "
        "to the trained model."
    )

    importance = pd.Series(
        location_model.feature_importances_,
        index=X_train.columns
    )

    importance = (
        importance
        .sort_values(
            ascending=False
        )
        .head(15)
    )

    st.bar_chart(
        importance
    )

    if not priority_df.empty:

        explanation_area = priority_df.iloc[0]

        st.subheader(
            "Prediction Explanation"
        )

        st.write(
            f"**Area:** "
            f"{explanation_area['Area']}"
        )

        st.write(
            f"**ML probability:** "
            f"{explanation_area['ML Probability']:.2f}%"
        )

        st.write(
            f"**Historical visit frequency:** "
            f"{explanation_area['Visit Frequency']:.2f}%"
        )

        st.write(
            f"**Route similarity:** "
            f"{explanation_area['Route Similarity']:.2f}%"
        )

        st.write(
            f"**Distance relevance:** "
            f"{explanation_area['Distance Relevance']:.2f}%"
        )

        st.write(
            f"**Time relevance:** "
            f"{explanation_area['Time Relevance']:.2f}%"
        )

        st.write(
            f"**Anomaly evidence:** "
            f"{explanation_area['Anomaly Evidence']:.2f}%"
        )

        st.info(
            "These factors explain the model inputs and "
            "scoring mechanism. They do not prove the "
            "person's actual location."
        )


# ============================================================
# INTERACTIVE MAP
# ============================================================

with tabs[8]:

    st.header(
        "🗺️ Interactive Investigation Map"
    )

    map_center = [
        data["latitude"].mean(),
        data["longitude"].mean()
    ]

    investigation_map = folium.Map(
        location=map_center,
        zoom_start=13,
        control_scale=True
    )

    # ========================================================
    # LAST KNOWN LOCATION
    # ========================================================

    folium.Marker(
        [
            last_latitude,
            last_longitude
        ],
        tooltip="Last Known Synthetic Location",
        popup=(
            f"Case: {case_id}<br>"
            f"Area: {last_known_area}<br>"
            f"Weather: {weather}<br>"
            f"Age Group: {age_group}"
        ),
        icon=folium.Icon(
            icon="info-sign"
        )
    ).add_to(
        investigation_map
    )

    # ========================================================
    # AREA CENTERS
    # ========================================================

    area_centers = (
        data.groupby("area")
        .agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            visits=("area", "size")
        )
        .reset_index()
    )

    for _, row in area_centers.iterrows():

        radius = max(
            4,
            min(
                12,
                4 + float(row["visits"]) / 100
            )
        )

        folium.CircleMarker(
            location=[
                row["latitude"],
                row["longitude"]
            ],
            radius=radius,
            tooltip=(
                f"{row['area']} | "
                f"Visits: {row['visits']}"
            ),
            popup=(
                f"Area: {row['area']}<br>"
                f"Visits: {row['visits']}"
            ),
            fill=True
        ).add_to(
            investigation_map
        )

    # ========================================================
    # TOP PREDICTED AREAS
    # ========================================================

    for _, row in predictions.head(5).iterrows():

        match = area_centers[
            area_centers["area"]
            == row["Area"]
        ]

        if not match.empty:

            point = match.iloc[0]

            folium.Marker(
                [
                    point["latitude"],
                    point["longitude"]
                ],
                tooltip=(
                    f"Predicted Area: "
                    f"{row['Area']}"
                ),
                popup=(
                    f"Predicted Probability: "
                    f"{row['Probability (%)']:.2f}%"
                ),
                icon=folium.Icon(
                    icon="flag"
                )
            ).add_to(
                investigation_map
            )

    # ========================================================
    # ANOMALOUS LOCATIONS
    # ========================================================

    for _, row in (
        data[data["anomaly"] == 1]
        .head(150)
        .iterrows()
    ):

        folium.CircleMarker(
            location=[
                row["latitude"],
                row["longitude"]
            ],
            radius=3,
            tooltip="Anomalous movement",
            popup=(
                f"Area: {row['area']}<br>"
                f"Anomaly Score: "
                f"{row['anomaly_score']:.3f}"
            ),
            fill=True
        ).add_to(
            investigation_map
        )

    # ========================================================
    # MARKOV ROUTE LINE
    # ========================================================

    if last_known_area in markov:

        route = predict_route(
            markov,
            last_known_area,
            steps=4
        )

        route_points = []

        for area in route:

            match = area_centers[
                area_centers["area"]
                == area
            ]

            if not match.empty:

                point = match.iloc[0]

                route_points.append([
                    point["latitude"],
                    point["longitude"]
                ])

        if len(route_points) >= 2:

            folium.PolyLine(
                route_points,
                tooltip="Markov predicted route",
                weight=5
            ).add_to(
                investigation_map
            )

    st_folium(
        investigation_map,
        width=None,
        height=650
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.warning(
    "ETHICAL REQUIREMENT: This is an academic simulation. "
    "Predictions are probabilistic. Anomalies do not establish "
    "criminal or suspicious behavior, and ML output does not "
    "prove a person's location."
)

st.caption(
    "CASEFILE AI — Advanced Machine Learning Individual Project"
)
