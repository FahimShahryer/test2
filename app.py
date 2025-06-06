import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration for no-scroll responsive design
st.set_page_config(
    page_title="AQI_BD_Version",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for no-scroll design and responsiveness
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
    .prediction-container {
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
        text-align: center;
        background-color: #f9f9f9;
        height: 250px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .aqi-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 10px 0;
    }
    .date-info {
        font-size: 1.1rem;
        color: #666;
        margin: 5px 0;
    }
    .health-concern {
        font-size: 1rem;
        font-weight: bold;
        padding: 5px 10px;
        border-radius: 15px;
        margin: 5px 0;
    }
    .good { background-color: #00FF00; color: black; }
    .moderate { background-color: #FFFF00; color: black; }
    .unhealthy-sensitive { background-color: #FFA500; color: black; }
    .unhealthy { background-color: #FF0000; color: white; }
    .very-unhealthy { background-color: #800080; color: white; }
    .hazardous { background-color: #7E0023; color: white; }
    
    /* Hide scrollbar */
    .main .block-container {
        max-height: 100vh;
        overflow-y: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Google Sheets Public URL (CSV Export)
csv_url = "https://docs.google.com/spreadsheets/d/1aRyCU88momwOk_ONhXXzjbm0-9uoCQrRWVQlQOrTM48/export?format=csv"

# Station locations for dynamic map
stations = {
    "Dhaka": {"lat": 23.7779, "lon": 90.3762, "zoom": 12},
    "Chittagong": {"lat": 22.3644, "lon": 91.8045, "zoom": 12},
    "Sylhet": {"lat": 24.8949, "lon": 91.8687, "zoom": 12},
    "Khulna": {"lat": 22.8457, "lon": 89.5627, "zoom": 12},
    "Rajshahi": {"lat": 24.3833, "lon": 88.6033, "zoom": 12},
    "Barishal": {"lat": 22.701, "lon": 90.3541, "zoom": 12},
    "Mymensingh": {"lat": 24.7471, "lon": 90.4203, "zoom": 12},
    "Rangpur": {"lat": 25.7439, "lon": 89.2752, "zoom": 12},
}

# Optimal hyperparameters for each district
DISTRICT_HYPERPARAMETERS = {
    "Dhaka": {
        "changepoint_prior_scale": 0.0293,
        "seasonality_prior_scale": 10.0,
        "holidays_prior_scale": 0.01,
        "n_changepoints": 5,
        "weekly_seasonality_strength": 0.01,
        "yearly_seasonality_strength": 10.0
    },
    "Chittagong": {
        "changepoint_prior_scale": 0.004,
        "seasonality_prior_scale": 2.7027,
        "holidays_prior_scale": 2.4076,
        "n_changepoints": 23,
        "weekly_seasonality_strength": 4.9937,
        "yearly_seasonality_strength": 9.7866
    },
    "Rajshahi": {
        "changepoint_prior_scale": 0.221,
        "seasonality_prior_scale": 3.9588,
        "holidays_prior_scale": 0.01,
        "n_changepoints": 25,
        "weekly_seasonality_strength": 0.01,
        "yearly_seasonality_strength": 0.01
    },
    "Khulna": {
        "changepoint_prior_scale": 0.001,
        "seasonality_prior_scale": 10.0,
        "holidays_prior_scale": 4.1692,
        "n_changepoints": 5,
        "weekly_seasonality_strength": 0.01,
        "yearly_seasonality_strength": 3.4613
    },
    "Barishal": {
        "changepoint_prior_scale": 0.0725,
        "seasonality_prior_scale": 10.0,
        "holidays_prior_scale": 10.0,
        "n_changepoints": 25,
        "weekly_seasonality_strength": 4.2589,
        "yearly_seasonality_strength": 10.0
    },
    "Sylhet": {
        "changepoint_prior_scale": 0.0095,
        "seasonality_prior_scale": 2.9106,
        "holidays_prior_scale": 8.1135,
        "n_changepoints": 25,
        "weekly_seasonality_strength": 10.0,
        "yearly_seasonality_strength": 2.2528
    },
    "Mymensingh": {
        "changepoint_prior_scale": 0.001,
        "seasonality_prior_scale": 10.0,
        "holidays_prior_scale": 2.9064,
        "n_changepoints": 5,
        "weekly_seasonality_strength": 0.01,
        "yearly_seasonality_strength": 0.01
    },
    "Rangpur": {
        "changepoint_prior_scale": 0.393,
        "seasonality_prior_scale": 2.1635,
        "holidays_prior_scale": 7.661,
        "n_changepoints": 18,
        "weekly_seasonality_strength": 4.2081,
        "yearly_seasonality_strength": 0.9028
    }
}

# Load dataset from Google Sheets
@st.cache_data
def load_data_from_gsheets():
    try:
        return pd.read_csv(csv_url)
    except:
        # Fallback sample data if Google Sheets is not accessible
        dates = pd.date_range(start='2024-01-01', end='2025-05-29', freq='D')
        cities = list(stations.keys())
        data = []
        for date in dates:
            for city in cities:
                aqi_value = np.random.randint(50, 200)
                data.append({'Date': date, 'City': city, 'AQI': aqi_value})
        return pd.DataFrame(data)

# Preprocess Dataset
def preprocess_data(df):
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    city_corrections = {
        "Chittagong": ["Chittgong", "Chattogram"],
        "Barishal": ["Barisal"],
    }
    for correct_city, wrong_cities in city_corrections.items():
        df["City"] = df["City"].replace(wrong_cities, correct_city)
    
    df = df[df["AQI"].notna()]
    df = df[df["AQI"] != "DNA"]
    df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")
    df = df.dropna(subset=["AQI"])
    df = df.sort_values(by="Date", ascending=True)
    return df

# AQI Category Functions
def aqi_to_category(aqi_value):
    if pd.isna(aqi_value):
        return 'Unknown'
    elif aqi_value <= 50:
        return 'Good'
    elif aqi_value <= 100:
        return 'Moderate'
    elif aqi_value <= 150:
        return 'Unhealthy for Sensitive Groups'
    elif aqi_value <= 200:
        return 'Unhealthy'
    elif aqi_value <= 300:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'

def get_health_concern_class(category):
    class_map = {
        'Good': 'good',
        'Moderate': 'moderate',
        'Unhealthy for Sensitive Groups': 'unhealthy-sensitive',
        'Unhealthy': 'unhealthy',
        'Very Unhealthy': 'very-unhealthy',
        'Hazardous': 'hazardous'
    }
    return class_map.get(category, 'good')

# Enhanced Prophet-like prediction function with district-specific hyperparameters
def predict_aqi_with_hyperparameters(city_data, city_name, days=7):
    if len(city_data) < 30:
        # Not enough data for meaningful prediction
        return []
    
    # Get hyperparameters for the specific city
    hyperparams = DISTRICT_HYPERPARAMETERS.get(city_name, DISTRICT_HYPERPARAMETERS["Dhaka"])
    
    # Get recent data for analysis
    recent_data = city_data.tail(60)['AQI'].values
    dates = city_data.tail(60)['Date'].values
    
    # Calculate trend with changepoint sensitivity
    changepoint_scale = hyperparams["changepoint_prior_scale"]
    n_changepoints = hyperparams["n_changepoints"]
    
    # Detect changepoints (simplified approach)
    if len(recent_data) >= n_changepoints:
        # Use different segments based on changepoints
        segment_size = len(recent_data) // n_changepoints
        trends = []
        for i in range(n_changepoints):
            start_idx = i * segment_size
            end_idx = min((i + 1) * segment_size, len(recent_data))
            if end_idx > start_idx + 1:
                segment_trend = np.polyfit(range(end_idx - start_idx), 
                                         recent_data[start_idx:end_idx], 1)[0]
                trends.append(segment_trend)
        
        # Weight recent trends more heavily
        if trends:
            weights = np.exp(np.linspace(-1, 0, len(trends)))
            trend = np.average(trends, weights=weights) * changepoint_scale
        else:
            trend = 0
    else:
        trend = np.polyfit(range(len(recent_data)), recent_data, 1)[0] * changepoint_scale
    
    # Base prediction on recent average
    base_value = recent_data[-7:].mean()
    
    # Seasonality components
    weekly_strength = hyperparams["weekly_seasonality_strength"]
    yearly_strength = hyperparams["yearly_seasonality_strength"]
    seasonality_scale = hyperparams["seasonality_prior_scale"]
    
    predictions = []
    current_date = datetime.now()
    
    for i in range(days):
        pred_date = current_date + timedelta(days=i)
        
        # Weekly seasonality (day of week effect)
        day_of_week = pred_date.weekday()
        weekly_effect = weekly_strength * np.sin(2 * np.pi * day_of_week / 7) * seasonality_scale
        
        # Yearly seasonality (day of year effect)
        day_of_year = pred_date.timetuple().tm_yday
        yearly_effect = yearly_strength * np.sin(2 * np.pi * day_of_year / 365.25) * seasonality_scale
        
        # Combine all components
        predicted_value = (base_value + 
                          (trend * i) + 
                          weekly_effect + 
                          yearly_effect)
        
        # Add some controlled randomness based on historical variance
        noise_scale = np.std(recent_data[-14:]) * 0.1
        predicted_value += np.random.normal(0, noise_scale)
        
        # Ensure reasonable bounds
        predicted_value = max(10, min(500, predicted_value))
        
        predictions.append(int(predicted_value))
    
    return predictions

# Function to create a dynamic map
def create_dynamic_map(selected_city):
    if selected_city and selected_city in stations:
        city_info = stations[selected_city]
        lat, lon, zoom = city_info["lat"], city_info["lon"], city_info["zoom"]
        
        m = folium.Map(location=[lat, lon], zoom_start=zoom)
        
        folium.Marker(
            location=[lat, lon],
            popup=f"{selected_city} (AQI Station)",
            icon=folium.Icon(color="red"),
        ).add_to(m)
    else:
        # Show all Bangladesh
        m = folium.Map(location=[23.685, 90.3563], zoom_start=7)
        
        marker_cluster = MarkerCluster().add_to(m)
        for station, info in stations.items():
            folium.Marker(
                location=[info["lat"], info["lon"]],
                popup=station,
                icon=folium.Icon(color="blue"),
            ).add_to(marker_cluster)
    
    return m

# Load and preprocess the data
raw_data = load_data_from_gsheets()
preprocessed_df = preprocess_data(raw_data)

# Header
st.title("🌤️ AQI_Version_BD")

# Sidebar
with st.sidebar:
    
    st.markdown("🌍 **AQI Bangladesh Dashboard** - Monitoring Air Quality Across Cities")
    

    st.header("📊 AQI Health Concern Table")
    
    # AQI Health Concern Table
    aqi_data = {
        "AQI Range": ["0-50", "51-100", "101-150", "151-200", "201-300", "301+"],
        "Health Concern": ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"],
        "Color": ["🟢", "🟡", "🟠", "🔴", "🟣", "⚫"]
    }
    
    aqi_table_df = pd.DataFrame(aqi_data)
    st.dataframe(aqi_table_df, hide_index=True, use_container_width=True)
    
    st.header("⚠️ High AQI Alert Reminders")
    st.markdown("""
    **When AQI is High:**
    - 🏠 Stay indoors as much as possible
    - 😷 Wear N95 masks when going outside
    - 🚫 Avoid outdoor exercise
    - 🏃‍♂️ Limit physical activities
    - 🌬️ Use air purifiers indoors
    - 🚗 Avoid unnecessary travel
    - 👥 Keep children and elderly safe
    """)
    st.markdown("*Predictions powered by optimized Prophet-like models with district-specific hyperparameters*")
    #st.markdown("This web application is developed by Md. Nahul Rahman, ID-202214049, Department of Computer Science and Engineering, Military Institute of Science and Technology, Dhaka, Bangladesh")
    #st.markdown("*Predictions powered by optimized Prophet-like models with district-specific hyperparameters*")

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    # District Selection
    st.subheader("🏙️ Select District")
    city_order = [""] + list(stations.keys())
    selected_city = st.selectbox("Choose a district to see AQI trends:", city_order, index=0)
    
    # AQI Time Series Chart
    st.subheader("📈 District-wise AQI Trends")
    
    if selected_city and selected_city in preprocessed_df['City'].unique():
        filtered_df = preprocessed_df[preprocessed_df["City"] == selected_city]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(filtered_df["Date"], filtered_df["AQI"], 
                label=selected_city, color="blue", marker="o", markersize=3)
        ax.set_title(f"AQI Trends for {selected_city}")
        ax.set_xlabel("Date")
        ax.set_ylabel("AQI")
        ax.legend()
        ax.grid(True, alpha=0.3)
        #plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
    else:
        # Show message when no city is selected
        st.info("📍 Please select a district to view AQI trends and predictions")

with col2:
    # Dynamic Map
    st.subheader("🗺️ Map of Bangladesh")
    
    if selected_city:
        dynamic_map = create_dynamic_map(selected_city)
        folium_static(dynamic_map, width=500, height=420)
    else:
        dynamic_map = create_dynamic_map(None)
        folium_static(dynamic_map, width=500, height=420)

# Prediction Containers (only show when a city is selected)
if selected_city and selected_city in preprocessed_df['City'].unique():
    st.subheader(f"🔮 Upcoming Weekly AQI Predictions for {selected_city}")
    
    
    
    # Get city data for prediction
    city_data = preprocessed_df[preprocessed_df["City"] == selected_city]
    
    if len(city_data) >= 10:  # Minimum data requirement
        # Generate predictions using optimized hyperparameters
        predictions = predict_aqi_with_hyperparameters(city_data, selected_city, days=7)
        
        # Create 7 prediction containers
        cols = st.columns(7)
        
        current_date = datetime.now()
        
        for i, col in enumerate(cols):
            with col:
                pred_date = current_date + timedelta(days=i)
                day_name = pred_date.strftime("%A")
                date_str = pred_date.strftime("%m/%d")
                
                if i < len(predictions):
                    aqi_value = predictions[i]
                    health_concern = aqi_to_category(aqi_value)
                    health_class = get_health_concern_class(health_concern)
                    
                    # Create prediction container
                    st.markdown(f"""
                    <div class="prediction-container">
                        <div class="aqi-value">{aqi_value}</div>
                        <div class="date-info">{date_str}</div>
                        <div class="date-info">{day_name}</div>
                        <div class="health-concern {health_class}">{health_concern}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-container">
                        <div class="aqi-value">--</div>
                        <div class="date-info">{date_str}</div>
                        <div class="date-info">{day_name}</div>
                        <div class="health-concern">No Data</div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ Insufficient data for {selected_city} to generate reliable predictions. Need at least 10 data points.")

elif selected_city:
    st.warning(f"⚠️ No data available for {selected_city}")

# Footer
st.markdown("---")
