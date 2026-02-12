import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Kenya Data Visualization Hub",
    page_icon="🇰🇪",
    layout="wide"
)

st.title("🇰🇪 Kenya Data Visualization Hub")
st.write("Interactive analytics of Kenyan public datasets")

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/raw/population.csv")

data = load_data()

# -------------------------
# Sidebar Filters
# -------------------------
st.sidebar.header("Filter Data")

year_range = st.sidebar.slider(
    "Select Year Range",
    int(data["Year"].min()),
    int(data["Year"].max()),
    (int(data["Year"].min()), int(data["Year"].max()))
)

filtered_data = data[
    (data["Year"] >= year_range[0]) &
    (data["Year"] <= year_range[1])
]

# -------------------------
# Layout Columns
# -------------------------
col1, col2 = st.columns(2)

# -------------------------
# Population Chart
# -------------------------
with col1:
    fig = px.line(
        filtered_data,
        x="Year",
        y="Population",
        markers=True,
        title="Kenya Population Trend"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Metrics Card
# -------------------------
with col2:
    latest_pop = filtered_data["Population"].iloc[-1]
    st.metric("Latest Population", f"{latest_pop:,}")

# -------------------------
# Machine Learning Prediction
# -------------------------
st.subheader("📈 Population Prediction")

X = data[["Year"]]
y = data["Population"]

model = LinearRegression()
model.fit(X, y)

future_year = st.slider("Predict Population for Year", 2024, 2040)

prediction = model.predict([[future_year]])

st.success(f"Estimated Population in {future_year}: {int(prediction[0]):,}")

# -------------------------
# Data Table
# -------------------------
st.subheader("Dataset Preview")
st.dataframe(filtered_data)
