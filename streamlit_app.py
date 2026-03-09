import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Kenya Data Visualization Hub",
    page_icon="🇰🇪",
    layout="wide"
)

# ------------------------------------------------
# TITLE
# ------------------------------------------------
st.title("🇰🇪 Kenya Data Visualization Hub")
st.markdown(
"""
### Explore Kenyan population trends, insights and predictions
Interactive analytics powered by **Python, Streamlit and Machine Learning**
"""
)

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/raw/population.csv")

data = load_data()

# ------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------
st.sidebar.header("🔎 Filter Data")

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

# ------------------------------------------------
# KPI DASHBOARD
# ------------------------------------------------
st.subheader("📊 Key Metrics")

c1, c2, c3 = st.columns(3)

c1.metric("Start Year", int(filtered_data["Year"].min()))
c2.metric("End Year", int(filtered_data["Year"].max()))

latest_pop = filtered_data["Population"].iloc[-1]
c3.metric("Latest Population", f"{latest_pop:,}")

# ------------------------------------------------
# MAIN CHARTS
# ------------------------------------------------
col1, col2 = st.columns(2)

# Population Trend
with col1:

    fig = px.line(
        filtered_data,
        x="Year",
        y="Population",
        markers=True,
        title="Kenya Population Trend"
    )

    st.plotly_chart(fig, use_container_width=True)

# Growth Rate
with col2:

    filtered_data["Growth %"] = filtered_data["Population"].pct_change() * 100

    fig2 = px.bar(
        filtered_data,
        x="Year",
        y="Growth %",
        title="Population Growth Rate (%)"
    )

    st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------
# MACHINE LEARNING PREDICTION
# ------------------------------------------------
st.subheader("🤖 Population Prediction")

X = data[["Year"]]
y = data["Population"]

model = LinearRegression()
model.fit(X, y)

future_year = st.slider(
    "Predict Population For Year",
    int(data["Year"].max()) + 1,
    2050
)

prediction = model.predict(np.array([[future_year]]))

st.success(
    f"Estimated Population in {future_year}: {int(prediction[0]):,}"
)

# ------------------------------------------------
# FORECAST VISUALIZATION
# ------------------------------------------------
future_df = pd.DataFrame({
    "Year": [future_year],
    "Population": [prediction[0]]
})

combined = pd.concat([data, future_df])

fig3 = px.line(
    combined,
    x="Year",
    y="Population",
    markers=True,
    title="Population Trend + Forecast"
)

st.plotly_chart(fig3, use_container_width=True)

# ------------------------------------------------
# DATASET TABLE
# ------------------------------------------------
st.subheader("📂 Dataset Preview")

st.dataframe(filtered_data, use_container_width=True)

# ------------------------------------------------
# DOWNLOAD DATA
# ------------------------------------------------
st.download_button(
    "⬇ Download Filtered Dataset",
    filtered_data.to_csv(index=False),
    "kenya_population_data.csv",
    "text/csv"
)
