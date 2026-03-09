import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
import os
import openai

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Kenya AI Data Platform",
    page_icon="🇰🇪",
    layout="wide"
)

st.title("🇰🇪 Kenya AI Data Platform")
st.markdown(
"""
Interactive platform for **Kenya population analytics**, forecasting, county maps, and AI insights.
"""
)

# -------------------------------
# DATA LOADING FUNCTIONS
# -------------------------------
@st.cache_data
def load_population():
    path = "data/raw/population.csv"
    if not os.path.exists(path):
        st.error("Dataset not found. Upload population.csv to data/raw/")
        st.stop()
    return pd.read_csv(path)

@st.cache_data
def load_counties():
    path = "data/raw/counties.csv"
    if not os.path.exists(path):
        st.warning("County dataset not found. Map will be disabled.")
        return None
    return pd.read_csv(path)

population = load_population()
counties = load_counties()

# -------------------------------
# SIDEBAR NAVIGATION
# -------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "📊 Population Dashboard",
        "🤖 AI Forecast",
        "🗺 County Map",
        "📂 Dataset Explorer",
        "💬 AI Data Assistant"
    ]
)

# ===============================
# POPULATION DASHBOARD
# ===============================
if page == "📊 Population Dashboard":
    st.subheader("Kenya Population Analytics")

    year_range = st.slider(
        "Select Year Range",
        int(population["Year"].min()),
        int(population["Year"].max()),
        (int(population["Year"].min()), int(population["Year"].max()))
    )

    filtered = population[
        (population["Year"] >= year_range[0]) &
        (population["Year"] <= year_range[1])
    ]

    # KPI metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Start Year", int(filtered["Year"].min()))
    c2.metric("End Year", int(filtered["Year"].max()))
    c3.metric("Latest Population", f"{filtered['Population'].iloc[-1]:,}")
    growth_rate = filtered["Population"].pct_change().mean() * 100
    c4.metric("Average Growth %", f"{growth_rate:.2f}%")

    col1, col2 = st.columns(2)
    # Population trend
    with col1:
        fig = px.line(
            filtered,
            x="Year",
            y="Population",
            markers=True,
            title="Population Trend"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Growth rate
    with col2:
        filtered["Growth %"] = filtered["Population"].pct_change() * 100
        fig2 = px.bar(
            filtered,
            x="Year",
            y="Growth %",
            title="Population Growth Rate"
        )
        st.plotly_chart(fig2, use_container_width=True)

# ===============================
# AI FORECAST
# ===============================
elif page == "🤖 AI Forecast":
    st.subheader("AI Population Forecast")

    X = population[["Year"]]
    y = population["Population"]
    model = LinearRegression()
    model.fit(X, y)

    future_year = st.slider(
        "Select Future Year",
        int(population["Year"].max()) + 1,
        2050
    )

    prediction = model.predict(np.array([[future_year]]))
    st.success(f"Estimated population in {future_year}: {int(prediction[0]):,}")

    # Forecast chart
    future_df = pd.DataFrame({"Year": [future_year], "Population": [prediction[0]]})
    combined = pd.concat([population, future_df])
    fig = px.line(
        combined,
        x="Year",
        y="Population",
        markers=True,
        title="Population Trend + Forecast"
    )
    st.plotly_chart(fig, use_container_width=True)

# ===============================
# COUNTY MAP
# ===============================
elif page == "🗺 County Map":
    st.subheader("Kenya County Population Map")
    if counties is None:
        st.warning("County dataset not available.")
    else:
        fig = px.choropleth(
            counties,
            geojson="assets/kenya.geojson",
            locations="County",
            featureidkey="properties.county",  # adjust if your GeoJSON uses a different property name
            color="Population",
            color_continuous_scale="Viridis",
            title="Population by County"
        )
        fig.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig, use_container_width=True)

# ===============================
# DATASET EXPLORER
# ===============================
elif page == "📂 Dataset Explorer":
    st.subheader("Dataset Explorer")
    st.dataframe(population, use_container_width=True)
    st.download_button(
        "⬇ Download Dataset",
        population.to_csv(index=False),
        "kenya_population.csv",
        "text/csv"
    )

# ===============================
# AI DATA ASSISTANT (ChatCompletion API)
# ===============================
elif page == "💬 AI Data Assistant":
    st.subheader("Ask Questions About the Dataset")

    question = st.text_input("Type your question here:")

    if question:
        # Load API key safely from Streamlit secrets
        openai.api_key = st.secrets["GROQ_API_KEY"]

        # Prepare dataset preview for GPT
        dataset_preview = population.tail(20).to_dict(orient="records")
        prompt = f"""
        You are a helpful data assistant for Kenya population dataset.
        Dataset preview: {dataset_preview}
        Answer the following question based on this data:
        {question}
        """

        # New ChatCompletion API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful data assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=200
        )

        # Display GPT answer
        st.info(response.choices[0].message.content.strip())
