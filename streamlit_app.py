import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Kenya AI Data Platform",
    page_icon="🇰🇪",
    layout="wide"
)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.title("🇰🇪 Kenya AI Data Platform")

st.markdown(
"""
Explore Kenyan public datasets using **interactive dashboards,
machine learning predictions and geographic visualization**.
"""
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_population():
    return pd.read_csv("data/raw/population.csv")

@st.cache_data
def load_counties():
    return pd.read_csv("data/raw/counties.csv")

population = load_population()

# ---------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "📊 Population Dashboard",
        "🤖 AI Forecast",
        "🗺 County Map",
        "📂 Dataset Explorer"
    ]
)

# ===================================================
# POPULATION DASHBOARD
# ===================================================
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
    c1, c2, c3 = st.columns(3)

    c1.metric("Start Year", int(filtered["Year"].min()))
    c2.metric("End Year", int(filtered["Year"].max()))
    c3.metric("Latest Population", f"{filtered['Population'].iloc[-1]:,}")

    col1, col2 = st.columns(2)

    # population trend
    with col1:
        fig = px.line(
            filtered,
            x="Year",
            y="Population",
            markers=True,
            title="Population Trend"
        )

        st.plotly_chart(fig, use_container_width=True)

    # growth rate
    with col2:

        filtered["Growth %"] = filtered["Population"].pct_change() * 100

        fig2 = px.bar(
            filtered,
            x="Year",
            y="Growth %",
            title="Population Growth Rate"
        )

        st.plotly_chart(fig2, use_container_width=True)

# ===================================================
# AI FORECAST
# ===================================================
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

    st.success(
        f"Estimated population in {future_year}: {int(prediction[0]):,}"
    )

    # forecast chart
    future_df = pd.DataFrame({
        "Year": [future_year],
        "Population": [prediction[0]]
    })

    combined = pd.concat([population, future_df])

    fig = px.line(
        combined,
        x="Year",
        y="Population",
        markers=True,
        title="Population Trend + Forecast"
    )

    st.plotly_chart(fig, use_container_width=True)

# ===================================================
# COUNTY MAP
# ===================================================
elif page == "🗺 County Map":

    st.subheader("Kenya County Population Map")

    counties = load_counties()

    fig = px.choropleth(
        counties,
        geojson="assets/kenya.geojson",
        locations="County",
        featureidkey="properties.county",
        color="Population",
        title="Population by County"
    )

    fig.update_geos(fitbounds="locations", visible=False)

    st.plotly_chart(fig, use_container_width=True)

# ===================================================
# DATASET EXPLORER
# ===================================================
elif page == "📂 Dataset Explorer":

    st.subheader("Dataset Explorer")

    st.dataframe(population, use_container_width=True)

    st.download_button(
        "⬇ Download Dataset",
        population.to_csv(index=False),
        "kenya_population.csv",
        "text/csv"
    )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")

st.markdown(
"""
Built with **Python, Streamlit, Plotly and Machine Learning**

Kenyan Open Data Analytics Platform
"""
)
