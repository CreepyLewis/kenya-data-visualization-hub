import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Kenya Data Hub",
    page_icon="🇰🇪",
    layout="wide"
)

st.title("🇰🇪 Kenya Data Visualization Hub")

# Sidebar filters
st.sidebar.header("Filters")

data = pd.read_csv("data/raw/population.csv")

year_range = st.sidebar.slider(
    "Select Year Range",
    int(data["Year"].min()),
    int(data["Year"].max()),
    (2000, 2023)
)

filtered = data[
    (data["Year"] >= year_range[0]) &
    (data["Year"] <= year_range[1])
]

col1, col2 = st.columns(2)

with col1:
    fig = px.line(filtered, x="Year", y="Population",
                  title="Population Trend",
                  markers=True)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.metric(
        "Latest Population",
        filtered["Population"].iloc[-1]
    )

st.dataframe(filtered)
