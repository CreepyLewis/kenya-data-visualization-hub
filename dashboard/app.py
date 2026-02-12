import streamlit as st
import pandas as pd
import plotly.express as px

st.title("🇰🇪 Kenya Data Visualization Hub")

data = pd.read_csv("data/raw/population.csv")

st.subheader("Population Growth")

fig = px.line(
    data,
    x="Year",
    y="Population",
    markers=True
)

st.plotly_chart(fig)

st.write("Dataset Preview")
st.dataframe(data)
