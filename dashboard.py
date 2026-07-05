import streamlit as st
import pandas as pd
import sqlite3
import os


st.header("📊 Dashboard")

# Connect to database
conn = sqlite3.connect("detections.db")

df = pd.read_sql_query("SELECT * FROM detections", conn)

conn.close()

if df.empty:
    st.warning("No detections found.")
    st.stop()

# Statistics
total = len(df)
real = len(df[df["prediction"] == "REAL"])
fake = len(df[df["prediction"] == "FAKE"])
avg_conf = round(df["confidence"].mean(), 2)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Detections", total)
col2.metric("REAL", real)
col3.metric("FAKE", fake)
col4.metric("Average Confidence", f"{avg_conf}%")

st.divider()

# Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Prediction Distribution")
    st.bar_chart(df["prediction"].value_counts())

with col2:
    st.subheader("Confidence")
    st.line_chart(df["confidence"])

st.divider()

st.subheader("Detection History")

st.dataframe(df, width="stretch")

st.divider()

st.subheader("Saved Fake Screenshots")

if os.path.exists("screenshots"):

    images = sorted(os.listdir("screenshots"), reverse=True)

    if images:

        cols = st.columns(3)

        for i, img in enumerate(images):

            with cols[i % 3]:
                st.image(
                    os.path.join("screenshots", img),
                    caption=img,
                    width="stretch"
                )

    else:
        st.info("No fake screenshots saved yet.")