# importing all libraries
import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.set_page_config(page_title="Predictions", page_icon="🪄", layout="wide")

# backend fastapi endpoint URL
backendAPI = "http://localhost:8000/predict/"

st.title("NYC Yellow Taxi Fare Prediction")
st.divider()

st.write("Upload a CSV file to get predictions")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # read the CSV file
    df = pd.read_csv(uploaded_file)
    # display the uploaded data
    st.subheader("Sample of Uploaded Data")
    st.write(df.head())
    # sending the file to the endpoint
    files = {"file": ("input.csv", uploaded_file.getvalue(), "text/csv")}
    response = requests.post(backendAPI, files=files)

    # if the response is 200 then display the predictions
    if response.status_code == 200:
        # parsing the JSON response
        predictions = pd.DataFrame(response.json())
        # displaying the predictions
        st.subheader("Predictions")
        st.write(predictions)

        # computing model's performance
        mae = (predictions['fare_amount'] - predictions['prediction']).abs().mean()
        mse = ((predictions['fare_amount'] - predictions['prediction']) ** 2).mean()

        st.subheader("Model Performance")
        col1, col2 = st.columns(2)
        col1.metric("Mean Absolute Error", f"{mae:.2f}")
        col2.metric("Mean Squared Error", f"{mse:.2f}")

        # displaying a scatter plot of actual vs predicted fares
        fig = px.scatter(predictions, x='fare_amount', y='prediction', labels={'fare_amount': 'Actual Fare', 'prediction': 'Predicted Fare'}, title='Actual vs Predicted Fares')
        fig.add_shape(type='line', x0=0, y0=0,
                      x1=max(predictions['fare_amount'].max(), predictions['prediction'].max()),
                      y1=max(predictions['fare_amount'].max(), predictions['prediction'].max()),
                      line=dict(color='red', dash='dash'))
        st.plotly_chart(fig)

    else:
        st.error(f"Error: {response.status_code} - {response.text}")
