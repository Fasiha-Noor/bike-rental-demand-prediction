import io
import urllib.request
import zipfile

import gradio as gr
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Official UCI Bike Sharing Dataset
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"


def load_model():
    """Download day.csv and train the same final Linear Regression model."""
    data = urllib.request.urlopen(DATA_URL).read()
    dataset_zip = zipfile.ZipFile(io.BytesIO(data))

    with dataset_zip.open("day.csv") as file:
        df = pd.read_csv(file)

    # Final model features used in the project
    X = df.drop(
        ["cnt", "casual", "registered", "dteday", "instant"],
        axis=1
    )
    y = df["cnt"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


# Train the model when the app starts
model = load_model()


def predict_bike_demand(
    season,
    year,
    month,
    holiday,
    weekday,
    workingday,
    weather,
    temperature,
    feels_like,
    humidity,
    windspeed,
):
    """Predict the daily bike rental demand from user inputs."""
    input_data = pd.DataFrame(
        {
            "season": [season],
            "yr": [year],
            "mnth": [month],
            "holiday": [holiday],
            "weekday": [weekday],
            "workingday": [workingday],
            "weathersit": [weather],
            "temp": [temperature],
            "atemp": [feels_like],
            "hum": [humidity],
            "windspeed": [windspeed],
        }
    )

    prediction = model.predict(input_data)[0]

    # Rental demand cannot be negative
    return round(max(0, prediction))


# Gradio application

demo = gr.Interface(
    fn=predict_bike_demand,
    inputs=[
        gr.Dropdown(
            choices=[1, 2, 3, 4],
            value=2,
            label="Season (1=Winter, 2=Spring, 3=Summer, 4=Fall)",
        ),
        gr.Dropdown(
            choices=[0, 1],
            value=1,
            label="Year (0=2011, 1=2012)",
        ),
        gr.Slider(
            minimum=1,
            maximum=12,
            step=1,
            value=6,
            label="Month",
        ),
        gr.Dropdown(
            choices=[0, 1],
            value=0,
            label="Holiday (0=No, 1=Yes)",
        ),
        gr.Slider(
            minimum=0,
            maximum=6,
            step=1,
            value=3,
            label="Weekday (0=Sunday, 6=Saturday)",
        ),
        gr.Dropdown(
            choices=[0, 1],
            value=1,
            label="Working Day (0=No, 1=Yes)",
        ),
        gr.Dropdown(
            choices=[1, 2, 3, 4],
            value=1,
            label="Weather (1=Clear, 2=Mist, 3=Light Rain/Snow, 4=Heavy Rain)",
        ),
        gr.Slider(
            minimum=0,
            maximum=1,
            step=0.01,
            value=0.7,
            label="Temperature (normalized)",
        ),
        gr.Slider(
            minimum=0,
            maximum=1,
            step=0.01,
            value=0.65,
            label="Feels-like Temperature (normalized)",
        ),
        gr.Slider(
            minimum=0,
            maximum=1,
            step=0.01,
            value=0.5,
            label="Humidity (normalized)",
        ),
        gr.Slider(
            minimum=0,
            maximum=1,
            step=0.01,
            value=0.2,
            label="Windspeed (normalized)",
        ),
    ],
    outputs=gr.Number(label="Predicted Bike Rental Demand"),
    title="🚲 Bike Rental Demand Prediction",
    description=(
        "Enter daily weather and time conditions to predict the expected "
        "number of bike rentals."
    ),
)


if __name__ == "__main__":
    demo.launch()
