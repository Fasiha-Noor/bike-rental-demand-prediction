# 🚲 Bike Rental Demand Prediction

## Project Overview

This Machine Learning project predicts the daily number of bike rentals
using the Bike Sharing Dataset. The project follows a complete Machine
Learning workflow, including data exploration, data cleaning,
preprocessing, exploratory data analysis (EDA), model training,
evaluation, and a Gradio-based prediction application.

## Problem Statement

Bike rental demand changes according to conditions such as season,
weather, temperature, humidity, working days, and month. The purpose of
this project is to build a regression model that estimates expected
daily bike rental demand from these conditions.

## Dataset

The project uses the **Bike Sharing Dataset (`day.csv`)**.

-   **Records:** 730
-   **Original columns:** 16
-   **Target variable:** `cnt`
-   **Problem type:** Regression

The main columns are:

`instant`, `dteday`, `season`, `yr`, `mnth`, `holiday`, `weekday`,
`workingday`, `weathersit`, `temp`, `atemp`, `hum`, `windspeed`,
`casual`, `registered`, and `cnt`.

The target variable `cnt` represents the total number of bike rentals.

## Data Exploration

The dataset was explored using:

-   `head()`
-   `shape`
-   `columns`
-   `dtypes`
-   `info()`
-   `describe()`

The dataset contains 730 rows and 16 columns. The initial inspection
showed that `dteday` was stored as an object while the other variables
were numeric.

## Data Cleaning and Preprocessing

The following steps were performed:

1.  Checked missing values.
2.  Checked duplicate records.
3.  Removed duplicate records using `drop_duplicates()`.
4.  Converted `dteday` into datetime format.
5.  Extracted year, month, day, and day of week from the date.
6.  Removed the original `dteday` column after extracting date
    information.
7.  Applied one-hot encoding to categorical variables.
8.  Removed `casual` and `registered` from the modeling dataset because
    they are components of total rental count and could cause target
    leakage.

No missing values were found, and the duplicate check returned zero
duplicate rows.

## Exploratory Data Analysis (EDA)

EDA was performed to understand the relationship between bike rental
demand and different variables.

Visualizations included:

-   Temperature vs. total bike rental demand
-   Humidity vs. total bike rental demand
-   Windspeed vs. total bike rental demand
-   Actual vs. predicted bike rental demand

The project uses the daily dataset, so the analysis focuses on daily
demand rather than hourly demand.

## Machine Learning Model

### Linear Regression

Linear Regression was selected as the main regression algorithm.

For the final model, the following columns were excluded:

-   `cnt` --- target variable
-   `casual` --- component of the target
-   `registered` --- component of the target
-   `dteday` --- date column
-   `instant` --- ID-like column

The model uses these input features:

`season, yr, mnth, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed`

The data was divided into **80% training data and 20% testing data**
using `random_state=42`.

## Model Evaluation

The Linear Regression model was evaluated using MAE, MSE, RMSE, and R².

  Metric          Result
  ---------- -----------
  MAE             573.31
  MSE          554832.81
  RMSE            744.87
  R² Score        0.8379

The project also compares actual and predicted demand using an Actual
vs. Predicted Bike Rental Demand visualization.

## Prediction Application

A simple **Gradio** application was created to allow users to enter
conditions and receive a predicted bike rental demand.

The application accepts:

-   Season
-   Year
-   Month
-   Holiday
-   Weekday
-   Working Day
-   Weather
-   Temperature
-   Feels-like Temperature
-   Humidity
-   Windspeed

The application returns the predicted number of bike rentals.

### Input Note

Temperature, feels-like temperature, humidity, and windspeed are entered
in the representation expected by the application.

## Test Scenarios

The application was tested with different conditions.

### Scenario 1 --- Favorable Conditions

**Predicted bike rental demand: 3791 rentals**

### Scenario 2 --- Unfavorable Weather and Higher Humidity

**Predicted bike rental demand: 2577 rentals**

### Scenario 3 --- Winter Conditions

**Predicted bike rental demand: 3355 rentals**

These tests demonstrate that changing the input conditions produces
different predicted rental demand values.

## Technologies Used

-   Python
-   Google Colab
-   Pandas
-   NumPy
-   Scikit-learn
-   Matplotlib
-   Gradio
-   GitHub

## Project Workflow

``` text
Dataset Collection
       ↓
Data Exploration
       ↓
Data Cleaning
       ↓
Data Preprocessing
       ↓
Exploratory Data Analysis
       ↓
Linear Regression Model
       ↓
Model Evaluation
       ↓
Gradio Prediction Application
       ↓
GitHub / Deployment
```

## Project Files

``` text
bike-rental-demand-prediction/
│
├── Machine_learning_project_.ipynb
├── README.md
└── app.py
```

The notebook contains the data exploration, preprocessing, EDA, model
training, evaluation, and prediction application work.

## How to Run

1.  Open the notebook in Google Colab.
2.  Make sure the required Python libraries are available.
3.  Load the `day.csv` dataset.
4.  Run the notebook cells in order.
5.  Train the Linear Regression model.
6.  Run the Gradio application to make predictions.

## Deployment

The Gradio application was tested through a temporary public sharing
link from Google Colab. Permanent hosting can be completed using Hugging
Face Spaces.

## Results

The final Linear Regression evaluation results were:

-   **MAE:** 573.31
-   **MSE:** 554832.81
-   **RMSE:** 744.87
-   **R²:** 0.8379

The prediction application also produced different results for different
test scenarios, demonstrating that the model responds to changes in
input conditions.

## Conclusion

This project demonstrates a complete Machine Learning pipeline for a
regression problem. It covers dataset exploration, cleaning,
preprocessing, EDA, model training, evaluation, and application
development.

The Gradio application provides a simple interface where users can enter
seasonal and weather conditions and receive an estimated daily bike
rental demand.

## Author

**Fasiha Noor**
