
# Flight Fare Prediction

What decides the price of a domestic flight in India, and can we predict it? I took a messy dataset of 300,000 flights and built the whole thing myself: cleaning, SQL, EDA, feature engineering, machine learning and a live web app.

**Live app:** https://8p4ftry4sjbnvvvhipawcm-flightfareprediction.streamlit.app/

![Overview page](images/overview.png)

## Results at a glance

| | |
|---|---|
| Raw rows | 300,561 |
| Rows after cleaning | 262,642 |
| Best model | Random Forest |
| Test R2 | 0.984 |
| Test MAE | Rs 1,323 |
| Baseline MAE (class median) | Rs 4,882 |

The Random Forest cuts the baseline error by about 73%. Every score is measured on a test set that shares no identical rows with the training set. I explain why that matters further down.

## The problem

Flight prices jump around, and a traveller has no way to know if a fare is fair. I wanted to find out which details push the price up or down, and then build a tool that gives a fair estimate from simple inputs like route, airline, class, stops and duration.

## The data

- Source: Goibibo flights dataset (Kaggle), 11 columns
- Period: 26 June 2023 to 13 August 2023
- Cities: Delhi, Mumbai, Bangalore, Kolkata, Hyderabad and Chennai (30 routes)
- Airlines: Vistara, Air India, Indigo, Go First, Airasia and Spicejet. Two very small airlines and the unknown names are grouped as "Other".

## What I did

### 1. Data cleaning

The raw file was a mess, and this was the longest part of the project.

- Dates came in three different formats (`10 Jul 2023`, `29-06-2023`, `2023/07/19`)
- Airline names had mixed capitals and extra spaces (`AIR INDIA`, `GO `)
- Departure times mixed AM/PM with the 24 hour format
- Price and duration were stored as text
- 257 duplicate rows
- Missing values in every column, between 2% and 5%

I fixed the formats, cleaned the names, removed the duplicates and dropped rows that had no usable price, date, class, route or duration. Where only the airline, stops or departure time was missing, I kept the row and marked it "Unknown". The final data has 262,642 rows, so 12.6% was removed.

I also looked at price outliers (3.9% of business and 6.7% of economy fares by the IQR rule). When I checked the extreme rows, they were real fares, like a long Vistara business trip with two stops, so I kept them.

### 2. SQL analysis

I loaded the clean data into MySQL and answered questions with queries. The full file is [`project_sql.sql`](project_sql.sql).

- Average, minimum and maximum price by class
- Airline ranking by price inside each class (CTE and `RANK()`)
- Routes where business costs the most times more than economy
- Top 3 most expensive routes in each class (`ROW_NUMBER()`)
- Share of all flights for the busiest routes (`SUM() OVER`)
- Price and travel time by number of stops
- Price by time of day (`CASE WHEN`)
- Week over week price change (`LAG()`)
- Economy price quartiles (`NTILE()`)
- Which airlines charge more or less than their own route average (CTE and `JOIN`)
- An index on the route columns and a `route_summary` view

### 3. Exploratory data analysis

Here is what the data showed.

- **Class is the biggest factor.** The median business fare is Rs 53,961 and the median economy fare is Rs 5,830. That is about 9 times more.
- **Business class is mostly Vistara and Air India.** Other airlines almost never show business fares in this data.
- **More stops means a higher price.** Non-stop flights are the cheapest in both classes.
- **Night flights are cheaper in economy.** Around Rs 4,500, against Rs 5,700 to 6,000 at other times of the day.
- **Day of the week makes no difference.** The bars are almost flat.
- **Prices fell in the first three weeks and then stayed steady.** Economy went from about Rs 14,000 to about Rs 5,000.
- **Routes from Kolkata are the most expensive in economy.**

![Price insights](images/price-insights.png)

### 4. Feature engineering

- `day_index`: number of days since the first flight date
- `is_business`: 1 for business, 0 for economy
- `time_of_day`: Morning, Afternoon, Evening or Night, taken from the departure time
- `duration_min`: duration converted into minutes
- One hot encoding for airline, source city, destination city, stops and time of day

This gave 26 input features.

### 5. Machine learning

**Why I did not use a normal train test split**

A lot of rows in this data have exactly the same features. In total, 22,583 rows had an identical twin somewhere else. With a random split, one twin can land in training and the other in the test set, and then the model only looks smart because it has already seen the answer.

To avoid this, I gave every unique feature combination a group number and used `GroupShuffleSplit`. A whole group goes either to train or to test, never both. I then checked it, and there were zero identical rows across the split (210,109 rows for training and 52,533 for testing).

**Results**

| Model | Test R2 | Test MAE | Test RMSE |
|---|---|---|---|
| Baseline (class median) | 0.878 | 4,882 | 8,061 |
| **Random Forest** | **0.984** | **1,323** | **2,952** |
| Extra Trees | 0.982 | 1,480 | 3,109 |
| LightGBM | 0.979 | 1,919 | 3,354 |
| XGBoost | 0.973 | 2,144 | 3,769 |
| Average (RF + XGB) | 0.982 | 1,643 | 3,126 |

**What the numbers really say**

- A baseline that only knows the class already gets R2 0.878. So the real value of the model is how well it predicts inside each class.
- Economy: MAE of Rs 776, which is 13.2% of the median price. 83% of predictions are within 20% of the real fare.
- Business: MAE of Rs 2,530, which is 4.7% of the median price. 96% of predictions are within 20% of the real fare.
- The errors are largest in the first week of the data, when prices were changing the fastest.
- Feature importance: `is_business` 88.8%, `duration_min` 5.9%, `day_index` 1.6%. Everything else is small.

![Model page](images/model.png)

### 6. Streamlit app

The app has five pages: Overview, Price Insights, Model, Predict and About. The sidebar lets you filter by class and airline.

On the Predict page you choose the route, class, airline, stops, departure time, duration and date. The app shows the predicted price, the real fare range on that route, and the same trip priced on every airline.

![Predict page](images/predict.png)

The app trains a lighter Random Forest (60 trees) when it starts, so its numbers are a little different from the notebook (R2 0.983, MAE 1,343).

## Limits

- It has only about 7 weeks of data, so it cannot learn seasons or festivals
- It covers only 6 cities
- The data has no "days before departure" column, which usually matters a lot for fares
- It should not be used for dates outside the data range

## Tools used

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, LightGBM, Plotly, Streamlit, MySQL, Google Colab

## Run it on your computer

```bash
git clone https://github.com/AyushDagar/FLIGHT-FARE-PREDICTION.git
cd FLIGHT-FARE-PREDICTION
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
.
├── app.py                              Streamlit app
├── flights_clean.csv                   Cleaned data used by the app
├── flight_fare_prediction_final.ipynb  Cleaning, EDA, features and models
├── project_sql.sql                     MySQL queries
├── requirements.txt
├── images/                             Screenshots used in this README
└── README.md
```

## About me

I am Ayush Dagar, and I am looking for data analyst and data scientist roles. If you have feedback on this project or an opening, I would be glad to hear from you.

[GitHub](https://github.com/AyushDagar) | [LinkedIn](https://www.linkedin.com/in/ayush-dagar-9401a7230/)
