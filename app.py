import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

st.set_page_config(page_title="Flight Fare Prediction", layout="wide")
T = "plotly_dark"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TIMES = ["Morning", "Afternoon", "Evening", "Night", "Unknown"]
CATS = ["airline", "from", "to", "stops", "time_of_day"]


@st.cache_data
def load():
    df = pd.read_csv("flights_clean.csv", parse_dates=["flight_date"])
    df["airline"] = df["airline"].replace({"Unknown": "other", "starair": "other", "trujet": "other"}).str.title()
    df["stops"] = df["stops"].astype(str)
    df["day_index"] = (df["flight_date"] - df["flight_date"].min()).dt.days
    df["dow"] = df["flight_date"].dt.day_name()
    df["route"] = df["from"].str.title() + " to " + df["to"].str.title()

    # departure hour -> time of day
    hr = pd.to_datetime(df["dep_time"], format="%H:%M", errors="coerce").dt.hour
    df["time_of_day"] = pd.cut(hr, [-1, 5, 11, 17, 21, 24],
                               labels=["Night", "Morning", "Afternoon", "Evening", "Night"],
                               ordered=False).astype(str).replace("nan", "Unknown")
    return df


def features(d):
    x = d[["duration_min", "day_index"]].copy()
    x["is_business"] = (d["class"] == "business").astype(int)
    x = pd.concat([x, d[CATS]], axis=1)
    return pd.get_dummies(x, columns=CATS, drop_first=True).astype(float)


@st.cache_resource
def train(df):
    X, y = features(df), df["price"]

    # grouped split so same-feature rows don't sit in both train and test
    groups = X.groupby(list(X.columns)).ngroup()
    tr, te = next(GroupShuffleSplit(test_size=0.2, random_state=42).split(X, y, groups=groups))
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]

    med = ytr.groupby(Xtr["is_business"]).median()
    lr = LinearRegression().fit(Xtr, ytr)
    rf = RandomForestRegressor(n_estimators=60, min_samples_leaf=3, n_jobs=-1, random_state=42).fit(Xtr, ytr)

    preds = {"Baseline (class median)": Xte["is_business"].map(med).values,
             "Linear Regression": lr.predict(Xte),
             "Random Forest": rf.predict(Xte)}
    table = pd.DataFrame([{"Model": k, "R2": round(r2_score(yte, p), 3),
                           "MAE": round(mean_absolute_error(yte, p))} for k, p in preds.items()])

    res = pd.DataFrame({"actual": yte.values, "predicted": preds["Random Forest"],
                        "class": np.where(Xte["is_business"] == 1, "Business", "Economy")})
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
    return rf, list(X.columns), table, res, imp


df = load()
with st.spinner("Training model, this takes a few seconds on first run..."):
    model, cols, table, res, imp = train(df)

# sidebar
page = st.sidebar.radio("Page", ["Overview", "Price Insights", "Model", "Predict", "About"])
st.sidebar.write("---")
cls = st.sidebar.multiselect("Class", ["economy", "business"], default=["economy", "business"])
airlines_all = sorted(df["airline"].unique())
air = st.sidebar.multiselect("Airline", airlines_all, default=airlines_all)
f = df[df["class"].isin(cls) & df["airline"].isin(air)]

if page in ("Overview", "Price Insights") and f.empty:
    st.warning("Nothing selected in the filters.")
    st.stop()


if page == "Overview":
    st.title("Flight Fare Prediction")
    st.write("Domestic flights between 6 Indian metro cities, Goibibo data (Jun to Aug 2023).")

    a, b, c, d = st.columns(4)
    a.metric("Flights", f"{len(f):,}")
    b.metric("Airlines", f["airline"].nunique())
    c.metric("Routes", f["route"].nunique())
    d.metric("Median price", f"Rs {f['price'].median():,.0f}")

    l, r = st.columns(2)
    l.plotly_chart(px.histogram(f, x="price", color="class", log_x=True, nbins=60,
                                title="Price distribution (log scale)", template=T), width="stretch")
    n = f["airline"].value_counts().reset_index()
    r.plotly_chart(px.bar(n, x="airline", y="count", title="Flights per airline", template=T), width="stretch")

    l, r = st.columns(2)
    l.plotly_chart(px.box(f, x="class", y="price", color="class", title="Price by class", template=T), width="stretch")
    s = f["stops"].map({"0": "Non-stop", "1": "1 stop", "2": "2+ stops", "unknown": "Unknown"}).value_counts().reset_index()
    r.plotly_chart(px.pie(s, names="stops", values="count", title="Stops", template=T), width="stretch")


elif page == "Price Insights":
    st.title("Price Insights")

    # same chart, only the grouping column changes
    def bar(col, title, order=None):
        g = f.groupby([col, "class"])["price"].median().reset_index()
        order = {col: order} if order else None
        return px.bar(g, x=col, y="price", color="class", barmode="group",
                      category_orders=order, title=title, template=T)

    l, r = st.columns(2)
    l.plotly_chart(bar("airline", "Median price by airline"), width="stretch")
    r.plotly_chart(bar("stops", "Median price by stops"), width="stretch")
    l, r = st.columns(2)
    l.plotly_chart(bar("time_of_day", "Median price by departure time", TIMES), width="stretch")
    r.plotly_chart(bar("dow", "Median price by day of week", DAYS), width="stretch")

    daily = f.groupby(["flight_date", "class"])["price"].median().reset_index()
    st.plotly_chart(px.line(daily, x="flight_date", y="price", color="class",
                            title="Daily median price", template=T), width="stretch")

    # only routes with enough flights, otherwise the median is noisy
    counts = f["route"].value_counts()
    top = f[f["route"].isin(counts[counts >= 500].index)].groupby("route")["price"].median().nlargest(10)
    l, r = st.columns(2)
    l.plotly_chart(px.bar(top[::-1], orientation="h", title="Top 10 routes by median price", template=T)
                   .update_layout(showlegend=False, yaxis_title=""), width="stretch")
    r.plotly_chart(px.scatter(f.sample(min(5000, len(f)), random_state=1), x="duration_min", y="price",
                              color="class", opacity=0.4, title="Duration vs price", template=T), width="stretch")


elif page == "Model":
    st.title("Model")
    st.write("Tested on a held-out 20%, split so identical rows don't leak into the test set.")
    st.dataframe(table, hide_index=True, width="stretch")

    l, r = st.columns(2)
    s = res.sample(min(4000, len(res)), random_state=1)
    fig = px.scatter(s, x="actual", y="predicted", color="class", opacity=0.4,
                     title="Actual vs predicted (Random Forest)", template=T)
    m = float(s[["actual", "predicted"]].max().max())
    fig.add_shape(type="line", x0=0, y0=0, x1=m, y1=m, line=dict(color="red", dash="dash"))
    l.plotly_chart(fig, width="stretch")
    r.plotly_chart(px.bar(imp.tail(15), orientation="h", title="Top 15 features", template=T)
                   .update_layout(showlegend=False, yaxis_title=""), width="stretch")

    rows = []
    for name, p in res.groupby("class"):
        err = (p["actual"] - p["predicted"]).abs()
        rows.append({"Class": name, "Median price": round(p["actual"].median()), "MAE": round(err.mean()),
                     "Within 10%": f"{(err / p['actual'] <= 0.1).mean() * 100:.0f}%",
                     "Within 20%": f"{(err / p['actual'] <= 0.2).mean() * 100:.0f}%"})
    st.write("Error by class")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


elif page == "Predict":
    st.title("Predict a fare")
    cities = sorted(df["from"].unique())
    stop_names = {"0": "Non-stop", "1": "1 stop", "2": "2+ stops"}

    a, b, c = st.columns(3)
    src = a.selectbox("From", cities, format_func=str.title)
    dst = b.selectbox("To", [x for x in cities if x != src], format_func=str.title)
    kind = c.selectbox("Class", ["economy", "business"], format_func=str.title)

    a, b, c = st.columns(3)
    airline = a.selectbox("Airline", airlines_all)
    stops = b.selectbox("Stops", list(stop_names), index=1, format_func=stop_names.get)
    tod = c.selectbox("Departure", ["Morning", "Afternoon", "Evening", "Night"])

    same = df[(df["from"] == src) & (df["to"] == dst) & (df["stops"] == stops)]
    default = int(same["duration_min"].median()) if len(same) else 150
    a, b = st.columns(2)
    dur = a.number_input("Duration (minutes)", 30, 3000, default, step=5)
    first, last = df["flight_date"].min(), df["flight_date"].max()
    day = b.date_input("Date", first.date(), first.date(), last.date())

    def predict(al):
        row = pd.DataFrame([{"duration_min": dur, "day_index": (pd.Timestamp(day) - first).days,
                             "class": kind, "airline": al, "from": src, "to": dst,
                             "stops": stops, "time_of_day": tod}])
        return model.predict(features(row).reindex(columns=cols, fill_value=0))[0]

    if st.button("Predict", type="primary"):
        st.metric("Predicted price", f"Rs {predict(airline):,.0f}")
        real = df[(df["from"] == src) & (df["to"] == dst) & (df["class"] == kind)]["price"]
        if len(real):
            st.write(f"Actual fares on this route: Rs {real.quantile(.25):,.0f} to Rs {real.quantile(.75):,.0f} "
                     f"(median Rs {real.median():,.0f})")
        cmp = pd.DataFrame({"airline": airlines_all, "price": [predict(a) for a in airlines_all]})
        st.plotly_chart(px.bar(cmp.sort_values("price"), x="airline", y="price",
                               title="Same trip on each airline", template=T), width="stretch")


else:
    st.title("About")
    st.write(f"""
I built this to find out what drives domestic flight prices in India and to predict a fare from the flight details.

The data is {len(df):,} cleaned rows from Goibibo, {df['flight_date'].min().date()} to {df['flight_date'].max().date()}.
I cleaned the price, date, time and duration columns, fixed airline names, removed duplicates, then did EDA,
built features (day index, business flag, one-hot encoding) and compared a baseline, Linear Regression and Random Forest.
In my notebook I also tried XGBoost and LightGBM.

Limits: only about 7 weeks of data, only 6 cities, and there is no "days before departure" column,
which usually matters a lot. So it can't predict seasons or far-future dates.

Made by Ayush Dagar | [GitHub](https://github.com/AyushDagar) | [LinkedIn](https://www.linkedin.com/in/ayush-dagar-9401a7230/)
""")
