import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from IMLearn.utils import split_train_test

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data = pd.read_csv(filename, parse_dates=["Date"]).dropna().drop_duplicates()
    data["DayOfYear"] = data["Date"].dt.day_of_year
    data = data[data["Year"].isin(range(2023))]
    data = data[data["Temp"] > -70]
    data = data[data["Temp"] < 70]

    return data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    data_israel = data[data["Country"] == "Israel"]
    fig1 = px.scatter(data_israel,
                      title="Average daily temperature in israel", x="DayOfYear"
                      , y="Temp",
                      color=data_israel.Year.astype(str))
    fig1.show()

    fig2 = px.bar(data_israel.groupby(["Month"]).Temp.agg(["std"])
                  , title="std of daily temperature")
    fig2.show()

    # Question 3 - Exploring differences between countries

    data_countries = data.groupby(["Country", "Month"], as_index=False).agg(dict(Temp=["mean", "std"]))

    fig3 = px.line(x=data_countries["Month"],
                   y=data_countries[("Temp", "mean")],
                   color=data_countries['Country'], labels=dict(x="Month", y="Mean Temp", color="Country"),
                   error_y=data_countries[("Temp", "std")],
                   title="Temp mean in months at different countries")
    fig3.show()
    # Question 4 - Fitting model for different values of `k`

    k_array = np.array(range(1, 11))
    loss_array = np.zeros(10)
    train_x, train_y, test_x, test_y = split_train_test(data_israel["DayOfYear"], data_israel["Temp"])

    for k in k_array:
        poly = PolynomialFitting(k)
        poly.fit(train_x.to_numpy(), train_y.to_numpy())
        a = round(poly.loss(test_x.to_numpy(), test_y.to_numpy()), 2)
        print("The loss with polynomial fit of", k, "is", a)
        loss_array[k - 1] = a

    fig4 = px.bar(x=k_array, y=loss_array, title="Loss in relation to polynomial fit of k",
                  labels=dict(x="K value", y="Loss"))
    fig4.show()

    # Question 5 - Evaluating fitted model on different countries

    country_array , loss_array = [], []
    poly_israel_5 = PolynomialFitting(5)
    poly_israel_5.fit(data_israel["DayOfYear"], data_israel["Temp"])
    group_country = data.groupby(["Country"])
    for group_name, df_group in group_country:
        country_array.append(group_name)
        loss_array.append(poly_israel_5.loss(df_group["DayOfYear"], df_group["Temp"]))

    fig5 = px.bar(x=country_array, y=loss_array, title="Loss of certain countries in polynomial fit with degree: 5"
                  , labels=dict(x="Country", y="Loss"))
    #fig5.show()
