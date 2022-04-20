import pandas

import IMLearn
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
from IMLearn.metrics import mean_square_error
from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import linear_regression

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    data = pd.read_csv(filename).dropna().drop_duplicates()
    for i in ["id", "long"]:
        data = data.drop(i, axis=1)

    for i in ["price", "sqft_living"
       , "sqft_lot15", "floors"]:
        data = data[data[i] > 0]

    for i in ["bedrooms", "bathrooms"]:
        data = data[data[i] >= 0]
    data['yr_sold'] = pd.to_datetime(data['date']).dt.year
    data = data.drop("date", axis=1)
    data["recently_renovated"] = np.where(data["yr_sold"] - data["yr_renovated"] <= 30, 1, 0)

    data = pd.get_dummies(data, prefix="zipcode_", columns=["zipcode"])

    data = data[data["waterfront"].isin([0, 1])]
    data = data[data["view"].isin(range(5))]
    data = data[data["condition"].isin(range(6))]
    data = data[data["grade"].isin(range(14))]
    data = data.drop("yr_renovated", axis=1)
    data = data.drop("lat", axis=1)


    prices_vector = data["price"]
    data = data.drop("price", axis=1)
    return data, prices_vector


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for f in X:
        corr = np.cov(X[f], y)[1, 0] / (np.std(X[f]) * np.std(y))
        fig = px.scatter(x=X[f], y=y)
        fig.update_layout(title=f"{corr}"
                          , xaxis_title=f"{f} value", yaxis_title="price")
        fig.write_image(f"{output_path}/{f}.png")


if __name__ == '__main__':
    np.random.seed(0)
    y_true = np.array([279000, 432000, 326000, 333000, 437400, 555950])
    y_pred = np.array(
        [199000.37562541, 452589.25533196, 345267.48129011, 345856.57131275, 563867.1347574, 395102.94362135])

    # print(mean_square_error(y_true, y_pred))
    # Question 1 - Load and preprocessing of housing prices dataset

    X, y = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, "../Ex2_figs")

    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = split_train_test(X, y)
    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    p_array = np.arange(10, 101, 1)
    p_loss_values = np.zeros(10)
    std_loss_values = np.zeros(91)
    mean_loss_values = np.zeros(91)
    for p in range(10, 101):
        sum_mse = 0
        for i in range(10):
            train_X, train_Y, test_X, test_Y = split_train_test(train_x, train_y, p / 100)
            X_train_array, y_train_array = train_X.to_numpy(), train_Y.to_numpy()

            linr = LinearRegression()
            linr.fit(X_train_array, y_train_array)
            p_loss_values[i] = linr.loss(test_x.to_numpy(), test_y.to_numpy())
        mean_loss_values[p - 10] = np.mean(p_loss_values)
        std_loss_values[p - 10] = np.std(p_loss_values)

    fig = go.Figure()
    fig.update_layout(xaxis_title="p%", yaxis_title="Mean loss")
    fig.add_scatter(name="mean loss of p%", x=p_array, y=mean_loss_values)
    fig.add_scatter(name="cofidence interval (+2)", x=p_array, y=mean_loss_values + 2 * std_loss_values
                    , fill="tonexty", mode="lines", marker=dict(color="lightgrey"),
                    showlegend=False
                    )
    fig.add_scatter(name="cofidence interval (-2)", x=p_array, y=mean_loss_values - 2 * std_loss_values
                    , fill="tonexty", mode="lines", marker=dict(color="lightgrey"),
                    showlegend=False
                    )
    fig.show()
