from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"
import plotly.express as px

TRUE_VAR = 1
TRUE_EXP = 10

def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(10, 1, 1000)
    univar = UnivariateGaussian()
    univar.fit(samples)
    print()
    print("(", univar.mu_, ",", univar.var_, ")\n")

    def estimate_expectancy(size):
        return UnivariateGaussian().fit(samples[0:size]).mu_

    # Question 2 - Empirically showing sample mean is consistent
    sizes = np.arange(10, 1010, 10)
    vector = np.abs(np.vectorize(estimate_expectancy)(sizes) - TRUE_EXP)
    fig_diff = px.scatter(x=sizes, y=vector)
    fig_diff.update_layout(title="The effect of the amount of samples on the expectation accuracy",
                           xaxis_title="Samples amount", yaxis_title="Distant from real expectancy")
    fig_diff.show()


    # Question 3 - Plotting Empirical PDF of fitted model
    pdf = univar.pdf(samples)
    fig_pdf = px.scatter(x=samples, y=pdf)
    fig_pdf.update_layout(title="PDF's", xaxis_title="sample", yaxis_title="pdf")
    fig_pdf.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mean = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    samples_multi = np.random.multivariate_normal(mean, cov, 1000)
    multivar = MultivariateGaussian()
    multivar.fit(samples_multi)
    print(multivar.mu_, "\n")
    print(multivar.cov_)

    # Question 5 - Likelihood evaluation
    heat_data = np.zeros(shape=(200, 200))
    f_arr = np.linspace(-10, 10, 200)
    i = 0
    for f1 in f_arr:
        j = 0
        for f3 in f_arr:
            new_mu = np.array([f1, 0, f3, 0])
            heat_data[i][j] = MultivariateGaussian.log_likelihood(new_mu, cov, samples_multi)
            j += 1
        i += 1

    fig = go.Figure(go.Heatmap(x=f_arr, y=f_arr, z=heat_data))
    fig.update_layout(title="The log likelihood of different values of expectancy", xaxis_title="f3", yaxis_title="f1")
    fig.show()

    # Question 6 - Maximum likelihood
    max_idx = np.unravel_index(np.argmax(heat_data, axis=None), heat_data.shape)
    print("The maximum log likliehood is: [", round(f_arr[max_idx[0]], 3), "0,", round(f_arr[max_idx[1]], 3), ",0]")

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
