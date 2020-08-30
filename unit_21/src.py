# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def plot_acfs(base_series, periods=None, lags=None):
    num_optional_args = (periods is None) + (lags is None)
    if num_optional_args == 2:
        raise ValueError('`diffs` and `lags` can\'t be pluged in together')
    elif num_optional_args == 0:
        raise ValueError('require `diffs` or `lags`')

    if lags is None:
        series = [base_series if p == 0 else base_series.diff(p).dropna()
                  for p in periods]
        lags = [None for _ in periods]
    elif periods is None:
        series = [base_series for _ in lags]
        periods = [None for _ in lags]

    n_cols = len(series)
    fig, axes = plt.subplots(2, n_cols, figsize=(6*n_cols, 8))

    for i, (s, l, p) in enumerate(zip(series, lags, periods)):
        # Plot ACF
        acf_ax = axes[0, i]
        _ = plot_acf(s, lags=l, ax=acf_ax)
        acf_ax.set_title(f'Autocorrelation \n(lag={l}, diff={p})')
        acf_ax.set_xticklabels('')

        # Plot PACF
        pacf_ax = axes[1, i]
        _ = plot_pacf(s, lags=l, ax=pacf_ax)
        pacf_ax.set_title(f'Partial Autocorrelation\n(lag={l}, diff={p})')

    plt.show()


def select_best_model(series, model_cls, p, d, q,
                      P=None, D=None, Q=None, m=None, **kwargs):
    '''Select the best hyperparameters for ARIMA or SARIMAX for univariate
    time series data, based on step-forward MSE score.
    Return 5 best sets of parameters with the least MSE scores.

    Parameters
    ----------
    series : pandas.Series
        Dependent variable, time series data
    model_cls : subclass of tsbase.TimeSeriesModel
        Model class from stats models to find the best hyperparams for.
    p, d, q : lists of int
        list of variables to test (order).
    P, D, Q : lists of int, optional
        list of variables to test (seasonal order).
    s : int, optional
        seasonal periodicity.
    kwargs :
        Arguments to provide when initiate `model_cls`

    Returns
    -------
    list :
        List of 5 sets of MSE and parameters with lowest MSE scores.
    '''
    model_name = model_cls.__name__
    scores = []
    i = 0
    for order, s_order in gen_orders(p, d, q, P, D, Q, m):
        i += 1
        kwargs['order'] = order

        if s_order is None:
            prefix = f'{model_name}{order}:'
        else:
            prefix = f'{model_name}({order}, {s_order[:3]}{s_order[3]}):'
            kwargs['seasonal_order'] = s_order

        mse = calc_mse(series, model_cls, **kwargs)

        if mse is None:
            print(f'{i:0>3n}) {prefix} Rejected        \r', end='')
        else:
            print(f'{i:0>3n}) {prefix} MSE={mse:.4e}\r', end='')
            scores.append((mse, order, s_order))
    scores.sort(key=lambda x: x[0])

    return scores[:5]


def gen_orders(p, d, q, P=None, D=None, Q=None, s=None):
    '''Yield each combinations of given orders.

    Parameters
    ----------
    p, d, q : lists of int
        list of variables to test (order).
    P, D, Q : lists of int, optional
        list of variables to test (seasonal order).
    s : int, optional
        seasonal periodicity.

    Raises
    ------
    ValueError
        When any of P, D, Q, s is provided but not all.

    Yields
    ------
    Tuple of lists
        Contains a combination of two lists: orders and seasonal_orders.
    '''

    # Confirm that all or none of `P`, `D`, `Q`, and `s` are correctly provided
    provided_PDQM = ((P is not None)
                     + (D is not None)
                     + (Q is not None)
                     + (s is not None))
    if provided_PDQM not in (0, 4):
        raise ValueError(
            'All of `P`, `D`, `Q`, and `m`, or none of them has to be provided'
        )

    # generate combinations of orders (p, d, q)
    orders = np.array(np.meshgrid(p, d, q)).T.reshape(-1, 3)

    # If seasonal orders are not provided, yield a tuple of orders and None
    if provided_PDQM == 0:
        for o in orders:
            yield (o, None)

    # If seasonal orders are provided, yield a tuple of both
    else:
        # generate combinations of orders (P, D, Q, s)
        seasonal_orders = np.array(np.meshgrid(P, D, Q, s)).T.reshape(-1, 4)
        for o in orders:
            for so in seasonal_orders:
                yield (o, so)


def calc_mse(series, model_cls, test_size=0.2, true_series=None, **kwargs):
    '''Calculate MSE on the step-forward predictions. Return None when an
    exception raised during model fitting, otherwise MSE.

    Parameters
    ----------
    data : pandas.Series
        Univariate time series dependent variable
    model_cls : Subclass of tsbase.TimeSeriesModel
        Model class from stats models to fit and calculate MSE for.
    test_size : float, optional
        Size of the test set, default 0.2
    true_series : pandas.Series, optional
        Series of true y.
    kwargs :
        Arguments to provide when initiate `model_cls`

    Returns
    -------
    float or None:
        MSE when successful, None when an exception raises during model fit.
    '''
    split = int(len(series) * test_size)
    train = series[:-split]
    test = series[-split:] if true_series is None else true_series[-split:]

    try:
        preds = []  # to store predicted values.
        for i in range(split):  # perform Step-forward optimization.
            model = model_cls(train, **kwargs)
            res = model.fit()
            forecast = res.forecast()
            if isinstance(forecast, tuple):
                forecast = forecast[0]
            preds.extend(forecast)  # append step-forward forecast

            train = train.append(test[[i]])  # Extend train for the next round

        return mean_squared_error(test, preds)

    except Exception:
        return None
