from pylab import subplot
import matplotlib.pyplot as plt


def dist(*args, subflag=True):
    """
    Plot the distribution of data.
    :param *args: One dimensional numpy arrays, containing data to be plotted.
    :param subflag: Flag for plotting the distribution(s) in separte figures
    (subflag = True) or in the same figure (subflag = False).
    """
    from seaborn import distplot

    if not subflag:
        fig, ax1 = plt.subplots()

    for idx, arg in enumerate(args):
        if subflag:
            ax1 = subplot(len(args), 1, idx+1)

        distplot(arg, ax=ax1)

    plt.show()


def unikde(*args, subflag=True):
    """
    Plot the univariate kernel density estimation of data.
    :param args: One dimensional numpy arrays, containing data to be plotted.
    :param subflag: Flag for plotting the univariate kernel density estimation(s) in separate figures
    (subflag = True) or in the same figure (subflag = False).
    """
    from seaborn import kdeplot

    if not subflag:
        fig, ax1 = plt.subplots()

    for idx, arg in enumerate(args):
        if subflag:
            ax1 = subplot(len(args), 1, idx+1)

        kdeplot(arg, ax=ax1)

    plt.show()


def bikde(*args, subflag=False):
    """
    Plot the bivariate kernel density estimation of data if pairwise arguments are inserted.
    :param args: One dimensional numpy arrays, containing data to be plotted.
    :param subflag: Flag for plotting the bivariate kernel density estimation(s) in separate figures
    (subflag = True) or in the same figure (subflag = False).
    """
    from seaborn import kdeplot,jointplot

    if len(args) % 2 == 0:
        if not subflag:
            fig, ax1 = plt.subplots()

            for i, _ in enumerate(args[::2]):
                kdeplot(args[i], args[i+1], ax=ax1)

        #else:
        #    for j, _ in enumerate(args[::2]):
        #        ax1 = subplot(len(args), 1, j + 1)

    else:
        print("Input was not inserted pairwise, returning univariate KDE plot instead.")
        unikde(*args)

    plt.show()


def hex(x, y, df, subflag=False):
    """
    Plot the bivariate kernel density estimation in hex of data if pairwise arguments are inserted with
    distributions on each axis.
    :param x: String for column in df to be plotted
    :param y: String for column in df to be plotted
    :param df: DF
    :param subflag: Flag for plotting the bivariate kernel density estimation(s) in separate figures
    (subflag = True) or in the same figure (subflag = False).
    """
    from seaborn import jointplot
    import sys

    if not subflag:
        jointplot(x, y, df, kind='hex')

    else:
        sys.exit("Input was not inserted pairwise, returning univariate KDE plot instead.")

    plt.show()


def rug(*args, subflag=True):
    """
    Plot the datapoints in an array as sticks on an axis.
    :param args: One dimensional numpy arrays, containing data to be plotted.
    :param subflag: Flag for plotting the rug(s) in separate figures
    (subflag = True) or in the same figure (subflag = False).
    """
    from seaborn import rugplot

    if not subflag:
        fig, ax1 = plt.subplots()

    for idx, arg in enumerate(args):
        if subflag:
            ax1 = subplot(len(args), 1, idx+1)

        rugplot(arg, ax=ax1)

    plt.show()


def timeseries(*args, subflag=True):
    """
    Plot time series data.
    :param args: One dimensional numpy arrays, containing data to be plotted.
    :param subflag: Flag for plotting the timeserie(s) in separate figures
    (subflag = True) or in the same figure (subflag = False).
    """

    if not subflag:
        fig, ax1 = plt.subplots()

    for idx, arg in enumerate(args):
        if subflag:
            ax1 = subplot(len(args), 1, idx+1)

        ax1.plot(arg)

    plt.show()


def pair(*args, legend=''):
    """
    Plot pairwise relationships in a dataset.
    :param args: A dataframe and the columns to be plotted
    """
    from seaborn import pairplot
    import pandas as pd

    columns = []
    for idx, arg in enumerate(args):
        if isinstance(arg, pd.DataFrame):
            df = arg
        else:
            columns.append(arg)

    pairplot(df, x_vars=columns, y_vars=columns, diag_kind="kde")

    plt.show()


def nnloss(*args, subflag=False):
    """
    Plot the loss vs epochs from Keras history object.
    :param mdlhist: Keras history object.
    """
    if not subflag:
        fig, ax1 = plt.subplots()

    for idx, arg in enumerate(args):
        if subflag:
            ax1 = subplot(len(args), 1, idx+1)

        plt.plot(arg.history['loss'])
        plt.plot(arg.history['val_loss'])
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper right')

    plt.show()


def fancyplot(df, title='', subflag=False):
    """
    Plotting predictions vs. true labels very fancy
    :param args: Pandas dataframe, containing predicted and true value.
    :return: Prediction vs. true label plot
    """
    from matplotlib import pyplot as plt
    import seaborn as sns; sns.set()

    if not subflag:
        fig, ax1 = plt.subplots(figsize=(16, 8))

    for idx, regr in enumerate(df.iteritems()):
        sns.set_palette("husl", 3)

        if subflag:
            ax1 = subplot(len(df), 1, idx+1)

        ax1 = plt.plot(regr[1])

    plt.legend(df.columns)
    plt.title(title)
    plt.show()


def correlations(df, title=''):
    """
    Plotting a diagonal correlation matrix
    :param df: Dataframe holding correlation data
    :param title: Optional title name
    :return: Correlation matrix plot
    """
    from string import ascii_letters
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(style="white")

    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()


def variance(arr, window=3):
    """
    Rolling variance of array
    :param arr: Numpy array to calculate rolling variance from
    :param window: Window to use for rolling variance calc.
    :return: Plot
    """
    import pandas as pd

    arr = arr.reshape(len(arr))
    arr_series = pd.Series(arr)
    roll_var = pd.rolling_var(arr_series, window=window)
    timeseries(roll_var)
