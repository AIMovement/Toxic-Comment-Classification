def getfilenames(path = '../../data', ext = '.dat'):
    """
    Get the file names of specified extension.
    :param path: Path to the search directory.
    :param ext: File extension to search for.
    :return filenames: List of filenames.
    """
    import os

    filenames = []
    filectr = 0
    for idx, file in enumerate(os.listdir(path)):
        if file.endswith(ext):
            filenames.append(file)
            print('Found: |  {}  | on Index: |  {}  |'.format(file, filectr))
            filectr += 1

    print('\n')

    return filenames


def printmdfkeys(file):
    """
    Print "keys" from mdf(.dat) file.
    :param file: Name of .dat file.
    """
    import mdfreader

    mdfobject = mdfreader.mdf(file)
    print(mdfobject.keys())


def mdf2df(filedir, file, signals, channel):
    """
    Get data field from specified channel and .dat file.
    :param filedir: Directory of .dat file.
    :param file: Name of .dat file.
    :param signals: Signal name(s).
    :param channel: Signal related channel (time-raster).
    :return: Pandas dataframe.
    """
    import mdfreader
    import numpy as np
    import pandas as pd

    sigdat = []
    for sig in signals:
        mdfsig = mdfreader.mdf(filedir + file, channelList=[sig, channel], convertAfterRead=False)
        tmpdata = mdfsig.getChannelData(sig)
        sigdat.append(tmpdata.astype(np.float32))

    mdfdat = np.stack(sigdat, axis=-1)
    mdfdf = pd.DataFrame(data=mdfdat, columns=signals)

    return mdfdf


def cleanheader(header):
    """
    Build header from txt file
    :param header: string without word separation
    :return: list with header words
    """
    header = header.replace(" ", "")
    header = list(header.split("\t"))
    header = header[:-1]

    return header


def txt2csv(path, save_flag=True):
    """
    Convert txt-file to csv and save with the same name
    :param filedir: Name of .dat file.
    :param file: Signal name(s).
    :param signals: Signal related channel (time-raster).
    :return: Pandas dataframe
    """
    import pandas as pd
    import numpy as np

    with open(path) as f:
        data = f.readlines()

    header = cleanheader(data[3])
    data = data[5:]

    temp_list = []
    for idx, row in enumerate(data):
        columns = row.split()
        col_stack = np.hstack(columns)
        temp_list.append(col_stack)

    df = pd.DataFrame(temp_list, columns=header)

    if save_flag:
        save_to_file = path[:-3] + 'csv'
        df.to_csv(path_or_buf=save_to_file, header=True, index=False)
        print('Saving {}'.format(save_to_file))

    return df


def csv2df(dir):
    """
    Load csv file
    :param dir: Path directory
    :return: Dataframe
    """
    import pandas as pd

    df = pd.read_csv(filepath_or_buffer=dir)

    if 'Time' in df:
        df = df.drop('Time', 1)

    return df


def resample(df, ratio):
    """
    Resamples dataframe df to ratio
    :param df: Dataframe to be resampled
    :param ratio: Ratio to resample with
    :return: Resampled dataframe 
    """
    resample_df = df.iloc[::ratio, :]

    return resample_df


def loadsimdirs(path_tuple, save_flag=False):
    """
    Load simulations from directories
    :param dir_tuple:
    :param save_flag
    :return: DF with simulated data
    """

    df = {}
    if save_flag:
        for dict_key in path_tuple:
            df[dict_key] = txt2csv(path=path_tuple[dict_key])

    else:
        for dict_key in path_tuple:
            load_path = path_tuple[dict_key][:-3] + 'csv'
            df[dict_key] = csv2df(load_path)

    return df


def commoncols(df1, df2):
    """
    Get common cols in 2 dataframes and return dataframes with common cols
    :param df1: Dataframe 1
    :param df2: Dataframe 2
    :return: Dataframes with common cols
    """
    common = []
    for col in df1.columns:
        if col in df2.columns:
            common.append(col)

    df1 = df1[common]
    df2 = df2[common]

    return df1, df2


def reshape2common(ref_df, to_reshape_df):
    """
    Reshapes to_reshape_df to the len of ref_df through down-sampling.
    :param ref_df:
    :param to_reshape_df:
    :return:
    """
    if to_reshape_df.shape[0] != ref_df.shape[0]:
        common_size = False
        ratio = round(to_reshape_df.shape[0] / ref_df.shape[0])
    else:
        common_size = True

    temp_df = to_reshape_df
    while not common_size:
        to_reshape_df = resample(temp_df, ratio)
        rest = to_reshape_df.shape[0] - ref_df.shape[0]

        if rest < 0:
            ratio -= 1

        elif rest == 0:
            common_size = True

        else:
            to_reshape_df = to_reshape_df[:-rest]
            common_size = True

    return to_reshape_df#.as_matrix()


def maxnorm(df):
    """
    Perform normalization on data based on max-value
    :param args: Numpy array to be normalized
    :param type: Type of normalization
    :return: Normalized array
    """
    import numpy as np
    import pandas as pd

    df_max = df.max()

    if isinstance(df_max, pd.DataFrame):
        for idx, row in enumerate(df_max):
            if row == 0:
                df_max[idx] = 0.001 # 0 safeguard
    df /= df_max

    return np.array(df), df_max


def testmaxnorm(df, df_max):
    """
     Perform normalization on TEST data based on TRAIN max-value
     :param args: Numpy array to be normalized
     :param type: Type of normalization
     :return: Normalized array
     """
    import pandas as pd
    if isinstance(df_max, pd.DataFrame):
        for idx, row in enumerate(df_max):
            if row == 0:
                df_max[idx] = 0.001 # 0 safeguard
    df /= df_max

    return df


def norm(df):
    """
    Perform normalization on data between -1 and 1
    :param args: Numpy array to be normalized
    :param type: Type of normalization
    :return: Normalized array
    """
    import numpy as np
    import pandas as pd

    df_max = df.max()
    df_min = df.min()

    if isinstance(df_max, pd.DataFrame):
        for idx, row in enumerate(df_max):
            if row == 0:
                df_max[idx] = 0.001 # 0 safeguard
    df = (df - df_min) / (df_max - df_min)

    return np.array(df), df_max, df_min


def infnanguard(df):
    """
    Removes Inf and NaN values from dataframe
    :param df: Dataframe to be checked
    :return: Inf and NaN free dataframe
    """

    import numpy as np

    df[np.isnan(df)] = 0
    df[np.isinf(df)] = 0

    return np.array(df)


def read_weights(weights_dir):
    """
    Read weights from Neural Network model
    :param weights_dir: Directory to hdf5-file
    :return: Weights
    """
    import h5py
    f = h5py.File(weights_dir)

    return list(f)