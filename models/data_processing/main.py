import sys
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
#import markovify as mk

path = '../../input/'

TRAIN_DATA_FILE=f'{path}train.csv'
TEST_DATA_FILE=f'{path}test.csv'

print('Load data')
train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)

print('Train shape {}'.format(train.shape))
print('Test shape {}'.format(test.shape))

