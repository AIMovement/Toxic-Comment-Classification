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

<<<<<<< HEAD
=======
""" VIZUALIZATION OF DATA """
# Count occurrences
occurences = train.iloc[:, 2:].sum()

# Visualize with histogram
plt.figure(figsize=(8, 4))
ax = sns.barplot(occurences.index, occurences.values, alpha=0.8)
plt.title("# per class")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('Type ', fontsize=12)

# Text labels
rects = ax.patches
labels = occurences.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()

""" CAPS """


""" SPECIAL CHARACTERS  """


""" WRONG SPELLINGS """

>>>>>>> c6072f5f09da606249eb8bfaf65c4c93785ec78b
