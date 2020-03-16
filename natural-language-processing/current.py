from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/spambase.data').as_matrix()
np.random.shuffle(data)

X = data[:, :48]
Y = data[:, -1]

print(data)