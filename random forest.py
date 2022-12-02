# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 21:17:31 2022

@author: Uyen Le
"""


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.ensemble import RandomForestClassifier

train = 'label04_T1ce_grade_spliting-11-29-2022.csv'
# comma separated values
df = pd.read_csv(train)
'''
corr_matrix = df.drop(columns=['Grade'])._get_numeric_data().sample(frac=0.25).corr()

# threshold above which is considered too high of a correlation
corr_threshold = 0.8

# Create correlation matrix
corr_matrix = corr_matrix.abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than the correlation threshold
to_drop_corr = [column for column in upper.columns if any(upper[column] >= corr_threshold)]
print(to_drop_corr)

"""Training data"""

train_df = df.drop(columns=to_drop_corr)

'''

# Drop the SalePrice (target of prediction) column and select all numerical columns
num_cols = df.drop(columns=['Grade']).select_dtypes(include='number').columns
print(num_cols)
#train_df[num_cols].head()
# num_cols: a list of column names having numerical data



test_size_ratio = 0.3

X_train, X_test, y_train, y_test = train_test_split(df[num_cols], df.Grade, test_size=0.2, random_state=42)



tree_model = RandomForestClassifier(random_state=101)
tree_model = tree_model.fit(X_train, y_train)

y_predict = tree_model.predict(X_test)
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_predict))

