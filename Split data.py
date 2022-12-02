# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 15:34:25 2022

@author: Uyen Le
"""



import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# models
## Scikit_Learn
from sklearn.linear_model import LinearRegression, Lasso, Ridge # Linear Regression with regularization
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

## Statsmodels
import statsmodels.api as sm # R_squared, p_value
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor # multicollinearity

# feature selection
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_regression, mutual_info_regression

# metrics
from sklearn.metrics import mean_squared_error, r2_score # R_squared ~ coefficient of determination

# seaborn for visualization
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')

# statistics
from scipy import stats
from scipy.stats import norm, skew #for some statistics

# # plotting
import matplotlib
import matplotlib.pyplot as plt
font = {'size' : 10}
matplotlib.rc('font', **font)

"""# Linear Regression: House Price Prediction

## Dự đoán giá nhà với hồi quy tuyến tính.

## 1. Data Loading & EDA

Load the dataset and describe it with summary statistics.\
Tải tập dữ liệu và thống kê mô tả.
"""

train = 'label04_T1ce_grade_processing-11-29-2022.csv'
# comma separated values
df = pd.read_csv(train)
print(df.shape)
df.describe() # summary statistics & descriptive statistics

"""Print the first 5 rows of the data frame.\
In ra 5 dòng đầu tiên của dataframe.
"""

df.head() # df : a DataFrame object
# .head(): method defined for class DataFrame

"""Print all column names.
In ra tên của các cột.
"""

df.columns # columns: 1 attribute of DataFrame object

"""The last column: `SalePrice` will be the target of today prediction! 😊\
Bài toán của chúng ta sẽ là dự đoán giá nhà lưu ở cột `SalePrice`.

Drop the `Id` column since it is not necessary.\
Bỏ cột `Id` vì không cần thiết.
"""

#df = df.drop(columns=['Id'])

#df.columns

df[['diagnostics_Mask_original_BoundingBox1','diagnostics_Mask_original_BoundingBox2','diagnostics_Mask_original_BoundingBox3','diagnostics_Mask_original_BoundingBox4','diagnostics_Mask_original_BoundingBox5','diagnostics_Mask_original_BoundingBox6']] = df.diagnostics_Mask_original_BoundingBox.str.split(", ", expand=True)
df[['wavelet_LLH_diagnostics_Mask_original_BoundingBox1','wavelet_LLH_diagnostics_Mask_original_BoundingBox2','wavelet_LLH_diagnostics_Mask_original_BoundingBox3','wavelet_LLH_diagnostics_Mask_original_BoundingBox4','wavelet_LLH_diagnostics_Mask_original_BoundingBox5','wavelet_LLH_diagnostics_Mask_original_BoundingBox6']] = df.wavelet_LLH_diagnostics_Mask_original_BoundingBox.str.split(", ", expand=True)
df[['wavelet_LHL_diagnostics_Mask_original_BoundingBox1','wavelet_LHL_diagnostics_Mask_original_BoundingBox2','wavelet_LHL_diagnostics_Mask_original_BoundingBox3','wavelet_LHL_diagnostics_Mask_original_BoundingBox4','wavelet_LHL_diagnostics_Mask_original_BoundingBox5','wavelet_LHL_diagnostics_Mask_original_BoundingBox6']] = df.wavelet_LHL_diagnostics_Mask_original_BoundingBox.str.split(", ", expand=True)
df[['wavelet_LHH_diagnostics_Mask_original_BoundingBox1','wavelet_LHH_diagnostics_Mask_original_BoundingBox2','wavelet_LHH_diagnostics_Mask_original_BoundingBox3','wavelet_LHH_diagnostics_Mask_original_BoundingBox4','wavelet_LHH_diagnostics_Mask_original_BoundingBox5','wavelet_LHH_diagnostics_Mask_original_BoundingBox6']] = df.wavelet_LHH_diagnostics_Mask_original_BoundingBox.str.split(", ", expand=True)
df[['wavelet_HLL_diagnostics_Mask_original_BoundingBox1','wavelet_HLL_diagnostics_Mask_original_BoundingBox2','wavelet_HLL_diagnostics_Mask_original_BoundingBox3','wavelet_HLL_diagnostics_Mask_original_BoundingBox4','wavelet_HLL_diagnostics_Mask_original_BoundingBox5','wavelet_HLL_diagnostics_Mask_original_BoundingBox6']] = df.wavelet_HLL_diagnostics_Mask_original_BoundingBox.str.split(", ", expand=True)
df[['wavelet_HLH_diagnostics_Mask_original_BoundingBox1','wavelet_HLH_diagnostics_Mask_original_BoundingBox2','wavelet_HLH_diagnostics_Mask_original_BoundingBox3','wavelet_HLH_diagnostics_Mask_original_BoundingBox4','wavelet_HLH_diagnostics_Mask_original_BoundingBox5','wavelet_HLH_diagnostics_Mask_original_BoundingBox6']] = df.wavelet_HLH_diagnostics_Mask_original_BoundingBox.str.split(", ", expand=True)
df[['wavelet_HHL_diagnostics_Mask_original_BoundingBox1','wavelet_HHL_diagnostics_Mask_original_BoundingBox2','wavelet_HHL_diagnostics_Mask_original_BoundingBox3','wavelet_HHL_diagnostics_Mask_original_BoundingBox4','wavelet_HHL_diagnostics_Mask_original_BoundingBox5','wavelet_HHL_diagnostics_Mask_original_BoundingBox6']] = df.wavelet_HHL_diagnostics_Mask_original_BoundingBox.str.split(", ", expand=True)
df[['wavelet_HHH_diagnostics_Mask_original_BoundingBox1','wavelet_HHH_diagnostics_Mask_original_BoundingBox2','wavelet_HHH_diagnostics_Mask_original_BoundingBox3','wavelet_HHH_diagnostics_Mask_original_BoundingBox4','wavelet_HHH_diagnostics_Mask_original_BoundingBox5','wavelet_HHH_diagnostics_Mask_original_BoundingBox6']] = df.wavelet_HHH_diagnostics_Mask_original_BoundingBox.str.split(", ", expand=True)
df[['wavelet_LLL_diagnostics_Mask_original_BoundingBox1','wavelet_LLL_diagnostics_Mask_original_BoundingBox2','wavelet_LLL_diagnostics_Mask_original_BoundingBox3','wavelet_LLL_diagnostics_Mask_original_BoundingBox4','wavelet_LLL_diagnostics_Mask_original_BoundingBox5','wavelet_LLL_diagnostics_Mask_original_BoundingBox6']] = df.wavelet_LLL_diagnostics_Mask_original_BoundingBox.str.split(", ", expand=True)

df[['diagnostics_Mask_original_CenterOfMassIndex1','diagnostics_Mask_original_CenterOfMassIndex2','diagnostics_Mask_original_CenterOfMassIndex3']] = df.diagnostics_Mask_original_CenterOfMassIndex.str.split(", ", expand=True)
df[['wavelet_LLH_diagnostics_Mask_original_CenterOfMassIndex1','wavelet_LLH_diagnostics_Mask_original_CenterOfMassIndex2','wavelet_LLH_diagnostics_Mask_original_CenterOfMassIndex3']] = df.wavelet_LLH_diagnostics_Mask_original_CenterOfMassIndex.str.split(", ", expand=True)
df[['wavelet_LHL_diagnostics_Mask_original_CenterOfMassIndex1','wavelet_LHL_diagnostics_Mask_original_CenterOfMassIndex2','wavelet_LHL_diagnostics_Mask_original_CenterOfMassIndex3']] = df.wavelet_LHL_diagnostics_Mask_original_CenterOfMassIndex.str.split(", ", expand=True)
df[['wavelet_LHH_diagnostics_Mask_original_CenterOfMassIndex1','wavelet_LHH_diagnostics_Mask_original_CenterOfMassIndex2','wavelet_LHH_diagnostics_Mask_original_CenterOfMassIndex3']] = df.wavelet_LHH_diagnostics_Mask_original_CenterOfMassIndex.str.split(", ", expand=True)
df[['wavelet_HLL_diagnostics_Mask_original_CenterOfMassIndex1','wavelet_HLL_diagnostics_Mask_original_CenterOfMassIndex2','wavelet_HLL_diagnostics_Mask_original_CenterOfMassIndex3']] = df.wavelet_HLL_diagnostics_Mask_original_CenterOfMassIndex.str.split(", ", expand=True)
df[['wavelet_HLH_diagnostics_Mask_original_CenterOfMassIndex1','wavelet_HLH_diagnostics_Mask_original_CenterOfMassIndex2','wavelet_HLH_diagnostics_Mask_original_CenterOfMassIndex3']] = df.wavelet_HLH_diagnostics_Mask_original_CenterOfMassIndex.str.split(", ", expand=True)
df[['wavelet_HHL_diagnostics_Mask_original_CenterOfMassIndex1','wavelet_HHL_diagnostics_Mask_original_CenterOfMassIndex2','wavelet_HHL_diagnostics_Mask_original_CenterOfMassIndex3']] = df.wavelet_HHL_diagnostics_Mask_original_CenterOfMassIndex.str.split(", ", expand=True)
df[['wavelet_HHH_diagnostics_Mask_original_CenterOfMassIndex1','wavelet_HHH_diagnostics_Mask_original_CenterOfMassIndex2','wavelet_HHH_diagnostics_Mask_original_CenterOfMassIndex3']] = df.wavelet_HHH_diagnostics_Mask_original_CenterOfMassIndex.str.split(", ", expand=True)
df[['wavelet_LLL_diagnostics_Mask_original_CenterOfMassIndex1','wavelet_LLL_diagnostics_Mask_original_CenterOfMassIndex2','wavelet_LLL_diagnostics_Mask_original_CenterOfMassIndex3']] = df.wavelet_LLL_diagnostics_Mask_original_CenterOfMassIndex.str.split(", ", expand=True)

df[['diagnostics_Mask_original_CenterOfMass1','diagnostics_Mask_original_CenterOfMass2','diagnostics_Mask_original_CenterOfMass3']] = df.diagnostics_Mask_original_CenterOfMass.str.split(", ", expand=True)
df[['wavelet_LLH_diagnostics_Mask_original_CenterOfMass1','wavelet_LLH_diagnostics_Mask_original_CenterOfMass2','wavelet_LLH_diagnostics_Mask_original_CenterOfMass3']] = df.wavelet_LLH_diagnostics_Mask_original_CenterOfMass.str.split(", ", expand=True)
df[['wavelet_LHL_diagnostics_Mask_original_CenterOfMass1','wavelet_LHL_diagnostics_Mask_original_CenterOfMass2','wavelet_LHL_diagnostics_Mask_original_CenterOfMass3']] = df.wavelet_LHL_diagnostics_Mask_original_CenterOfMass.str.split(", ", expand=True)
df[['wavelet_LHH_diagnostics_Mask_original_CenterOfMass1','wavelet_LHH_diagnostics_Mask_original_CenterOfMass2','wavelet_LHH_diagnostics_Mask_original_CenterOfMass3']] = df.wavelet_LHH_diagnostics_Mask_original_CenterOfMass.str.split(", ", expand=True)
df[['wavelet_HLL_diagnostics_Mask_original_CenterOfMass1','wavelet_HLL_diagnostics_Mask_original_CenterOfMass2','wavelet_HLL_diagnostics_Mask_original_CenterOfMass3']] = df.wavelet_HLL_diagnostics_Mask_original_CenterOfMass.str.split(", ", expand=True)
df[['wavelet_HLH_diagnostics_Mask_original_CenterOfMass1','wavelet_HLH_diagnostics_Mask_original_CenterOfMass2','wavelet_HLH_diagnostics_Mask_original_CenterOfMass3']] = df.wavelet_HLH_diagnostics_Mask_original_CenterOfMass.str.split(", ", expand=True)
df[['wavelet_HHL_diagnostics_Mask_original_CenterOfMass1','wavelet_HHL_diagnostics_Mask_original_CenterOfMass2','wavelet_HHL_diagnostics_Mask_original_CenterOfMass3']] = df.wavelet_HHL_diagnostics_Mask_original_CenterOfMass.str.split(", ", expand=True)
df[['wavelet_HHH_diagnostics_Mask_original_CenterOfMass1','wavelet_HHH_diagnostics_Mask_original_CenterOfMass2','wavelet_HHH_diagnostics_Mask_original_CenterOfMass3']] = df.wavelet_HHH_diagnostics_Mask_original_CenterOfMass.str.split(", ", expand=True)
df[['wavelet_LLL_diagnostics_Mask_original_CenterOfMass1','wavelet_LLL_diagnostics_Mask_original_CenterOfMass2','wavelet_LLL_diagnostics_Mask_original_CenterOfMass3']] = df.wavelet_LLL_diagnostics_Mask_original_CenterOfMass.str.split(", ", expand=True)


df = df.drop(columns=['diagnostics_Mask_original_BoundingBox'])
df = df.drop(columns=['wavelet_LLH_diagnostics_Mask_original_BoundingBox'])
df = df.drop(columns=['wavelet_LHL_diagnostics_Mask_original_BoundingBox'])
df = df.drop(columns=['wavelet_LHH_diagnostics_Mask_original_BoundingBox'])
df = df.drop(columns=['wavelet_HLL_diagnostics_Mask_original_BoundingBox'])
df = df.drop(columns=['wavelet_HLH_diagnostics_Mask_original_BoundingBox'])
df = df.drop(columns=['wavelet_HHL_diagnostics_Mask_original_BoundingBox'])
df = df.drop(columns=['wavelet_LLL_diagnostics_Mask_original_BoundingBox'])
df = df.drop(columns=['wavelet_HHH_diagnostics_Mask_original_BoundingBox'])

df = df.drop(columns=['diagnostics_Mask_original_CenterOfMass'])
df = df.drop(columns=['wavelet_LLH_diagnostics_Mask_original_CenterOfMass'])
df = df.drop(columns=['wavelet_LHL_diagnostics_Mask_original_CenterOfMass'])
df = df.drop(columns=['wavelet_LHH_diagnostics_Mask_original_CenterOfMass'])
df = df.drop(columns=['wavelet_HLL_diagnostics_Mask_original_CenterOfMass'])
df = df.drop(columns=['wavelet_HLH_diagnostics_Mask_original_CenterOfMass'])
df = df.drop(columns=['wavelet_HHL_diagnostics_Mask_original_CenterOfMass'])
df = df.drop(columns=['wavelet_LLL_diagnostics_Mask_original_CenterOfMass'])
df = df.drop(columns=['wavelet_HHH_diagnostics_Mask_original_CenterOfMass'])

df = df.drop(columns=['diagnostics_Mask_original_CenterOfMassIndex'])
df = df.drop(columns=['wavelet_LLH_diagnostics_Mask_original_CenterOfMassIndex'])
df = df.drop(columns=['wavelet_LHL_diagnostics_Mask_original_CenterOfMassIndex'])
df = df.drop(columns=['wavelet_LHH_diagnostics_Mask_original_CenterOfMassIndex'])
df = df.drop(columns=['wavelet_HLL_diagnostics_Mask_original_CenterOfMassIndex'])
df = df.drop(columns=['wavelet_HLH_diagnostics_Mask_original_CenterOfMassIndex'])
df = df.drop(columns=['wavelet_HHL_diagnostics_Mask_original_CenterOfMassIndex'])
df = df.drop(columns=['wavelet_LLL_diagnostics_Mask_original_CenterOfMassIndex'])
df = df.drop(columns=['wavelet_HHH_diagnostics_Mask_original_CenterOfMassIndex'])

df.to_csv('label04_T1ce_grade_spliting-11-29-2022.csv', index=False)
