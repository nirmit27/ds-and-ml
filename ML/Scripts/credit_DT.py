# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:21:47 2024

@author: mishr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Auto EDA
import sweetviz

from feature_engine.outliers import Winsorizer


df = pd.read_csv(r"data/credit.csv")

df.describe()
df.isna().sum()

y = df['default']
X = df.iloc[:, :-1]

X.drop(columns=['phone'], inplace=True)

# Identifying unique values in categorical features
for column in X.select_dtypes(include='object').columns:
    print(f"{column} : {X[column].unique()}")

# Outliers in numerical features
X.select_dtypes(exclude="object").plot(
    kind='box', subplots=True, figsize=(10, 5), sharey=False)
plt.subplots_adjust(wspace=0.75)
plt.show()

# Winsorization


def winsorize(column: str):
    winsorizer = Winsorizer(capping_method='iqr', fold=1.5,
                            tail='both', variables=[column])
    X[column] = winsorizer.fit_transform(X[[column]])


for column in X.select_dtypes(exclude='object').columns[:-1]:
    winsorize(X[column].name)

# Input columns ['dependents'] have low variation for method 'iqr', so we didn't include it.


# Auto EDA
report = sweetviz.analyze([df, "data"])
report.show_html("Report_0.html")
