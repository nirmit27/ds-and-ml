import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist 



# Kmeans on University Data set 
Univ1 = pd.read_excel(r"D:\CODES AND DATASETS\PCA\University_Clustering.xlsx")

Univ1.describe()
Univ = Univ1.drop(["State"], axis = 1)

# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(Univ1.iloc[:, 2:])

# TODO