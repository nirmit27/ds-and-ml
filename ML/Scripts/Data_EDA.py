# Introduction to EDA - Day 2 of TnP

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read data into Python
education = pd.read_csv(r"data\education.csv")

# data\education.csv - this is windows default file path with a '\'
# data\\education.csv - change it to '\\' to make it work in Python
# OR use a raw string i.e. r"data\education.csv"


# Exploratory Data Analysis

"""
Measures of Central Tendency / First moment business decision
>> Mean, Median and Mode
"""

print(f"Mean Work Experience : {education['workex'].mean():.2f}")
print(f"Median Work Experience : {education['workex'].mean():.2f}")
print(f"Mode Work Experience : {education['workex'].mean():.2f}")


"""
Measures of Dispersion / Second moment business decision
>> Variance, Standard Deviation and Range
"""

print(f"Variance of Work Experience : {education.workex.var():.2f}")
print(f"Standard Deviation of Work Experience : {education.workex.std():.2f}")
print(f"Range of Work Experience : {education.workex.max() - education.workex.min()}")


"""
Third moment business decision
>> Skewness - Positvely/ Negatively/ Normal
"""

print(f"Skewness of Work Experience : {education.workex.skew():.2f}")
print(f"Skewness of GMAT scores : {education.gmat.skew():.2f}")


"""
Fourth moment business decision
>> Kurtosis
"""

print(f"Kurtosis of Work experience : {education.workex.kurt():.2f}")


# Data Visualization

# Barplot
plt.bar(height = education.gmat, x = np.arange(1, 774, 1)) # initializing the parameter

# Histogram
plt.hist(education.gmat) # histogram
plt.hist(education.gmat, bins = [600, 680, 710, 740, 780], color = 'green', edgecolor="red") 
plt.hist(education.workex)
plt.hist(education.workex, color='red', edgecolor = "black", bins = 6)

help(plt.hist)

# Histogram using Seaborn
import seaborn as sns
sns.distplot(education.gmat) # Deprecated histogram function from seaborn

sns.displot(education.gmat) # Histogram from seaborn


# Boxplot
plt.figure()
plt.boxplot(education.gmat) # boxplot

help(plt.boxplot)


# Density Plot
sns.kdeplot(education.gmat) # Density plot
sns.kdeplot(education.gmat , bw = 0.5 , fill = True)


# Descriptive Statistics
# describe function will return descriptive statistics including the central tendency, dispersion and shape of a dataset's distribution

education.describe()


# Bivariate visualization
# Scatter plot
import pandas as pd
import matplotlib.pyplot as plt

cars = pd.read_csv("C:/Data/Cars.csv")

cars.info()

plt.scatter(x = cars['HP'], y = cars['MPG']) 

plt.scatter(x = cars['HP'], y = cars['SP'], color = 'green') 

