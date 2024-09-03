# pip install lifelines

import pandas as pd
import matplotlib.pyplot as plt

# Loading the the survival un-employment data
survival_unemp = pd.read_csv(r"data/survival_unemployment.csv")
survival_unemp.head()
survival_unemp.describe()

survival_unemp["spell"].describe()

# Spell is referring to time 
T = survival_unemp.spell

# Importing the KaplanMeierFitter model to fit the survival analysis
from lifelines import KaplanMeierFitter

# Initiating the KaplanMeierFitter model
kmf = KaplanMeierFitter()

# Fitting KaplanMeierFitter model on Time and Events for death 
kmf.fit(T, event_observed = survival_unemp.event)

# Time-line estimations plot 
kmf.plot()

# Over Multiple groups 
# For each group, here group is ui
survival_unemp.ui.value_counts()

# Plotting the Kaplan Meier plot for the two groups
plt.figure(figsize=(8, 6))

# Applying KaplanMeierFitter model on Time and Events for the group "1" i.e. FUNDED
kmf.fit(T[survival_unemp.ui==1], survival_unemp.event[survival_unemp.ui==1], label='Funded')
kmf.plot()

# Applying KaplanMeierFitter model on Time and Events for the group "0" i.e. NOT FUNDED
kmf.fit(T[survival_unemp.ui==0], survival_unemp.event[survival_unemp.ui==0], label='Not funded')
kmf.plot()

plt.title("Kaplan Meier Survival Plot", fontweight='bold')
plt.xlabel("Timeline")
plt.ylabel("Probability of unemployment")
plt.show()

"""
We can observe that the people who are NOT recieving the unemployment funds
are MORE likely to get a job over the course of time, while the opposite is
true for the people who are RECIEVING the funds.
"""
