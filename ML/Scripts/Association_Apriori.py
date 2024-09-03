# Implementing Apriori algorithm from mlxtend

# conda install mlxtend
# or
# pip install mlxtend

import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

groceries = []
with open(r"data/groceries.csv") as f:
    groceries = f.read()

# splitting the data into separate transactions using separator as "\n"
groceries = groceries.split("\n")

groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))

all_groceries_list = [i for item in groceries_list for i in item]
item_frequencies = Counter(all_groceries_list)

# after sorting
item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1])

# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0].title() for i in item_frequencies]))


# Barplot of top 10

plt.figure(figsize=(12, 6))
plt.bar(height = frequencies[0:10], x = items[0:10], color=['r', 'g', 'b'])

plt.title("Top 10 items", fontweight='bold', fontsize=16)
plt.xlabel("Items")
plt.ylabel("Frequency")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# Creating Data Frame for the transactions data
groceries_series = pd.DataFrame(pd.Series(groceries_list), columns = ["Transactions"])
groceries_series = groceries_series.iloc[:9835, :] # removing the last empty transaction

# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = groceries_series['transactions'].str.join(sep = '*').str.get_dummies(sep = '*')

# Apriori algorithm application
frequent_itemsets = apriori(X, min_support = 0.0075, max_len = 4, use_colnames = True)

# Most Frequent item sets based on support  
frequent_itemsets.sort_values('support', ascending = False, inplace = True)


# Plotting the most Frequent item sets based on support

plt.figure(figsize=(12, 6))
plt.bar(height = frequent_itemsets.support[0:10], x = list(range(0, 10)), color=['r', 'g', 'b'])

plt.title("Top 10 itemsets", fontweight='bold', fontsize=16)
plt.xlabel("Itemsets", fontweight='bold',)
plt.ylabel("Support", fontweight='bold',)

plt.xticks(list(range(0, 10)), frequent_itemsets.itemsets[0:10], rotation=45, ha="right")
plt.tight_layout()
plt.show()


# Association rules

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)

################################# Extra part ###################################
def to_list(i):
    return (sorted(list(i)))

ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift', ascending = False).head(10)
