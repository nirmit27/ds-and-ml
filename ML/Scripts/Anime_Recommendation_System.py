# import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer


# Importing the dataset
movies = pd.read_csv(r"data/anime.csv", encoding='utf8')

movies.shape, movies.columns, movies.genre

# replacing the NaN values in overview column with empty string
movies["genre"].isnull().sum()
movies["genre"] = movies["genre"].fillna(" ")

# Creating a Tfidf Vectorizer to remove all stop words
# taking stop words from tfid vectorizer
tfidf = TfidfVectorizer(stop_words="english")

# Preparing the Tfidf matrix by fitting and transforming
# Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix = tfidf.fit_transform(movies.genre)
tfidf_matrix.shape  # 12294, 46

# term frequencey - inverse document frequncy is a numerical statistic that is intended to reflect how important
# a word is to document in a collecion or corpus

# with the above matrix we need to find the similarity score
# There are several metrics for this such as the euclidean,
# the Pearson and the cosine similarity scores

# For now we will be using cosine similarity matrix
# A numeric quantity to represent the similarity between 2 movies
# Cosine similarity - metric is independent of magnitude and easy to calculate

# cosine(x,y)= (x.y⊺)/(||x||.||y||)

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

# creating a mapping of movies name to index number
movies_index = pd.Series(movies.index, index=movies['name']).drop_duplicates()

movies_id = movies_index["Assassins (1995)"]
movies_id


def get_recommendations(Name, topN):
    # topN = 10
    # Getting the movie index using its title
    movies_id = movies_index[Name]

    # Getting the pair wise similarity score for all the movies's with that
    # movies
    cosine_scores = list(enumerate(cosine_sim_matrix[movies_id]))

    # Sorting the cosine_similarity scores based on scores
    cosine_scores = sorted(cosine_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of top N most similar movies
    cosine_scores_N = cosine_scores[0: topN+1]

    # Getting the movie index
    movies_idx = [i[0] for i in cosine_scores_N]
    movies_scores = [i[1] for i in cosine_scores_N]

    # Similar movies and scores
    movies_similar_show = pd.DataFrame(columns=["name", "Score"])
    movies_similar_show["name"] = movies.loc[movies_idx, "name"]
    movies_similar_show["Score"] = movies_scores
    movies_similar_show.reset_index(inplace=True)
    # movies_similar_show.drop(["index"], axis=1, inplace=True)
    print(movies_similar_show)
    # return (movies_similar_show)


# Enter your movies and number of movies's to be recommended
get_recommendations("Bad Boys (1995)", topN=10)
movies_index["Bad Boys (1995)"]
