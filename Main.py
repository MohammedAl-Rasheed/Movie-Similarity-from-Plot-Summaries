import numpy as np
import pandas as pd
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics.pairwise import cosine_similarity

movies_df = pd.read_csv("Data/movies.csv")
movies_df["plot"] = movies_df["wiki_plot"].astype(str) + "\n" + movies_df["imdb_plot"].astype(str)
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text):
    tokens = [sent for sent in nltk.sent_tokenize(text)]
    filtered_tokens = [token for token in tokens if re.search('[a-zA-Z]', token)]

    stems = [stemmer.stem(t) for t in filtered_tokens]
    
    return stems

tfidf_vectorizer = TfidfVectorizer(max_df=0.95, max_features=200000,
                                 min_df=0, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem,
                                 ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform([x for x in movies_df["plot"]])

km = KMeans(n_clusters=5)

km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

movies_df["cluster"] = clusters

movies_df['cluster'].value_counts() 
similarity_distance = 1 - cosine_similarity(tfidf_matrix)

mergings = linkage(similarity_distance, method='complete')
dendrogram_ = dendrogram(mergings,
               labels=[x for x in movies_df["title"]],
               leaf_rotation=90,
               leaf_font_size=16,
)
fig = plt.gcf()
_ = [lbl.set_color('r') for lbl in plt.gca().get_xmajorticklabels()]
fig.set_size_inches(108, 21)
#print(dendrogram_)
#plt.show()