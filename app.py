from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.express as px
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
#from wordcloud import WordCloud
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer
import re

app = Flask(__name__, static_url_path='/static')

# Load your Netflix dataset (replace 'netflix_data.csv' with your actual dataset)
df = pd.read_csv("netflix_titles.csv")
df.drop_duplicates(inplace=True)

# Handle missing values
df.dropna(subset=['title', 'description', 'listed_in'], inplace=True)

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['description'])

# Calculate Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Simple recommendation logic (replace with your actual recommendation logic)
def get_recommendations(title,top_n=5):
    # Find the index of the movie in the dataset
    idx = df[df['title'] == title].index[0]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the top-n most similar movies
    sim_scores = sim_scores[1:top_n+1]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top-n most similar movies
    return df['title'].iloc[movie_indices].tolist()
# Simple trend analysis (replace with your actual trend analysis)
def get_trend_data():
    genre_counts = df['listed_in'].value_counts().head(10)
    return genre_counts.to_dict()

# Define routes
@app.route('/')
def home():
    # Get initial trend data for the home page
    trend_data = get_trend_data()

    # Render home page with trend analysis
    return render_template('home.html', trend_data=trend_data)

# Route for getting recommendations based on user input
@app.route('/get_recommendations', methods=['POST'])
def get_user_recommendations():
    data = request.get_json()
    movie_title = data['movieTitle']
    recommendations = get_recommendations(movie_title)
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)
