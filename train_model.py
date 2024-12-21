import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pickle

recipes = pd.read_csv('recipes.csv')

if 'RecipeCategory' not in recipes.columns or 'RecipeIngredientParts' not in recipes.columns:
    raise ValueError("The dataset must contain 'RecipeCategory' and 'RecipeIngredientParts' columns.")

recipes['combined_features'] = recipes['RecipeCategory'] + " " + recipes['RecipeIngredientParts']

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(recipes['combined_features'])

num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

recipes.to_csv('recipes.csv', index=False)

print("Model and vectorizer have been saved successfully.")
