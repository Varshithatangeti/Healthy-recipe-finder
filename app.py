from flask import Flask, request, render_template
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

recipes = pd.read_csv('recipes.csv')
tfidf_matrix = vectorizer.transform(recipes['combined_features'])

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recipe')
def recipe_finder():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_category = request.form.get('category', '').strip().lower()
    user_ingredients = request.form.get('ingredients', '').strip().lower()

    if not user_category or not user_ingredients:
        return "Please enter both Recipe Category and Ingredients!"

    user_input = user_category + " " + user_ingredients

    user_vector = vectorizer.transform([user_input])

    cluster = kmeans.predict(user_vector)[0]

    similar_recipes_indices = [i for i in range(len(recipes)) if kmeans.labels_[i] == cluster]

    cosine_sim = cosine_similarity(user_vector, tfidf_matrix[similar_recipes_indices])
    best_match_idx = cosine_sim.argmax()

    recommended_recipe = recipes.iloc[similar_recipes_indices[best_match_idx]]

    return render_template('index.html',
                           recipe_name=recommended_recipe['Name'],
                           recipe_instructions=recommended_recipe['RecipeInstructions'])

if __name__ == '__main__':
    app.run(debug=True)
