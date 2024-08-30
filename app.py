import pickle
import streamlit as st
import numpy as np
import pandas as pd

# Header for the app
st.header('Book Recommender System Using Machine Learning')

# Loading the models and data
model = pd.read_pickle(open('artifacts/model.pkl', 'rb'))
book_names = pd.read_pickle(open('artifacts/book_names.pkl', 'rb'))
final_rating = pd.read_pickle(open('artifacts/final_rating_v2.pkl', 'rb'))
book_pivot = pd.read_pickle(open('artifacts/book_pivot.pkl', 'rb'))

# Fetch poster URLs, ratings, and author names for the suggested books
def fetch_poster_and_details(suggestion):
    book_name = []
    ids_index = []
    poster_url = []
    ratings = []
    authors = []

    # Get the book names based on suggestions
    for book_id in suggestion:
        book_name.append(book_pivot.index[book_id])

    # Find the corresponding indices for the book names in the final_rating DataFrame
    for name in book_name[0]:
        ids = np.where(final_rating['title'] == name)[0][0]
        ids_index.append(ids)

    # Fetch the details: poster URL, rating, and author name
    for idx in ids_index:
        url = final_rating.iloc[idx]['image_url']
        poster_url.append(url)
        ratings.append(final_rating.iloc[idx]['rating'])  # Assuming there's a 'rating' column
        authors.append(final_rating.iloc[idx]['author'])  # Assuming there's an 'author' column

    return poster_url, ratings, authors

# Function to recommend books based on the selected book
def recommend_book(book_name):
    books_list = []
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

    poster_url, ratings, authors = fetch_poster_and_details(suggestion)
    
    # Retrieve book names from suggestions
    for i in range(len(suggestion)):
        books = book_pivot.index[suggestion[i]]
        for j in books:
            books_list.append(j)

    return books_list, poster_url, ratings, authors

# Streamlit UI for selecting and displaying recommendations
selected_books = st.selectbox(
    "Type or select a book from the dropdown",
    book_names
)

# Display recommendations when the button is clicked
if st.button('Show Recommendation'):
    recommended_books, poster_url, ratings, authors = recommend_book(selected_books)
    cols = st.columns(5)
    
    # Loop through the columns and display book details
    for i, col in enumerate(cols):
        if i < len(recommended_books):
            with col:
                st.text(f"Title: {recommended_books[i]}")
                st.text(f"Author: {authors[i]}")
                st.text(f"Rating: {ratings[i]}")
                st.image(poster_url[i])

