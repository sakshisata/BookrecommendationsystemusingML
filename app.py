import pickle
import streamlit as st
import numpy as np
import pandas as pd

# Set page config
st.set_page_config(page_title='Book Recommender System', page_icon='📚', layout='wide')

# Custom CSS for styling
st.markdown("""
    <style>
    .card {
        background-color: #f8f9fa;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
        transition: transform 0.2s;
    }
    .card:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .title {
        font-size: 18px;
        font-weight: bold;
        color: #333;
    }
    .author, .rating {
        font-size: 14px;
        color: #555;
    }
    </style>
""", unsafe_allow_html=True)

# Header for the app
st.title('📚 Book Recommender System Using Machine Learning')

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

# Function to search books by author and fetch their details
def search_books_by_author(author_name):
    author_books = final_rating[final_rating['author'].str.contains(author_name, case=False, na=False)]
    titles = author_books['title'].unique()
    ratings = author_books['rating'].tolist()
    
    return titles, ratings

# Function to search books by rating
def search_books_by_rating(min_rating, max_rating):
    rating_books = final_rating[(final_rating['rating'] >= min_rating) & (final_rating['rating'] <= max_rating)]
    titles = rating_books['title'].unique()
    ratings = rating_books['rating'].tolist()
    
    return titles, ratings

# Streamlit UI for selecting search option and displaying recommendations
search_option = st.radio("🔍 Search by:", ("Book Title", "Author", "Rating"))

# Handling search based on selected option
if search_option == "Book Title":
    selected_books = st.selectbox(
        "📖 Type or select a book from the dropdown",
        book_names
    )

    # Display recommendations when the button is clicked
    if st.button('🔍 Show Recommendation'):
        recommended_books, poster_url, ratings, authors = recommend_book(selected_books)
        cols = st.columns(5)
        
        # Loop through the columns and display book details
        for i, col in enumerate(cols):
            if i < len(recommended_books):
                with col:
                    # Creating a card-like structure for each book
                    st.markdown(
                        f"""
                        <div class="card">
                            <img src="{poster_url[i]}" width="100%" height="200px" style="border-radius: 8px;"/>
                            <p class="title">{recommended_books[i]}</p>
                            <p class="author">Author: {authors[i]}</p>
                            <p class="rating">Rating: ⭐ {ratings[i]}</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

elif search_option == "Author":
    author_name = st.text_input("✍️ Enter author name:")
    
    if author_name:
        titles, ratings = search_books_by_author(author_name)
        
        if len(titles) > 0:
            cols = st.columns(5)
            for i, col in enumerate(cols):
                if i < len(titles):
                    with col:
                        # Displaying book details without images, focusing on title and rating
                        st.markdown(
                            f"""
                            <div class="card">
                                <p class="title">{titles[i]}</p>
                                <p class="rating">Rating: ⭐ {ratings[i]}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
        else:
            st.write("🚫 No books found by this author.")

elif search_option == "Rating":
    min_rating = st.slider("📊 Select Minimum Rating", 0.0, 10.0, 3.0, 0.1)
    max_rating = st.slider("📊 Select Maximum Rating", 0.0, 10.0, 10.0, 0.1)
    
    if st.button('🔍 Search by Rating'):
        titles, ratings = search_books_by_rating(min_rating, max_rating)
        
        if len(titles) > 0:
            cols = st.columns(5)
            for i, col in enumerate(cols):
                if i < len(titles):
                    with col:
                        # Displaying book details with titles and ratings
                        st.markdown(
                            f"""
                            <div class="card">
                                <p class="title">{titles[i]}</p>
                                <p class="rating">Rating: ⭐ {ratings[i]}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
        else:
            st.write("🚫 No books found within this rating range.")






