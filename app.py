import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Function to load the dataset (use st.cache_data for caching)
@st.cache_data
def load_data():
    # Use the path to your dataset directly
    file_path = 'https://raw.githubusercontent.com/fkabanga7/book_cat/main/AllITBooks_DataSet%202.xlsx'
    return pd.read_excel(file_path)

# Function to preprocess the book descriptions
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    # Remove stopwords and non-alphabetical characters
    return ' '.join([word for word in text.split() if word.lower() not in stop_words and word.isalpha()])

# Function to extract features using TF-IDF
def extract_features(descriptions):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    return vectorizer.fit_transform(descriptions)

# Function to perform K-Means clustering
def perform_kmeans(features, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(features)
    return kmeans

# Streamlit UI
def main():
    st.title("Book Categorization App")
    
    # Load the dataset without file uploader
    df = load_data()

    # Normalize column names by stripping whitespace and making them lowercase
    df.columns = df.columns.str.strip().str.lower()

    # Ensure the column exists by checking lowercase version
    if 'book_name' not in df.columns:
        st.error("The dataset does not contain a 'book_name' column.")
        return

    # Check if the dataset contains a 'description' column
    if 'description' not in df.columns:
        st.error("The dataset does not contain a 'description' column.")
        return
    
    # Preprocess the descriptions
    df['processed_description'] = df['description'].apply(preprocess_text)
    
    # Extract features from the descriptions
    features = extract_features(df['processed_description'])
    
    # Perform KMeans clustering
    num_clusters = st.slider("Select number of clusters:", 2, 10, 5)
    kmeans = perform_kmeans(features, num_clusters)
    
    # Add the cluster labels to the dataframe
    df['Cluster'] = kmeans.labels_

    # Ask the user to input a book name for categorization
    book_name = st.text_input("Enter the book's name to categorize:")

    if book_name:
        # Search for the book in the dataset
        book_row = df[df['book_name'].str.contains(book_name, case=False, na=False)]
        
        if not book_row.empty:
            # Show the book's information and its assigned cluster
            st.write("Book found:", book_row[['book_name', 'description', 'Cluster']].iloc[0])
            
            # Display the category (Cluster) for the entered book
            cluster = book_row['Cluster'].values[0]
            st.write(f"The book belongs to Cluster {cluster}")
            
            # Show the related topics for the book's cluster
            related_books = df[df['Cluster'] == cluster]
            top_keywords = related_books['processed_description'].str.split().explode().value_counts().head(2)
            st.write("Related Topics:", ", ".join(top_keywords.index))
        else:
            st.write("Book not found in the dataset. Please try a different name.")
    
        # Display the categorized books
        st.write("Books Categorized into Clusters:")
        st.dataframe(df[['book_name', 'description', 'Cluster']])

        # Visualize the clusters (PCA for dimensionality reduction)
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(features.toarray())
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=df['Cluster'], palette='Set2', s=100)
        plt.title("Books Categorization - KMeans Clusters")
        st.pyplot(plt)

if __name__ == '__main__':
    main()

