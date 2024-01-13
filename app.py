import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st

def convert_views_to_numeric(views):
    if 'K' in views:
        return int(float(views.replace('K', '')) * 1000)
    elif 'M' in views:
        return int(float(views.replace('M', '')) * 1e6)
    elif 'B' in views:
        return int(float(views.replace('B', '')) * 1e9)
    elif ' views' in views:
        return int(''.join(filter(str.isdigit, views)))
    else:
        return int(views)


def load(file_path):
    df = pd.read_csv(file_path, encoding='utf-16')
    
    df['title'] = df['title'].str.lower().str.strip()
    df['view'] = df['view'].str.replace(' views', '')
    df['view'] = df['view'].apply(convert_views_to_numeric)
    
    return df



def recommend(df, input_title):
    tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['title'])


    cosine_similarities = linear_kernel(tfidf_vectorizer.transform([input_title]), tfidf_matrix).flatten()

    video_indices = cosine_similarities.argsort()[:-6:-1]

    return df.loc[video_indices, ['title', 'view', 'channel name', 'link']]


def display_recommendations(predictions):
    predictions=predictions.sort_values(by='view',ascending=False)
    st.write(predictions)

# Main Streamlit app
def main():
    st.title('Content-Based Video Recommender System')
    file_path = "dataset.csv"
    df = load(file_path)

    input_title = st.text_input('Enter a video title')
    if st.button("Get Recommendations"):
        predictions = recommend(df, input_title)
        
        display_recommendations(predictions)

# Run the Streamlit app
if __name__ == '__main__':
    main()
