
import streamlit as st
from streamlit_option_menu import option_menu
from numpy.core.fromnumeric import prod
import tensorflow as tf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import os

#Project Folder
project_folder = os.path.dirname(__file__)
print(project_folder)

# Import the Dataset
skincare = pd.read_csv(f"{project_folder}/MP-Skin Care Product Recommendation System3.csv", encoding='utf-8', index_col=None)

# Set the header for the Streamlit app
st.set_page_config(page_title="Simbolo AI", page_icon=f"{project_folder}/media/simbolo-icon.jfif", layout="centered")

# Example number for menu style
EXAMPLE_NO = 2

def streamlit_menu(example=1):
    if example == 1:
        # Sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",
                options=["Skin Care", "Get Recommendation", "Skin Care 101"],
                icons=["house", "stars", "book"],
                menu_icon="cast",
                default_index=0,
            )
        return selected

    if example == 2:
        # Horizontal menu without custom style
        selected = option_menu(
            menu_title=None,
            options=["Skin Care", "Get Recommendation", "Skin Care 101"],
            icons=["house", "stars", "book"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
        )
        return selected

    if example == 3:
        # Horizontal menu with custom style
        selected = option_menu(
            menu_title=None,
            options=["Skin Care", "Get Recommendation", "Skin Care 101"],
            icons=["house", "stars", "book"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "green"},
            },
        )
        return selected

selected = streamlit_menu(example=EXAMPLE_NO)

if selected == "Skin Care":
    
    st.title("AI Foundation Batch 16 - Team Data Odyssey")
    st.title(f"{selected} Product Recommender")
#    st.title(f"{selected} Product Recommender :sparkles:")
    st.write('---') 

    st.write(
        """
        ##### This project aims to develop a skincare product recommendation system using machine learning techniques. By leveraging TF-IDF and cosine similarity algorithms, the system effectively analyzes product descriptions, allowing for accurate recommendations based on user preferences.        
        """)
       
    st.write(
        """
        ##### The system incorporates user-defined parameters such as skin type, desired effects, and skin problems to generate tailored product suggestions. By considering factors like product category, notable effects, and user-specified criteria, the system provides a personalized experience for skincare consumers. 
        ##### The project successfully demonstrates the application of machine learning to a real-world problem, offering a potential solution to the challenges faced by consumers when selecting suitable skincare products.
        ##### Please select the *Get Recommendation* page to start receiving recommendations or choose the *Skin Care 101* page to see skincare tips and tricks.
        """)
    
    st.write(
        """
        Project Members: Myat Noe Kabyar, Moe Thet Paing & Lin Thein Naing
        """)
    
    st.info("Thank you for visiting us!")

if selected == "Get Recommendation":

    c = st.container(border=True)
    
    c.title(f"Let's {selected}")
    
    c.write(
        """
        ##### **To get recommendations, please enter your skin type, issues, and desired benefits to receive the right skincare product recommendations.**
        """) 
    
    #c.write('---') 

    first, last = st.columns(2)

    # Choose a product category
    category = first.selectbox(label='Product Category: ', options=skincare['product_type'].unique())
    category_pt = skincare[skincare['product_type'] == category]

    # Choose a skin type
    skin_type = last.selectbox(label='Your Skin Type: ', options=['Normal', 'Dry', 'Oily', 'Combination', 'Sensitive'])
    category_st_pt = category_pt[category_pt['skintype'] == skin_type]
    print(category_st_pt)

    # Select skin problems
    prob = st.multiselect(label='Skin Problems: ', options=['Dull Skin', 'Acne', 'Acne Scars', 'Large Pores', 'Dark Spots', 'Fine Lines and Wrinkles', 'Blackheads', 'Uneven Skin Tone', 'Redness', 'Sagging Skin'])
    

    # Choose notable effects
    opsi_ne = category_st_pt['notable_effects'].unique().tolist()
    selected_options = st.multiselect('Notable Effects: ', opsi_ne)
    category_ne_st_pt = category_st_pt[category_st_pt["notable_effects"].isin(selected_options)]
    #print(category_ne_st_pt)

    # Choose product
    opsi_pn = category_ne_st_pt['product_name'].unique().tolist()
    product = st.selectbox(label='Recommended Product for You', options=sorted(opsi_pn))

    # Displaying choosen product image
    choosen_pi = category_st_pt[category_st_pt['product_name'] == product]
    if not choosen_pi['picture_src'].empty:
        image_url = choosen_pi['picture_src'].iloc[0]
        st.write('---')
        st.image(image_url,width=500)
    else:
          st.write("Please select a recommended product.")

    
    # MODELLING with Content Based Filtering
    tf = TfidfVectorizer()

    # Calculate idf for 'notable_effects'
    tf.fit(skincare['notable_effects']) 

    # Map feature index to feature names
    features = tf.get_feature_names_out()
    
    # to delete later - check out the features in "Notable Effects"
    #print(features)

    # Transform data to matrix
    tfidf_matrix = tf.fit_transform(skincare['notable_effects']) 

    # Check tfidf matrix size
    shape = tfidf_matrix.shape

    # to delete later - checking columns & rows
    #print(shape)

    # Convert tf-idf matrix to dense format
    tfidf_matrix.todense()

    # Create dataframe to view tf-idf matrix
    pd.DataFrame(
        tfidf_matrix.todense(), 
        columns=tf.get_feature_names_out(),
        index=skincare.product_name
    ).sample(shape[1], axis=1).sample(10, axis=0)

    # Compute cosine similarity on tf-idf matrix
    cosine_sim = cosine_similarity(tfidf_matrix) 

    # Create dataframe for cosine similarity
    cosine_sim_df = pd.DataFrame(cosine_sim, index=skincare['product_name'], columns=skincare['product_name'])

    # View similarity matrix
    cosine_sim_df.sample(7, axis=1).sample(10, axis=0)

    

    # Function to get recommendations
    def skincare_recommendations(product_name, similarity_data=cosine_sim_df, items=skincare[['product_name', 'brand', 'description','picture_src']], k=5):

        # Find the most similar products
        index = similarity_data.loc[:, product_name].to_numpy().argpartition(range(-1, -k, -1))
        closest = similarity_data.columns[index[-1:-(k+2):-1]]

        # Exclude the selected product from recommendations
        closest = closest.drop(product_name, errors='ignore')
        df = pd.DataFrame(closest).merge(items, on='product_name').head(k)
        return df

    # Button to display recommendations
    model_run = st.button('Find Other Similar Products!')
    if model_run:
        st.write('Here are other similar product recommendations based on your preferences:')
        #st.write(skincare_recommendations(product))
        recommendations = skincare_recommendations(product)
        st.write(recommendations)

        for index, row in recommendations.iterrows():
            st.markdown(f"[{row['product_name']}]({row['picture_src']})")  # Create hyperlink with image as target
            st.write(f"Brand: {row['brand']}")
            st.write(f"Description: {row['description']}")
            st.image(row['picture_src'])    

if selected == "Skin Care 101":
    st.title(f"Take a Look at {selected}")
    st.write('---')
    st.video(f"{project_folder}/media/Skincare_Story.mov", format="video/mov", start_time=0, loop=True, autoplay=True, muted=False) 
