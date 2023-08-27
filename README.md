# Northwestern_capstone
Capstone project 

Project-centered around making a recipe recommender based on Kaggle's Food.com dataset.

After EDA and cleaning the ingredients and recipe instructions were used to create custom embeddings via Fasttext.
These embeddings were clustered via DBSCAN then visualized by word cloud and the nutritional data by Seaborn.

Finally, the text data was analyzed by LangChain to describe the differences of recipes within a cluster and provide a description of the taste/flavor of each recipe.

The main file are Capstone.py and Capstone Streamlit app.py - they are the same file.

The notebooks are what was created in google colab then transferred into the streamlit python file.
