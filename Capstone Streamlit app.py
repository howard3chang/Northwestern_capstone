import streamlit as st
import pandas as pd
import numpy as np

import requests
import json
import edgedb
import string
import ast
import gensim
import gensim.models
import os
from gensim.models import FastText
from sklearn.cluster import DBSCAN
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from datetime import datetime
import seaborn as sns

################################  try adding chatbot to help narrow down - this is for langchain/
load_dotenv(dotenv_path=".env")  # this load API keys and passwords without revealing them

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


from langchain import PromptTemplate
from langchain.llms import OpenAI
llm = OpenAI(model_name="text-davinci-003")

# system message is like "you are a helpful assistant" what you use to configure the system
# Human message is the user message
from langchain.chat_models import ChatOpenAI

st.title("MSDS 498 Recipe Recommender")
st.subheader("Created by RCH Consulting") 
st.subheader("Christopher Lee, Howard Chang, Richard Tsau- Northwestern University")

st.subheader("This application is to help user determine what recipes they should make given their current inventory of ingredients")

######


# get the available ingredients from the user through user input
# Initial list of common ingredients
common_ingredients = [
        "salt", "butter", "sugar", "egg", "onion",
        "garlic clove", "water", "olive oil", "flour",
        "milk", "pepper", "brown sugar", "lemon juice", "baking powder"
]

st.write("Do you have these 15 common ingredients: salt, butter, sugar, egg, onion, garlic clove, water, olive oil, flour, milk, pepper, brown sugar, flour, lemon juice, baking powder")
yesno_options = ['yes','no']
yesno= st.select_slider("Choose yes or no",
                        options=yesno_options)

# Build the available ingredients list
if yesno == 'Yes':
    available_ingredients = common_ingredients.copy()
else:
    missing_ingredients = st.text_input("List the ingredients that you do not have (comma-separated):")
    missing_ingredients = [ingredient.strip() for ingredient in missing_ingredients.split(',')]
    available_ingredients = [ingredient for ingredient in common_ingredients if ingredient not in missing_ingredients]


st.write('sample list: garlic, parsley, baking soda, parmesan cheese, carrot, vanilla, black pepper, cinnamon, tomato, sour cream, vanilla extract, margarine, green onion')
st.write('saffron, hot green chili pepper, garlic, clove, peppercorn, cardamom seed, cumin seed, poppy seed, mace, cilantro, mint leaf, yogurt, chicken, ghee, tomato, basmati rice, long grain rice, raisin, cashew')
st.write('chicken breast, heavy cream, chicken broth, garlic power, spinach, sun tomato, eggplant, okra')
# Allow the user to input additional ingredients
additional_ingredients = st.text_input("List any additional ingredients you have (comma-separated/lower case):")
additional_ingredients = [ingredient.strip() for ingredient in additional_ingredients.split(',')]
        
# Combine missing and additional ingredients to get available ingredients
available_ingredients = [ingredient for ingredient in common_ingredients if ingredient not in missing_ingredients]
available_ingredients.extend(additional_ingredients)


# Display the available ingredients list
st.write("Available Ingredients:")
st.write(', '.join(available_ingredients))

##### 


    
#### time selection for recipes

st.subheader("Recipe- Total Time")
time_options = ['30 minutes or less', 'over 30 minutes']
time = st.select_slider("Choose a time option",
                        options=time_options)

if time == '30 minutes or less':
    X = "30 minutes or less"
    time_filter = ".TotalTime <= 30"
elif time == 'over 30 minutes':
    X = "over 30 minutes"
    time_filter = ".TotalTime > 30"


st.write("The Total Time choosen is:", time)


###### Calorie selection for recipes

# Calorie selection for recipes
st.subheader("Recipe- Calories")
Calorie_options = ['400 calories or less', 'over 400 calories']
Calorie = st.select_slider("Choose a calorie option", options=Calorie_options)

if Calorie == '400 calories or less':
    Y = "400 calories or less"
    Calorie_filter = ".Calories <= 400"
elif Calorie == 'over 400 calories':
    Y = "over 400 calories"
    Calorie_filter = ".Calories > 400"

st.write("Calorie filter set to:", Calorie_filter)


#######

###### Dessert or Non-Dessert selection for recipes

# Dessert selection for recipes

st.subheader("Recipe - Dessert or Non-Dessert Category")
Dessert_options = ['Dessert', 'Non-Dessert']
Category = st.select_slider("Choose a category option", options=Dessert_options)

if Category == 'Dessert':
    Z = "Dessert"
    Category_filter = ".RecipeCategory = 'Dessert'"
else:
    Z = "Non-Dessert"
    Category_filter = ".RecipeCategory != 'Dessert'"

#######


# Combine filters using AND condition
combined_filter = f"{time_filter} AND {Calorie_filter} AND {Category_filter}"



# Create an EdgeDB client
client = edgedb.create_client()


if time_filter is not None and Calorie_filter is not None:
        # Build the query based on time option
          # Build the query based on time option and available ingredients
      
        query = f'''
            
            SELECT Recipe {{
                Calories,
                CarboyhdrateContent,
                CholestrerolContent,
                FatContent,
                FiberContent,
                Keywords,
                ingredient_num,
                ProteinContent,
                RecipeCategory,
                RecipeId,
                RecipeInstructions,
                SaturatedFatContent,
                SugarContent,
                name,
                Calories_PerServing,
                CarbohydrateContent_PerServing,
                FatContent_PerServing,
                FiberContent_PerServing,
                Popularity,
                ProteinContent_PerServing,
                Description,
                SaturatedFatContent_PerServing,
                SodiumContent_PerServing,
                SugarContent_PerServing,
                ingredients,
                TotalTime,
            }}
            FILTER {combined_filter};
            

        '''

#query = 'SELECT Recipe { Calories, CarboyhdrateContent, CholestrerolContent, FatContent, FiberContent, Keywords, ingredient_num, ProteinContent, RecipeCategory, RecipeId, RecipeInstructions, SaturatedFatContent, SugarContent, name, Calories_PerServing, CarbohydrateContent_PerServing, FatContent_PerServing, FiberContent_PerServing, Popularity, ProteinContent_PerServing, Description, SaturatedFatContent_PerServing, SodiumContent_PerServing, SugarContent_PerServing, ingredients}'  # recipe pull
result = client.query(query)

# Create an empty dataframe
recipe_df = pd.DataFrame(columns=['Calories', 'Carboyhdrate', 'CholestrerolContent', 'FatContent', 'FiberContent', 'Keywords', 'ingredient_num', 'ProteinContent', 'RecipeCategory', 'RecipeId', 'Recipeinstructions', 'SaturatedFatContent', 'SugarContent', 'name', 'Calories_PerServing', 'CarbohydrateContent_Perserving', 'FatContent_PerServing', 'FiberContent_PerServing', 'Popularity', 'ProteinContent_PerServing', 'Description', 'SaturatedFatContent_PerServing', 'SodiumContent_PerServing', 'SugarContent_PerServing', 'ingredients'])  # recipe pull

st.subheader("Inital query to database")
recipe_df = pd.DataFrame(result)
st.dataframe(recipe_df)



############# generate all potential recipes based on the ingredients avaialable


# Create an empty DataFrame to store potential recipes
potential_df = pd.DataFrame(columns=recipe_df.columns)

# Function to check if a recipe can be made with the given available_ingredients of ingredients
def can_make_recipe(recipe_ingredients, available_ingredients):
    return all(ingredient in available_ingredients for ingredient in recipe_ingredients)

# List to store recipes that can be made with the available_ingredients of ingredients
potential_recipes = []

# Iterate through each index and row (as Series) in the dataframe
for index, row in recipe_df.iterrows():
    recipe_ingredients = row['ingredients']  # Assuming 'ingredients' is the correct column name
    recipe_ingredients_updated = ast.literal_eval(recipe_ingredients)
    if can_make_recipe(recipe_ingredients_updated, available_ingredients):
            potential_df.loc[len(potential_df)] = row

### get rid of total recipe nutritional columns so that we only have PerServing for apples to apples comparison
columns_to_drop = [
    'Popularity', 'CarboyhdrateContent', 'CholestrerolContent', 'FatContent',
    'FiberContent', 'Keywords', 'ingredient_num', 'ProteinContent',
    'SaturatedFatContent', 'SugarContent'
]

potential_df.drop(columns=columns_to_drop, inplace=True)

st.subheader("Potential Recipes")
#st.dataframe(potential_df) this was a sanity check


######## initial Potential Recipe Dataframe created - for all available ingredients

#################################

potential_recipe_ids = potential_df['RecipeId'].tolist()

RecipeEmbedding_query = f'''
    SELECT RecipeEmbedding {{
        RecipeId,
        combined,
        embeddings
    }}
    FILTER .RecipeId IN {{ {', '.join(map(str, potential_recipe_ids))} }};
'''

result = client.query(RecipeEmbedding_query)

# Process the query results and create a list of dictionaries
embeddings_data = []
for row in result:
    embeddings_array = row.embeddings  # This is the array you want to convert
    embeddings_list = [float(value) for value in embeddings_array]  # Convert to float
    embeddings_data.append({
        'RecipeId': row.RecipeId,
        'combined': row.combined,
        'embeddings': embeddings_list  # Store the converted array
    })
#############


# Create a DataFrame from the list of dictionaries
embeddings_df = pd.DataFrame(embeddings_data)
#st.dataframe(embeddings_df) this was a sanity check
################################

combined_df = pd.merge(potential_df, embeddings_df, on='RecipeId', how='inner')
st.dataframe(combined_df)

Spice_query = f'''
    SELECT Spice {{
        Spice,
        Description
        
    }}
    
'''

result = client.query(Spice_query)

spice_df = pd.DataFrame(result)
#st.dataframe(spice_df) sanity check

Ingredient_query = f'''
    SELECT Ingredient {{
        Ingredient,
        Season
        
    }}
    
'''
result = client.query(Ingredient_query)

season_df = pd.DataFrame(result)
filtered_ingredient_df = season_df[season_df['Season'] != 'Year-round']
#st.dataframe(season_df) #sanity check
#st.dataframe(filtered_ingredient_df) sanity check

############################################ Cluster the potential recipes via DBSCAN
## use DBSCAN because we dont know how many clusters will appear - the choices change per user input 

# Extract the embeddings from the DataFrame
embeddings = combined_df['embeddings'].tolist()

# Convert the embeddings to a numpy array
embeddings_array = np.array(embeddings)

# Create a DBSCAN instance
dbscan = DBSCAN(eps=1.4, min_samples=6)  # EPS is distance between point (the smaller this is the closer they are together, min samples is how many points to a cluster)

# Fit the model to the embeddings
cluster_labels = dbscan.fit_predict(embeddings_array)

# Add the cluster labels to the DataFrame
combined_df['cluster'] = cluster_labels

# Count the occurrences of each cluster label
cluster_counts = Counter(cluster_labels)


######################################################################### seasonal ingredients

############################### datetime try for seasonal ingredients - need to fix the dataframe
# Define the start and end dates for each season
seasons = {
    "Spring": ((3, 21), (6, 20)),
    "Summer": ((6, 21), (9, 22)),
    "Fall": ((9, 23), (12, 20)),
    "Winter": ((12, 21), (3, 20))
}

# Get the current date
current_date = datetime.now()

# Determine the current season
current_season = None
for season, (start_date, end_date) in seasons.items():
    if (start_date <= (current_date.month, current_date.day) <= end_date):
        current_season = season
        break

# Display the current season using Streamlit - we actually don't need to write this out
st.write("Current Season:", current_season)

# Split the 'Season' column into individual seasons
season_df['Season'] = season_df['Season'].str.split('/')

# Create a list to store the ingredients in season
ingredients_in_season = []

# Loop through each available ingredient
for ingredient in available_ingredients:
    # Check if the ingredient is in season for the current season
    if any(current_season in seasons for seasons in season_df[season_df['Ingredient'] == ingredient]['Season']):
        ingredients_in_season.append(ingredient)

# Display the ingredients that are in season
st.write("Ingredients in Season:")
st.write(ingredients_in_season)

########################################  visualize the clusters with word cloud

# Get a list of unique cluster labels
unique_clusters = np.unique(cluster_labels)

# Display the distribution by label using Streamlit
st.write("Distribution by Cluster Label:")
for label, count in cluster_counts.items():
    st.write(f"Cluster {label}: {count} recipes")

# Loop through each unique cluster label
for label in unique_clusters:
    # Get the indices of recipes in the current cluster
    cluster_indices = np.where(cluster_labels == label)[0]

    # Get the recipes' text data for the current cluster
    cluster_text = combined_df.iloc[cluster_indices]['combined'].tolist()

    # Combine the text data into a single string
    cluster_text_combined = ' '.join(cluster_text)

    # Create a WordCloud for the current cluster
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cluster_text_combined)

    # Display the WordCloud using Streamlit
    st.write(f"Word Cloud for Cluster {label}:")
    st.image(wordcloud.to_array(), use_column_width=True)

    ############################### addition

     # Loop through recipes in the current cluster
for index in cluster_indices:
    ingredients = combined_df.loc[index, 'ingredients']
    ingredients_in_season_for_cluster = [ingredient for ingredient in ingredients if ingredient in ingredients_in_season]
        
    if ingredients_in_season_for_cluster:
        st.write(f"Ingredients in season for Cluster {label}, Recipe {index}:")
        st.write(ingredients_in_season_for_cluster)

######### add something that plots the max / average nutrition? - use plotly

#st.dataframe(combined_df) sanity check


# Loop through each unique cluster label and create a graph
for label in unique_clusters:
    # Get the indices of recipes in the current cluster
    cluster_indices = np.where(cluster_labels == label)[0]

    # Get the nutritional values for the current cluster
    cluster_nutrition = combined_df.iloc[cluster_indices][['Calories_PerServing', 'CarbohydrateContent_PerServing', 'FatContent_PerServing', 'FiberContent_PerServing', 'ProteinContent_PerServing']]

    # Calculate the average nutritional values
    average_nutrition = cluster_nutrition.mean()

    # Create a bar plot using Seaborn
    plt.figure(figsize=(10, 5))
    sns.barplot(x=average_nutrition.index, y=average_nutrition.values)
    plt.title(f"Average Nutritional Values for Cluster {label}")
    plt.ylabel("Average Value")
    plt.xlabel("Nutritional Category")
    plt.xticks(rotation=45)
    st.pyplot(plt)


################################## create a corpus for the cluster that the user finds interesting

st.text("From the above clusters of recipes choose one that interests you the most")
interest_query = st.text_input('Enter cluster label')
button = st.button("Submit")

if button:
    try:
        interest_label = int(interest_query)
        filtered_df = combined_df[combined_df['cluster'] == interest_label]

        ### move this block after so that the check can be conducted first
        #Create a corpus of all 'combined' values from the filtered DataFrame 
        corpus = filtered_df['combined'].tolist()

        # Now you have the corpus of combined values for the selected cluster
        st.write("Corpus of selected cluster:")
        st.write(corpus)

              

    except ValueError:
        st.write("Please enter a valid cluster label (integer).")

################################  try adding chatbot to help narrow down - this is for langchain/





###############################
template = """

you are an expert chef and food critic with an expertise in recipes and ingredients.
Explain the differences of the recipes in {concept} in a couple of lines.


"""

prompt = PromptTemplate (
    input_variables=["concept"],
    template=template,
)

x = llm(prompt.format(concept=corpus))

st.write(x)  ### try to figure out how to incorporate seasonal ingredients then taste?

##################################################### chain the prompts?

chain = LLMChain(llm=llm, prompt=prompt)

second_prompt = PromptTemplate (
    input_variables = ['ingredients'],
    template="Describe the flavors of the spices in each recipe {ingredients} and estimate the taste of the recipe"
)

chain_two = LLMChain(llm=llm, prompt=second_prompt)

overall_chain = SimpleSequentialChain(chains= [chain, chain_two], verbose=True)
explanation = overall_chain.run(corpus)

st.write(explanation)


###################################################  commented out - the results were not as good as the prompt template

#chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)  # temperature is the parameter that controls randomness the lower it is the more focues and determinisic the response
#messages = [
#            SystemMessage(content="you are an expert on food and recipes"),
#            HumanMessage(content=f"Please advise on the differences in each recipe and whether the ingredients are in season. Please put a line break between explanations. Corpus: {corpus}Ingredients in season: {ingredients_in_season}")

#           ]
#response=chat(messages)

#st.write(response)

###################################################
