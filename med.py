# -*- coding: utf-8 -*-
"""
Created on Thu May 30 00:43:18 2024

@author: aicha
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import pydeck as pdk
import plotly.express as px

# Page configuration
st.set_page_config(page_title="Olympic Medals Prediction 2024", layout="wide")

# CSS styles for Olympic colors
st.markdown("""
    <style>
    .reportview-container {
        background: linear-gradient(to right, #0057e7, #ffd700, #000000, #28a745, #ff3333);
        color: white;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(to right, #0057e7, #ffd700, #000000, #28a745, #ff3333);
        color: white;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #0057e7;
    }
    .stButton>button {
        background-color: #0057e7;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Path to the image
image_path = r'C:/Users/aicha/OneDrive/Documents/A MASERATI/M2 MASERATI/Mémoire 2023-2024/med.jpg'

# Load your data
data = pd.read_excel(r'C:/Users/aicha/Downloads/BDD.xlsx')

# Verify data is loaded
if data is not None:
    # Add geographic coordinates for each country
    coordinates = {
        'Argentina': [-38.4161, -63.6167],
        'Australia': [-25.2744, 133.7751],
        'Austria': [47.5162, 14.5501],
        'Morocco': [31.7917, -7.0926],
        'Belgium': [50.8503, 4.3517],
        'Brazil': [-14.2350, -51.9253],
        'Bulgaria': [42.7339, 25.4858],
        'Canada': [56.1304, -106.3468],
        'Chile': [-35.6751, -71.5430],
        'China': [35.8617, 104.1954],
        'Colombia': [4.5709, -74.2973],
        'Cuba': [21.5218, -77.7812],
        'Denmark': [56.2639, 9.5018],
        'Finland': [61.9241, 25.7482],
        'France': [46.6034, 1.8883],
        'Germany': [51.1657, 10.4515],
        'Greece': [39.0742, 21.8243],
        'Hungary': [47.1625, 19.5033],
        'India': [20.5937, 78.9629],
        'Iran': [32.4279, 53.6880],
        'Ireland': [53.1424, -7.6921],
        'Iceland': [64.9631, -19.0208],
        'Italy': [41.8719, 12.5674],
        'Japan': [36.2048, 138.2529],
        'Korea': [35.9078, 127.7669],
        'Mexico': [23.6345, -102.5528],
        'Norway': [60.4720, 8.4689],
        'New Zealand': [-40.9006, 174.8860],
        'Pakistan': [30.3753, 69.3451],
        'Poland': [51.9194, 19.1451],
        'Romania': [45.9432, 24.9668],
        'South Africa': [-30.5595, 22.9375],
        'Spain': [40.4637, -3.7492],
        'Sweden': [60.1282, 18.6435],
        'Turkey': [38.9637, 35.2433],
        'United Kingdom': [55.3781, -3.4360],
        'United States': [37.0902, -95.7129]
    }
    data['coordinates'] = data['entity'].map(coordinates)

    # Sort data by year
    data = data.sort_values(by='year')

    # Define features and targets
    features = ['entity', 'population_t_4', 'GDP_T_4', 'number_of_sports', 'host_region']
    targets = ['Gold', 'Silver', 'Bronze', 'Total']

    # Fill missing values with appropriate values
    data[features] = data[features].fillna(data[features].mean())

    # Ensure no infinite values
    data[features] = data[features].replace([float('inf'), -float('inf')], float('nan'))
    data[features] = data[features].fillna(data[features].mean())

    # Select training data up to 2020
    train_data = data[data['year'] <= 2020]

    X_train = train_data[features]
    y_train = train_data[targets]

    # Encode categorical variables and handle unknown categories
    categorical_features = ['entity', 'host_region']
    numerical_features = ['population_t_4', 'GDP_T_4', 'number_of_sports']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Create model pipelines
    models = {
        'Gold': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ]),
        'Silver': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ]),
        'Bronze': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ]),
        'Total': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
    }

    # Train the models
    for medal, model in models.items():
        model.fit(X_train, y_train[medal])

    # Mapping of original variable names to user-friendly names
    variable_names = {
        'entity': 'Country',
        'population_t_4': 'Population (4 years ago)',
        'GDP_T_4': 'GDP (4 years ago)',
        'number_of_sports': 'Number of Sports',
        'host_region': 'Host Region',
        'Gold': 'Gold Medals',
        'Silver': 'Silver Medals',
        'Bronze': 'Bronze Medals',
        'Total': 'Total Medals'
    }

    # Function to show descriptive statistics
    def show_descriptive_stats():
        st.write("### Descriptive Statistics")
        st.write("This page shows the descriptive statistics of the data used for the predictions.")
        
        # Separate numerical and categorical features
        numerical_data = data[numerical_features + targets]
        categorical_data = data[categorical_features]

        # Rename columns
        renamed_numerical_data = numerical_data.rename(columns=variable_names)
        renamed_categorical_data = categorical_data.rename(columns=variable_names)

        # Display descriptive statistics for numerical variables
        st.write("#### Numerical Variables")
        st.write(renamed_numerical_data.describe())

        # Display frequency counts for categorical variables
        st.write("#### Categorical Variables")
        for column in renamed_categorical_data:
            if column in variable_names:
                st.write(f"**{variable_names[column]}**")
            else:
                st.write(f"**{column}**")
            st.write(renamed_categorical_data[column].value_counts())

    # Function to show predictions
    def show_predictions():
        st.write("### Medal Predictions for 2024")
        st.write("This page presents the medal predictions for the 2024 Olympic Games.")
        
        # Select entity (country)
        entities = data['entity'].unique()
        if 'United States' in entities:
            selected_entity = st.selectbox('Select a country', entities, index=list(entities).index('United States'))
        else:
            selected_entity = st.selectbox('Select a country', entities)

        # Select medal type to predict
        medal_type = st.selectbox('Select the medal type to predict', targets)

        # Filter data for the selected country in 2024
        X_entity_2024 = data[(data['year'] == 2024) & (data['entity'] == selected_entity)][features]

        # Make predictions
        if not X_entity_2024.empty:
            prediction = models[medal_type].predict(X_entity_2024)
            st.write(f'Predicted {variable_names[medal_type]} for {selected_entity} in 2024: {round(prediction[0])}')
            st.write(f"**Interpretation:** The prediction for {variable_names[medal_type].lower()} for {selected_entity} in 2024 is based on past trends and available data. This includes factors like population, GDP, and the number of sports practiced. The predictions can help identify countries likely to perform well in the upcoming Olympics.")
        else:
            st.write(f'No data available for {selected_entity} in 2024.')

        # Display actual medals in 2020
        entity_2020 = data[(data['year'] == 2020) & (data['entity'] == selected_entity)]
        if not entity_2020.empty:
            actual_medals_2020 = entity_2020[medal_type].values[0]
            st.write(f'Actual {variable_names[medal_type]} for {selected_entity} in 2020: {actual_medals_2020}')
            st.write(f"**Interpretation:** The past performance in 2020 shows that {selected_entity} won {actual_medals_2020} {variable_names[medal_type].lower()}. This serves as a reference to evaluate the predictions for 2024.")
        else:
            st.write(f'No data available for {selected_entity} in 2020.')

        # Show country location on the map
        selected_coordinates = coordinates.get(selected_entity)
        if selected_coordinates:
            st.write("### Country Location on the Map")
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=pd.DataFrame([selected_coordinates], columns=['lat', 'lon']),
                get_position=["lon", "lat"],
                get_color="[200, 30, 0, 160]",
                get_radius=500000,
                pickable=True,
            )
            view_state = pdk.ViewState(latitude=selected_coordinates[0], longitude=selected_coordinates[1], zoom=3)
            r = pdk.Deck(layers=[layer], initial_view_state=view_state)
            st.pydeck_chart(r)

        # Add an interactive Plotly bar chart
        fig = px.bar(data[data['year'] <= 2020], x='entity', y=medal_type, title=f'{variable_names[medal_type]} Distribution by Country (up to 2020)', color='entity')
        st.plotly_chart(fig, use_container_width=True, height=600)

    # Home page
    def show_home():
        st.title("Home")
        st.write("Welcome to the Olympic medals prediction application.")
        st.write("Use the sidebar to navigate to the different sections.")
        st.write("""
            ### User Guide:
            1. **Home**: This home page where you find an introduction to the application.
            2. **Descriptive Statistics**: Displays descriptive statistics of the data used.
            3. **Predictions 2024**: Allows you to make medal predictions for the 2024 Olympic Games.
               - Select a country and the type of medal to see the predictions based on historical data.
        """)
        st.image(image_path, caption='Olympic Games', use_column_width=True)
        st.write("""
            ### Database:
            This database contains historical data from the Olympic Games since 1965,
            including information on populations, GDP, and the number of sports practiced by each country.
        """)

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Descriptive Statistics", "Predictions 2024"])

    # Display the selected page
    if page == "Home":
        show_home()
    elif page == "Descriptive Statistics":
        show_descriptive_stats()
    elif page == "Predictions 2024":
        show_predictions()
else:
    st.write("Error: Data could not be loaded.")

# Command to run the Streamlit application
# streamlit run app.py
