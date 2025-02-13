# ELEVEN STRATEGY X CentraleSupélec | The Endless Line 


## :mega: Team

Groupe 3

Léo BLANC, Juliette LACROIX, Hugo LE CORRE, Philippe MIRANDA-JEAN, Théo ROSSI, Elizaveta VASILEVA


## :dart: Project Overview

The goal of the project is to accurately forecast waiting times of rides at PortAventura Park to improve customer satisfaction. Our team followed a rigorous methodology that includes data preprocessing, exploratory data analysis, modeling, forecasting and dashboard visualization, to ensure our recommendations would accomplish our objective:
* Reduce Waiting Time
* Improve Satisfaction
* No Large Investment


## :open_file_folder: Project Structure 


### Data Preparation

#### 1. Data Overview

The data used for this project is stored in a zip file named 'final_data.7z' and contains:
* attendance.csv
* entity_schedule.csv
* link_attraction_park.csv
* parade_night_show.xlsx
* waiting_times.csv
* weather_data.csv


#### 2. Data Cleaning

* **processing.ipynb** : Data Cleaning & Pre-Processing


#### 3. Merging and Data Pre-Processing

* **merging_covid_EDA.ipynb**: Merging table with all the attributes from the different data sources, matching a 15 minutes interval


### Exploratory Data Analysis

* **EDA.ipynb**: Using the cleaned data from processing.ipynb, exploratory data analysis to give insights on the data 


### Modeling



#### 1. Predicting Attendance

* **model_attendance.ipynb**: Attendance forecast model that compares the performances of a Machine Learning model and a Time Series model. Performances are evaluated with the Mean Average Error and the Root Mean Squared Error. We found that the Exponential Smoothing Time Series model was the best performer of all models tried. 

#### 2. Predicting Waiting Times

* **wait_time_forecast.ipynb**: Waiting time forecast model *[...]*


### Dashboard Visualization

An interactive and user-friendly dashboard application to visualize and gather insights on predictions at PortAventura Park. Allows the user to:
* Get information on predicted attendance for a specific day
* Get information on predicted waiting times for each rides
* Visualize impact of BIG Strategy

The application is accessible by running **myapp.py**. 


## :rocket: Getting Started

* Install the required dependencies: ```requirements.txt```
* Download locally the data: ```final_data.7z```
* Navigate through data analysis: ```EDA.ipynb```
* Run the model: ```model_attendance.ipynb```
* Run the dashboard application: ```myapp.py```

## :construction: Future Improvements

* Improve simulation to better simulate direction of visitors across attractions waiting lines and the system of winning fast passes.
* Enhance user interface of the Streamlit app.
* Launch the **Win the Fast Pass** application for the visitors.
