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

The data used for this project is stored in a zip file named **final_data.7z** and contains:
* attendance.csv
* entity_schedule.csv
* link_attraction_park.csv
* parade_night_show.xlsx
* waiting_times.csv
* weather_data.csv


#### 2. Data Cleaning

* processing.ipynb : Data Cleaning & Pre-Processing


#### 3. Merging and Data Pre-Processing

* merging_covid_EDA.ipynb: Merging table with all the attributes from the different data sources, matching a 15 minutes interval


### Exploratory Data Analysis

* EDA.ipynb: Using the cleaned data from processing.ipynb, exploratory data analysis to give insights on the data 

### Modeling

#### 1. Predicting Attendance

#### 2. Predicting Waiting Times


### Dashboard Visualization

An interactive and user-friendly dashboard application to visualize and gather insights on predictions at PortAventura Park. Allows the user to:
* Get information on predicted attendance for a specific day
* Get information on predicted waiting times for each rides
* Visualize impact of BIG Strategy

The application is accessible by running **myapp.py**. 


## :rocket: Getting Started


## :construction: Future Improvements
