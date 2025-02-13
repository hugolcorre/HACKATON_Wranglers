# ELEVEN STRATEGY X CentraleSupélec | The Endless Line 


## Team

Groupe 3

Léo BLANC, Juliette LACROIX, Hugo LE CORRE, Philippe MIRANDA-JEAN, Théo ROSSI, Elizaveta VASILEVA


## Overview

The goal of the project is to accurately forecast waiting times of rides at PortAventura Park to improve customer satisfaction. Our team followed a rigorous methodology that includes data preprocessing, exploratory data analysis, modeling, forecasting and dashboard visualization, to ensure our recommendations would accomplish our objective:
* Reduce Waiting Time
* Improve Satisfaction
* No Large Investment


## Project Structure 


### Data Preparation

#### 1. Data Overview

The data used for this project is store in a zip file named 'final_data.7z' and contains:
* attendance.csv
* entity_schedule.csv
* link_attraction_park.csv
* parade_night_show.xlsx
* waiting_times.csv
* weather_data.csv


#### 2. Data Cleaning

* processing.ipynb

Data cleaning included:
* Converting the date to the appropriate format
* Droping NA values
* Dropping unecessary columns


#### 3. Merging and Data Pre-Processing

* merging_covid_EDA.ipynb: Merging table with all the attributes from the different data sources

Pre-Processing included:
* Dropping the COVID period
* Dealing with the closing schedule of the park
* Duplicating data to match a 15 minutes interval


### Exploratory Data Analysis

* EDA.ipynb: Using the cleaned data from processing.ipynb, exploratory data analysis to give insights on the data 

### Modeling

#### 1. Predicting Attendance

#### 2. Predicting Waiting Times


### Dashboard Visualization
