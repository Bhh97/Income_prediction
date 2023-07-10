# Adult Income Dataset Analysis and Prediction
This repository contains a Python-based project that performs an exploratory data analysis (EDA), preprocessing, and a machine learning prediction on the Adult Income Dataset from UCI Machine Learning Repository. The primary objective of the project is to predict whether income exceeds $50K/year based on census data. https://archive.ics.uci.edu/dataset/2/adult

## Dataset
The dataset used in this project, often referred to as the "Adult" dataset, includes various demographic, educational, and work-related information of about 32,000 adults. The data was extracted from the 1994 Census bureau database.

Data attributes include:
Age
Workclass
Final Weight (fnlwgt)
Education
Education Number of Years
Marital-status
Occupation
Relationship
Race
Sex
Capital Gain
Capital Loss
Hours Per Week
Native-country
Income (the label we're predicting)

## Data Preprocessing and Exploration
The data exploration step involves understanding the dataset, checking the data types, handling missing values, and exploring relationships between features. We use pandas, seaborn, and matplotlib libraries for these steps. The data preprocessing step includes handling missing data, encoding categorical variables, and normalizing numerical variables.

## Naive Bayes Classifier
I implemented a Naive Bayes Classifier from scratch to predict the 'Income' attribute of the dataset. The classifier handles both continuous features (using Gaussian distribution) and categorical features.

Our Naive Bayes Classifier includes:
Calculating Gaussian probability for continuous features.
Calculating probability for categorical features.
Predicting the class for a given row.
Evaluating the classifier by calculating metrics such as Mean Squared Error(MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error(MAE).
