# Credit Risk Prediction Model

This repository contains a machine learning project focused on predicting credit risk for loan applicants. The project leverages both internal bank data and external CIBIL data to classify the likelihood of loan approval into four priority categories: P1, P2, P3, and P4. The model selection and hyperparameter tuning were performed using advanced techniques, ultimately deploying a LightGBM classifier with optimized parameters.

## Table of Contents

- [Project Overview](#project-overview)
- [Data](#data)
- [Feature Engineering](#feature-engineering)
- [Model Selection](#model-selection)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Evaluation](#evaluation)

## Project Overview

The goal of this project is to develop a robust machine learning model to predict the creditworthiness of loan applicants. The target variable is a multi-class label representing different priority levels of loan approval: P1, P2, P3, and P4. The project involves data preprocessing, feature selection, model selection using the `lazypredict` library, and hyperparameter tuning using `Optuna`.

## Data

### Datasets

- **Internal Bank Dataset**: Contains historical data on customer features such as income, education status, max_recent_level_of_deliq, time_since_recent_payment,Tot_Missed_Pmnt etc.
- **CIBIL Dataset**: Contains external credit data for same user.

The datasets were merged to form a comprehensive dataset  with a shape of `(42064, 55)`.

### Target Variable

- **Approved Flag**: Multi-class target with categories P1, P2, P3, and P4 (where p1 suggest highest priority).

## Feature Engineering

Several feature engineering techniques were applied to prepare the data for modeling:

- **Chi-Square Test**: Used for feature selection to determine the relationship between categorical features and the target variable.
- **ANOVA Test**: Applied to assess the significance of continuous features in relation to the target variable.
- **One-Hot Encoding**: Employed to convert categorical variables into a format suitable for model training.
- **Scaling**: Standardization of numerical features to ensure uniformity across the dataset.

## Model Selection

Due to the large size of the dataset, a subset of 5,000 samples was used to perform rapid model prototyping using the `lazypredict` library. This approach allowed for the quick evaluation of multiple machine learning models to identify the most promising candidate.

### Selected Model

- **LightGBM Classifier**: Based on the initial model selection, the LightGBM classifier was chosen for further tuning due to its superior performance on the subset.
- Result of lazyPredict
  ![Screenshot 2024-08-23 233128](https://github.com/user-attachments/assets/78fa8c97-744c-4cdf-8bc6-c67a1f52cc93)


## Hyperparameter Tuning

Hyperparameter tuning was performed using `Optuna` with 25 trials to optimize the performance of the LightGBM model.

### Best Hyperparameters

- **lambda_l1**: 0.9860474664463378
- **lambda_l2**: 0.0015413967296900833
- **num_leaves**: 20
- **feature_fraction**: 0.8560911734282852
- **bagging_fraction**: 0.5662466478553893
- **bagging_freq**: 2
- **min_child_samples**: 21

### Model Performance

- **Test Accuracy**: 0.7816
- **Test F1 Score**: 0.7657

These results indicate that the tuned LightGBM model provides a balanced performance in predicting the loan approval priorities.

## Evaluation

The model's performance was evaluated using standard metrics:

- **Accuracy**: The overall correctness of the model's predictions.
- **F1 Score**: The harmonic mean of precision and recall, particularly useful for evaluating multi-class classification.




