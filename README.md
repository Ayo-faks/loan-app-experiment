# Loan Approval System

This repository contains the code for a experimental loan approval system, encompassing exploratory data analysis (EDA), feature engineering, model training, natural language processing (NLP) integration, and conversational agents. The system leverages various machine learning models, including HistGradientBoostingClassifier, DistilBERT, and OpenAI's ChatGPT, to facilitate a robust and insightful loan approval decision-making process.

## Table of Contents

1. [Installation](#installation)
2. [Exploratory Data Analysis](#exploratory-data-analysis)
3. [Feature Engineering](#feature-engineering)
4. [Model Training](#model-training)
5. [Loan Application Summary](#loan-application-summary)
6. [Evaluation](#evaluation)
7. [Database Integration](#database-integration)
8. [LangChain and OpenAI Integration](#langchain-and-openai-integration)

## Installation

To run this code, ensure you have the required dependencies installed. You can install them using the following commands:

```bash
!pip install langchain openai pandas scikit-learn seaborn transformers torch wordcloud numpy==1.25.0 --quiet
```

## Exploratory Data Analysis

The exploratory data analysis section involves loading and visualizing the dataset. It includes a heatmap to display the correlation matrix of numeric features and scatter plots to explore relationships between variables.

## Feature Engineering

Categorical columns are identified and encoded, and one-hot encoding is applied to multi-category categorical variables. The dataset is prepared for model training using the HistGradientBoostingClassifier.

## Model Training

The machine learning model, HistGradientBoostingClassifier, is trained on the dataset, and permutation feature importance is calculated to identify the key factors influencing loan approval decisions.

## Loan Application Summary

The DistilBERT model is utilized to generate a loan application summary based on provided applicant details. The model prediction and approval decision are combined for a comprehensive output.

## Evaluation

The model's performance is evaluated using a confusion matrix, accuracy, precision, recall, and F1-score. The classification report provides detailed metrics.

## Database Integration

The code integrates the dataset with SQLite, saving the training and testing sets into a SQLite database.

## LangChain and OpenAI Integration

LangChain and OpenAI are integrated to create a conversational agent capable of responding to natural language instructions. The agent leverages the trained model and dataset to provide loan approval decisions based on user queries.

Feel free to explore the code and adapt it to your specific use case. Happy coding!
```