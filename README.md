# AniRec
Developed a full-stack anime recommendation system generating personalized titles from user preferences and watch history.

## Problem
Most anime recommendation systems rely heavily on popularity or collaborative filtering, which fails for new or niche users.  
AniRec focuses on learning user preferences from semantic features of anime descriptions to generate personalized recommendations.

## Machine Learning Approach

### Final Model
- **Ridge Regression with Word Embeddings**
- Implemented in `RidgeReg_WordEmb.py`

Anime descriptions and metadata are embedded into a semantic vector space. A ridge regression model is trained to predict user preference scores, allowing the system to rank unseen anime titles.

### Why Ridge Regression?
- Stable and interpretable
- Performs well with high-dimensional embeddings
- Reduces overfitting compared to standard linear regression

## Model Selection
Multiple models were prototyped during development, including:
- Neural Networks
- Standard Ridge Regression
- SGD-based regressors

Ridge regression with word embeddings was selected for the final system due to its consistent performance, faster training time, and improved generalization on sparse user preference data.
