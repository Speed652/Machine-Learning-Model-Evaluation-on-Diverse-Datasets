from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pandas as pd
import os


# Defining the models 
models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000, solver='liblinear'), {'model__C': [0.001, 0.01, 0.1, 1, 10]}),  
    ('Decision Tree', DecisionTreeClassifier(), {'model__max_depth': [None, 10, 20, 30]}),
    ('Linear SVM', LinearSVC(dual=True), {'model__C': [0.001, 0.01, 0.1, 1, 10]}),  
    ('AdaBoost', AdaBoostClassifier(), {'model__n_estimators': [50, 100, 200]}),
    ('Random Forest', RandomForestClassifier(), {'model__n_estimators': [50, 100, 200]}),
]


def evaluate_model(model, X, y):
    cv_scores = cross_val_score(model, X, y, cv=5, scoring=make_scorer(accuracy_score))
    return cv_scores.mean()

def find_best_hyperparameters(model, X, y, param_grid):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring=make_scorer(accuracy_score), n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

# Loading the dataset
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
print(f"Number of documents: {len(twenty_train.data)}")

# Spliting the data into features and labels
X = twenty_train.data
y = twenty_train.target


# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#printing training and testing set sizes
print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Initializing results
results = {'Model': [], 'Accuracy': []}

# Iterating through the models
for model_name, model, param_grid in models:
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('model', model)
    ])
    
    # Find the best hyperparameters
    best_model = find_best_hyperparameters(pipeline, X_train, y_train, param_grid)

    accuracy = evaluate_model(best_model, X_train, y_train)

    # Store the results
    results['Model'].append(model_name)
    results['Accuracy'].append(accuracy)

results_df = pd.DataFrame(results)

# Print the results
print(results_df)

# Find and print the model with the highest accuracy
best_model_idx = results_df['Accuracy'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'Model']
best_model_accuracy = results_df.loc[best_model_idx, 'Accuracy']

print(f"\nThe best model is {best_model_name} with an accuracy of {best_model_accuracy:.2%}.")