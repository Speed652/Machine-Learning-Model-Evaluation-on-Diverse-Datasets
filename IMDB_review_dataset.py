from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.pipeline import Pipeline
import pandas as pd
import os

# Loading the IMDB dataset
def load_imdb_data(folder):
    texts = []
    labels = []
    for label in ['neg', 'pos']:
        path = os.path.join(folder, label)
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), 'r', encoding='utf-8') as file:
                texts.append(file.read())
                labels.append(0 if label == 'neg' else 1)
    return texts, labels

train_texts, train_labels = load_imdb_data("C:\\Users\\Mohamed El Sayed\\Downloads\\aclImdb_v1\\aclImdb\\train")          #file was too big to warrant uploading to Github
test_texts, test_labels = load_imdb_data("C:\\Users\\Mohamed El Sayed\\Downloads\\aclImdb_v1\\aclImdb\\test")

# Splitting the data into features and labels
X = train_texts
y = train_labels

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining the models
models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000, solver='liblinear'), {'model__C': [0.001, 0.01, 0.1, 1, 10]}),  
    ('Decision Tree', DecisionTreeClassifier(), {'model__max_depth': [None, 10, 20, 30]}),
    ('Linear SVM', LinearSVC(dual=True), {'model__C': [0.001, 0.01, 0.1, 1, 10]}),  
    ('AdaBoost', AdaBoostClassifier(), {'model__n_estimators': [50, 100, 200]}),
    ('Random Forest', RandomForestClassifier(), {'model__n_estimators': [50, 100, 200]}),
]

# results
results = {'Model': [], 'Accuracy': []}

# Iterating through the models
for model_name, model, param_grid in models:
    # Create a pipeline with TfidfVectorizer and the model
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('model', model)
    ])
    
    # Find the best hyperparameters
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring=make_scorer(accuracy_score), n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Evaluate the model using cross-validation
    accuracy = cross_val_score(best_model, X_train, y_train, cv=5, scoring=make_scorer(accuracy_score)).mean()

    # Store the results
    results['Model'].append(model_name)
    results['Accuracy'].append(accuracy)

# Convert the results to a DataFrame 
results_df = pd.DataFrame(results)

# Print the results
print(results_df)

# Find the model with the highest accuracy
best_model_idx = results_df['Accuracy'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'Model']
best_model_accuracy = results_df.loc[best_model_idx, 'Accuracy']

print(f"\nThe best model is {best_model_name} with an accuracy of {best_model_accuracy:.2%}.")

# Test on the Test Set
test_accuracy = best_model.score(test_texts, test_labels)
print(f"\nTest Accuracy: {test_accuracy:.2%}")