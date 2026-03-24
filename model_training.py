from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

def get_models():
    """
    Returns a dictionary structure specifying baseline algorithms and 
    associated Hyperparameter tuning grids (param_grid) designed explicitly
    for scikit-learn's generic `GridSearchCV`.
    Keeping it exactly to 2 models as per assignment requirements.
    """
    models = {
        'Logistic Regression (Baseline)': {
            'model': LogisticRegression(max_iter=2000, random_state=42),
            'params': {
                'classifier__C': [0.1, 1.0, 10.0]
            }
        },
        'Gradient Boosting (Optimized)': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__learning_rate': [0.01, 0.05, 0.1],
                'classifier__max_depth': [3, 4, 5]
            }
        }
    }
    return models

def tune_model(pipeline, param_grid, X_train, y_train):
    """
    Fits hyperparameters to optimizing strategy using generic 5-Fold cross-validation.
    """
    grid_search = GridSearchCV(
        estimator=pipeline, 
        param_grid=param_grid, 
        cv=5, 
        scoring='accuracy', 
        n_jobs=-1, 
        verbose=1
    )
    
    # We pass the completely raw X_train through our custom robust Pipeline 
    # instead of pre-processing. The GridSearchCV automatically refits the pipeline repeatedly.
    grid_search.fit(X_train, y_train)
    
    print(f"Optimal Parameters: {grid_search.best_params_}")
    print(f"Best CV Score (Accuracy): {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_score_, grid_search.best_params_
