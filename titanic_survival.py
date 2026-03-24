import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_data(filepath):
    """Loads the dataset from the given filepath."""
    print("--- 1. Data Loading ---")
    df = pd.read_csv(filepath)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    return df

def clean_data(df):
    """Handles missing values and drops unnecessary columns."""
    print("\n--- 2. Data Cleaning ---")
    df_cleaned = df.copy()
    
    # Fill Age with median
    age_median = df_cleaned['Age'].median()
    df_cleaned['Age'].fillna(age_median, inplace=True)
    print(f"Filled missing 'Age' values with median: {age_median}")
    
    # Fill Embarked with mode
    embarked_mode = df_cleaned['Embarked'].mode()[0]
    df_cleaned['Embarked'].fillna(embarked_mode, inplace=True)
    print(f"Filled missing 'Embarked' values with mode: '{embarked_mode}'")
    
    # Drop Cabin column due to high number of missing values
    if 'Cabin' in df_cleaned.columns:
        df_cleaned.drop('Cabin', axis=1, inplace=True)
        print("Dropped 'Cabin' column (too many missing values).")
        
    return df_cleaned

def encode_and_engineer_features(df):
    """Encodes categorical variables and creates new features."""
    print("\n--- 3 & 4. Encoding and Feature Engineering ---")
    df_fe = df.copy()
    
    # 4. Feature Engineering
    # Create FamilySize = SibSp + Parch
    df_fe['FamilySize'] = df_fe['SibSp'] + df_fe['Parch']
    print("Created 'FamilySize' feature.")
    
    # Create IsAlone feature
    df_fe['IsAlone'] = 0
    df_fe.loc[df_fe['FamilySize'] == 0, 'IsAlone'] = 1
    print("Created 'IsAlone' feature.")
    
    # Extract Title from Name
    if 'Name' in df_fe.columns:
        df_fe['Title'] = df_fe['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        # Group rare titles
        rare_titles = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
        df_fe['Title'] = df_fe['Title'].replace(rare_titles, 'Rare')
        df_fe['Title'] = df_fe['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
        print("Extracted and simplified 'Title' from 'Name'.")
    
    # Drop unnecessary columns
    cols_to_drop = ['PassengerId', 'Name', 'Ticket']
    existing_cols_to_drop = [c for c in cols_to_drop if c in df_fe.columns]
    df_fe.drop(columns=existing_cols_to_drop, inplace=True)
    print(f"Dropped unnecessary columns: {existing_cols_to_drop}")
    
    # 3. Encoding Categorical Variables
    # Label Encoding for Sex and Title
    le_sex = LabelEncoder()
    df_fe['Sex'] = le_sex.fit_transform(df_fe['Sex'])
    print("Encoded 'Sex' column to numerical values.")
    
    if 'Title' in df_fe.columns:
        le_title = LabelEncoder()
        df_fe['Title'] = le_title.fit_transform(df_fe['Title'])
        print("Encoded 'Title' column to numerical values.")
        
    # Label Encoding for Embarked
    le_embarked = LabelEncoder()
    df_fe['Embarked'] = le_embarked.fit_transform(df_fe['Embarked'])
    print("Encoded 'Embarked' column to numerical values.")
    
    return df_fe

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluates the model and prints accuracy, confusion matrix, and classification report."""
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    
    print(f"\n[{model_name} Evaluation]")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(cr)
    
    return acc

def plot_feature_importance(model, features, title="Feature Importance"):
    """Plots and saves the feature importance bar chart."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(title)
    sns.barplot(x=importances[indices], y=[features[i] for i in indices], palette="viridis")
    plt.xlabel('Relative Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("\n[Bonus] Feature importance plot saved as 'feature_importance.png'")

def main():
    # 1. Loading
    filepath = "titanic.csv"
    df = load_data(filepath)
    
    # 2. Cleaning
    df = clean_data(df)
    
    # 3 & 4. Encoding & Feature Engineering
    df = encode_and_engineer_features(df)
    
    print(f"\nFinal dataset shape before modeling: {df.shape}")
    print(f"Features being used: {list(df.columns)}")
    
    # Prepare X and y
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\n--- 5 & 6. Training Models and Evaluation ---")
    
    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_acc = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    
    # Random Forest (Main Focus)
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    rf_acc = evaluate_model(rf_model, X_test, y_test, "Random Forest (Baseline)")
    
    # Gradient Boosting (Optional)
    gb_model = GradientBoostingClassifier(random_state=42)
    gb_model.fit(X_train, y_train)
    gb_acc = evaluate_model(gb_model, X_test, y_test, "Gradient Boosting")
    
    print("\n--- 7. Cross Validation ---")
    # k-fold cross validation for Random Forest
    cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
    print(f"Random Forest 5-Fold CV Accuracies: {cv_scores}")
    print(f"Random Forest Mean CV Accuracy: {cv_scores.mean():.4f}")
    
    print("\n--- 8. Model Improvement (Hyperparameter Tuning) ---")
    print("Tuning RandomForest hyperparameters (n_estimators, max_depth)...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 15]
    }
    
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), 
                               param_grid=param_grid, 
                               cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_rf_model = grid_search.best_estimator_
    print(f"Best Parameters found: {grid_search.best_params_}")
    
    # Evaluate best model
    best_rf_acc = evaluate_model(best_rf_model, X_test, y_test, "Random Forest (Tuned)")
    
    print("\n--- 10. Output & Conclusion ---")
    print("FINAL MODEL PERFORMANCE COMPARISON:")
    print("-" * 40)
    print(f"{'Logistic Regression:':<25} {lr_acc:.4f}")
    print(f"{'Random Forest (Baseline):':<25} {rf_acc:.4f}")
    print(f"{'Gradient Boosting:':<25} {gb_acc:.4f}")
    print(f"{'Random Forest (Tuned):':<25} {best_rf_acc:.4f}")
    print("-" * 40)
    
    # Determine the best model
    accuracies = {
        "Logistic Regression": lr_acc,
        "Random Forest (Baseline)": rf_acc,
        "Gradient Boosting": gb_acc,
        "Random Forest (Tuned)": best_rf_acc
    }
    best_model_name = max(accuracies, key=accuracies.get)
    print(f"\n=> Best Model is '{best_model_name}' with an accuracy of {accuracies[best_model_name]:.4f}")
    
    print("\nApproach Explanation:")
    print("1. Data was loaded and missing values imputed (Age with median, Embarked with mode). The 'Cabin' column was dropped.")
    print("2. Unnecessary identifier features (PassengerId, Name, Ticket) were removed to avoid noise.")
    print("3. Both categorical labels (Sex, Embarked) and newly engineered ones (Title) were encoded numerically.")
    print("4. Feature engineering involved extracting 'Title' from 'Name', and establishing 'FamilySize' and 'IsAlone' properties.")
    print("5. Finally, we trained multiple models, proving robust estimation through K-fold CV and optimized Random Forest via Grid Search.")
    
    # 11. Bonus - Feature Importance Plot
    print("\n--- 11. Bonus: Feature Importance ---")
    plot_feature_importance(best_rf_model, list(X.columns), title="Feature Importance (Tuned Random Forest)")

if __name__ == "__main__":
    main()
