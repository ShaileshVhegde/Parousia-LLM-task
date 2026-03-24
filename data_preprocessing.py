from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from feature_engineering import TitanicFeatureEngineer

def build_pipeline(classifier):
    """
    Constructs an end-to-end Scikit-Learn Pipeline that includes:
    1. Custom feature engineering
    2. Missing value imputation
    3. Scaling for numerical features
    4. One-Hot encoding for categorical features
    5. Final classifier predictive model
    """
    
    # We define the columns the transformed pipeline will expect
    numeric_features = ['Age', 'Fare', 'FamilySize']
    categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup', 'IsAlone']

    # Step A: Numeric processing (Impute Median -> StandardScale)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Step B: Categorical processing (Impute Mode -> OneHotEncode)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Step C: Bundle processing within a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drops columns like 'PassengerId', 'Ticket', 'Cabin', 'Name'
    )

    # Compile the complete full dataset Pipeline
    pipeline = Pipeline(steps=[
        ('fe', TitanicFeatureEngineer()),       # Engineer robust features FIRST
        ('preprocessor', preprocessor),         # Preprocess and scale NEXT
        ('classifier', classifier)              # Train the chosen algorithm LAST
    ])
    
    return pipeline, preprocessor
