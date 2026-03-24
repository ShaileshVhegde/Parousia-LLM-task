from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import re

class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer for handling Titanic-specific 
    Feature Engineering. It implements BaseEstimator to easily 
    integrate inside a Scikit-Learn Pipeline.
    """
    
    def __init__(self):
        # We can store state/statistics if necessary (e.g. median for imputation)
        # But for pipeline purity, basic medians for bins or grouping logic
        self.age_median = None
        self.fare_median = None
        
    def fit(self, X, y=None):
        # Compute summary statistics on training data only to prevent data leakage
        self.age_median = X['Age'].median()
        self.fare_median = X['Fare'].median()
        return self
        
    def transform(self, X):
        X_out = X.copy()
        
        # We temporarily fill Age and Fare just to create robust groups
        # (The continuous equivalents will be properly imputed by SimpleImputer later)
        age_temp = X_out['Age'].fillna(self.age_median)
        fare_temp = X_out['Fare'].fillna(self.fare_median)
        
        # 1. Age Groups
        X_out['AgeGroup'] = pd.cut(age_temp, bins=[0, 12, 20, 40, 60, 100], 
                                   labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
        X_out['AgeGroup'] = X_out['AgeGroup'].astype(str)
        
        # 2. Fare Groups
        X_out['FareGroup'] = pd.qcut(fare_temp, 4, labels=['Low', 'Medium', 'High', 'VeryHigh'])
        X_out['FareGroup'] = X_out['FareGroup'].astype(str)
        
        # 3. Family Size & IsAlone
        X_out['FamilySize'] = X_out['SibSp'] + X_out['Parch'] + 1
        X_out['IsAlone'] = (X_out['FamilySize'] == 1).astype(int)
        
        # 4. Title Extraction
        def get_title(name):
            title_search = re.search(r' ([A-Za-z]+)\.', str(name))
            if title_search:
                return title_search.group(1)
            return "Unknown"
            
        X_out['Title'] = X_out['Name'].apply(get_title)
        
        rare_titles = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 
                       'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
        X_out['Title'] = X_out['Title'].replace(rare_titles, 'Rare')
        X_out['Title'] = X_out['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
        
        return X_out
