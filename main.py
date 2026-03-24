import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Custom modular implementations
from feature_engineering import TitanicFeatureEngineer
from data_preprocessing import build_pipeline
from model_training import get_models, tune_model
from evaluation import (evaluate_model_performance, plot_confusion_matrix, 
                        plot_roc_curve, plot_feature_importance, plot_model_comparison)

def main():
    print("====================================")
    print(" 🚀 ADVANCED TITANIC ML PIPELINE 🚀 ")
    print("====================================\n")
    
    # 1. Data Loading Strategies
    data_path = 'titanic.csv'
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset '{data_path}' is missing.")
        return
        
    print("-> Step 1: Loading raw data...")
    df = pd.read_csv(data_path)
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    # Advanced: Straified splitting to preserve survival proportions in splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   [Train Data] Shape: {X_train.shape}")
    print(f"   [Test Data]  Shape: {X_test.shape}\n")
    
    # 2. Architect Models Selection
    models = get_models()
    
    results = []
    roc_details = {}
    best_overall_accuracy = 0
    best_overall_model = None
    best_overall_name = ""
    
    print("-> Step 2: Training & Hybrid CV Hyperparameter Tuning...\n")
    
    for name, config in models.items():
        print(f"⚙️ Optimizing {name}...")
        
        # Generates a fresh scikit-learn pipeline for every model structure
        pipeline, preprocessor = build_pipeline(config['model'])
        
        # Tuning models completely cleanly on the train sets
        best_pipeline, best_cv_score, best_params = tune_model(
            pipeline, config['params'], X_train, y_train
        )
        
        # Validate effectively using precisely untouched Test subsets
        y_pred = best_pipeline.predict(X_test)
        y_prob = None
        if hasattr(best_pipeline, "predict_proba"):
            y_prob = best_pipeline.predict_proba(X_test)[:, 1]
            
        acc, prec, rec, f1 = evaluate_model_performance(name, y_test, y_pred, y_prob)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1,
            'Best CV': best_cv_score
        })
        
        roc_details[name] = {
            'y_test': y_test,
            'y_prob': y_prob
        }
        
        # Heatmap output dynamically rendering
        plot_confusion_matrix(name, y_test, y_pred)
        
        # Capture absolute best
        if acc > best_overall_accuracy:
            best_overall_accuracy = acc
            best_overall_model = best_pipeline
            best_overall_name = name

    # 3. Present Results
    results_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)
    
    # Store outputs specifically to meet exact print requirements explicitly
    m1_name = results[0]['Model']
    m1_acc = results[0]['Accuracy']
    m1_cv = results[0]['Best CV']
    
    m2_name = results[1]['Model']
    m2_acc = results[1]['Accuracy']
    m2_cv = results[1]['Best CV']
    
    print("\n===========================================")
    print(" 🏆 FINAL MODEL COMPARISON SUMMARY 🏆")
    print("===========================================")
    print(results_df.to_string(index=False))
    
    print("\n===========================================")
    print(f" Model 1 ({m1_name}) Accuracy: {m1_acc:.4f}")
    print(f" Model 2 ({m2_name}) Accuracy: {m2_acc:.4f}")
    print(f" Cross-validation scores (Model 1): {m1_cv:.4f}")
    print(f" Cross-validation scores (Model 2): {m2_cv:.4f}")
    print(f" Best model based on performance: {best_overall_name.upper()} ")
    print("===========================================\n")
    
    print("-> Step 3: Generating ultimate visual charts...")
    plot_model_comparison(results_df)
    plot_roc_curve(roc_details)
    
    # Feature engineering importances exclusively outputted for optimal performing model
    best_preprocessor = best_overall_model.named_steps['preprocessor']
    plot_feature_importance(best_overall_model, best_preprocessor, best_overall_name)
    
    print("\n✅ Execution completed successfully! View your artifacts in the 'plots/' directory.")

if __name__ == "__main__":
    main()
