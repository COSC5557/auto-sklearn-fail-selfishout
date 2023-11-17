# Modified Version
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imblearn_pipeline
from autosklearn.classification import AutoSklearnClassifier
from sklearn import preprocessing


X, y = fetch_openml(data_id=40691, as_frame=True, return_X_y=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), [i for i in range(X_train.select_dtypes(include=['int', 'float']).shape[1])]),
        ('cat', preprocessing.OneHotEncoder(handle_unknown='ignore'), [i for i in range(X_train.select_dtypes(include=['category', 'object']).shape[1])])
    ]
)

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('imputer', SimpleImputer(strategy='mean')),
    ('classifier', RandomForestClassifier(random_state=42))
])

cv = StratifiedKFold(n_splits=5)
rf_cv_scores = cross_val_score(rf_pipeline, X_train, y_train, cv=cv, scoring='accuracy')

print(f"Random Forest cross-validation accuracy: {rf_cv_scores.mean():.4f}")

oversample_pipeline = make_imblearn_pipeline(
    SMOTE(),
    rf_pipeline
)


rf_cv_scores_smote = cross_val_score(oversample_pipeline, X_train, y_train, cv=cv, scoring='accuracy')
print(f"Random Forest with SMOTE cross-validation accuracy: {rf_cv_scores_smote.mean():.4f}")


automl = AutoSklearnClassifier(
    time_left_for_this_task=300,  # 5 minutes
    memory_limit=10240,  
    resampling_strategy='cv',
    resampling_strategy_arguments={'folds': 5},
    ensemble_size=1,  
    seed=42,  
    n_jobs=-1  
)


automl.fit(X_train, y_train)


y_hat_automl = automl.predict(X_test)
print("AutoML Accuracy on test set:", accuracy_score(y_test, y_hat_automl))


print(automl.sprint_statistics())



# Results:
# Random Forest with SMOTE cross-validation accuracy: 0.6239
# AutoML Accuracy on test set: 0.684375
# auto-sklearn results:
  # Dataset name: cf110c42-84df-11ee-9468-0242ac1c000c
  # Metric: accuracy
  # Best validation score: 0.683346
  # Number of target algorithm runs: 81
  # Number of successful target algorithm runs: 61
  # Number of crashed target algorithm runs: 16
  # Number of target algorithms that exceeded the time limit: 4
  # Number of target algorithms that exceeded the memory limit: 0
