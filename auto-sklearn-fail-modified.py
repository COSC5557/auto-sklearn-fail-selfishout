# Modified Version
import sklearn.model_selection
from sklearn.datasets import fetch_openml
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

X, y = fetch_openml(data_id=40691, as_frame=True, return_X_y=True)
enc = OneHotEncoder(handle_unknown='ignore')
X = enc.fit_transform(X)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf = clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)
print("RF Accuracy", sklearn.metrics.accuracy_score(y_test, y_hat))

from autosklearn.classification import AutoSklearnClassifier

automl = AutoSklearnClassifier(
    time_left_for_this_task=20 * 60,  # 20 minutes
    memory_limit=10240,  # 10 GB
    resampling_strategy='cv',
    resampling_strategy_arguments={'folds': 5},
    ensemble_size=1,  # Focus on the single best model
    seed=42,  # For reproducibility
    n_jobs=-1  # Use all available CPUs
)

automl.fit(X_train, y_train)
y_hat_automl = automl.predict(X_test)
print("AutoML Accuracy:", sklearn.metrics.accuracy_score(y_test, y_hat_automl))
print(automl.sprint_statistics())
