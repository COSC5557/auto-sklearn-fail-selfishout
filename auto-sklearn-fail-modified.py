# Modified Version
import sklearn.model_selection
from sklearn.datasets import fetch_openml
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

X, y = fetch_openml(data_id=40691, as_frame=True, return_X_y=True)

print(y)
class_distribution = y.value_counts()
print("class_distribution")
print(class_distribution)


def check_imbalance(y):
    # Calculate the class distribution
    class_distribution = y.value_counts()
    total_count = class_distribution.sum()

    # Calculate the percentage of each class
    class_percentage = (class_distribution / total_count) * 100

    # Determine if the dataset is imbalanced
    imbalance = any(class_percentage < 20)  # Threshold of 20%

    return class_percentage, imbalance



class_percentage, is_imbalanced = check_imbalance(y)
print("Class Distribution (%):\n", class_percentage)
print("\nIs the dataset imbalanced?", is_imbalanced)


class_distribution = y.value_counts()
class_distribution_percent = (class_distribution / class_distribution.sum()) * 100


# Plotting the class distribution
plt.figure(figsize=(8, 6))
class_distribution_percent.plot(kind='bar', color='skyblue')
plt.title('Class Distribution in Wine Dataset (%)')
plt.xlabel('Class')
plt.ylabel('Percentage')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()





enc = OneHotEncoder(handle_unknown='ignore')
# X = enc.fit_transform(X)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf = clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)
print("RF Accuracy", sklearn.metrics.accuracy_score(y_test, y_hat))

from autosklearn.classification import AutoSklearnClassifier

automl = AutoSklearnClassifier(time_left_for_this_task=300,
                               resampling_strategy='cv',
                               # resampling_strategy_arguments={'folds': 5},
                               memory_limit=6000,
                               # ensemble_size=1,
                               # ensemble_nbest=100,
                               per_run_time_limit=50
                               #delete_tmp_folder_after_terminate=False,
                               # n_jobs=1
                               )
automl.fit(X_train, y_train)
y_hat = automl.predict(X_test)
print("AutoML Accuracy", sklearn.metrics.accuracy_score(y_test, y_hat))
print(automl.sprint_statistics())


# Results:
# RF Accuracy 0.67
# AutoML Accuracy 0.68
# auto-sklearn results:
# Dataset name: cffadd39-8e6c-11ee-962d-0242ac1c000c
# Metric: accuracy
# Best validation score: 0.686405
# Number of target algorithm runs: 14
# Number of successful target algorithm runs: 11
# Number of crashed target algorithm runs: 0
# Number of target algorithms that exceeded the time limit: 3
# Number of target algorithms that exceeded the memory limit: 0
