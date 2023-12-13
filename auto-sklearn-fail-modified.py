# Modified Version
import sklearn.model_selection
from sklearn.datasets import fetch_openml
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from autosklearn.classification import AutoSklearnClassifier

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

automl = AutoSklearnClassifier(time_left_for_this_task=360)
automl.fit(X_train, y_train)

y_train_pred_automl = automl.predict(X_train)
y_test_pred_automl = automl.predict(X_test)

print("Training Accuracy AutoML Simple:"accuracy_score(y_train, y_train_pred_automl))
print("Test Accuracy AutoML Simple:", accuracy_score(y_test, y_test_pred_automl))


automl = AutoSklearnClassifier(time_left_for_this_task=360, resampling_strategy='cv'
                               #resampling_strategy_arguments={'folds': 5}
                               )
automl.fit(X_train, y_train)
y_hat = automl.predict(X_test)
y_train_automl = cross_val_predict(automl, X_train, y_train, cv=5)
automl_train_accuracy = sklearn.metrics.accuracy_score(y_train, y_train_automl)
print("AutoML Training Accuracy:", automl_train_accuracy)
print("AutoML Test Accuracy", accuracy_score(y_test, y_hat))


# Results:
RF Accuracy: 0.6525
Training Accuracy AutoML Simple: 0.890742285237698
Test Accuracy AutoML Simple: 0.61
AutoML Training Accuracy: 0.5821517931609674
AutoML Test Accuracy 0.6475
