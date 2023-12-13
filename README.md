[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=12536514&assignment_repo_type=AssignmentRepo)
# Auto-sklearn Fail

The code in `fail.py` runs
[auto-sklearn](https://automl.github.io/auto-sklearn/master/) on a dataset for 5
minutes. The model it finds after that time is *worse* than just using a random
forest with default hyperparameters.

Find out what's going on, why auto-sklearn's performance is so bad, and how to
fix it.

# Response: 
Five minutes may be insufficient for the algorithm to explore the model space adequately. Memory space is important to perform its operations, which can be critical for larger datasets. Another remedy to make it more performant is applying some preprocessing techniques. Defines a preprocessing pipeline with scaling for numerical features and one-hot encoding for categorical features. Using resampling strategies like cross-validation. The final results are: 

Random Forest cross-validation accuracy: 0.6724
AutoML Accuracy on test set: 0.684375
auto-sklearn results:
  Dataset name: 8ccd64a6-7f30-11ee-998c-0242ac1c000c
  Metric: accuracy
  Best validation score: 0.679437
  Number of target algorithm runs: 26
  Number of successful target algorithm runs: 12
  Number of crashed target algorithm runs: 8
  Number of target algorithms that exceeded the time limit: 6
  Number of target algorithms that exceeded the memory limit: 0
