import mlflow
import sklearn

# Set mlflow path

# S - GridSearch on model params
sklearn.model_selection.GridSearchCV(estimator, param_grid, scoring)

# S - GridSearch on datasetsize
sklearn.model_selection.GridSearchCV(estimator, param_grid, scoring)

# S - Imputer
# S - MinMax scaler
# S - Cross validation (with data unseen from above?)
sklearn.model_selection.cross_validate(estimator, X, y, scoring=[], cv=TrainTestSplit)

pipe = sklearn.pipeline.Pipeline(steps, memory=None, verbose)

# mlflow w/ autologging
with mlflow.start_run()
    pipe = generate_pipeline(X, y)
    pipe.fit(X,y)

# Then Query mlflow for items in post process YAML
