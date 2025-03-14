import mlflow
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from utility import pipeline

# Set MLFlow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:6001")

# Set experiment name
email = "jnuta@uic.edu"
experiment_name = f"{email}-lab6"
mlflow.set_experiment(experiment_name)

# Load dataset
X_train, X_test, y_train, y_test = pipeline.data_preprocessing()

# ✅ Define model parameters
params = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42,
}

# ✅ Train Random Forest model
print("Training Random Forest model...")
trained_model = RandomForestClassifier(**params)
trained_model.fit(X_train, y_train)
print("✅ Model training complete.")

# ✅ Ensure evaluation function receives the correct model object
print("Evaluating model...")
accuracy = pipeline.evaluation(X_test, y_test, trained_model)  # X_test, y_test must be first!
print(f"📊 Model Accuracy: {accuracy}")

# ✅ Log model and metrics to MLFlow
with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.set_tag("Training Info", "Random Forest model for digits_model data")

    # Infer the model signature
    signature = infer_signature(X_train, trained_model.predict(X_train))

    # ✅ Register new model version in MLFlow
    mlflow.sklearn.log_model(
        sk_model=trained_model,
        artifact_path="random_forest_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="13eab921",
    )

print("🎯 ✅ Random Forest Model training complete and registered as Version 2 in MLFlow!")
