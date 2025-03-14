import mlflow
import numpy as np
import pandas as pd 

# TODO: Set tht MLFlow server uri
mlflow.set_tracking_uri("http://127.0.0.1:6001")

# TODO: Provide model path/url
logged_model = "runs:/cb6c506fa3244256b35eb6b787936637/digits_model"

# Load model as a PyFuncModel.
loaded_model = mlflow.sklearn.load_model(logged_model)

# Input a random datapoint
np.random.seed(42)
data = np.random.rand(1, 64)
df = pd.DataFrame(data)

# Predict the output
prediction = loaded_model.predict(df)


# Print out prediction result
print(f"Model Prediction: {prediction}")
