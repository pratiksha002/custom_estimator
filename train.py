import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from src.preprocessor import SmartPreprocessor
from sklearn.model_selection import GridSearchCV
import joblib
import os

#sample dataset
data = pd.read_csv("data/car data.csv")
target_column = "Selling_Price"
data = data.dropna(subset=[target_column])
x = data.drop(target_column, axis=1)
x = x.drop("Car_Name", axis=1)
y = data[target_column]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ("preprocessing", SmartPreprocessor()),
    ("model", Ridge())
])

param_grid = {
    "preprocessing__add_polynomial" : [False],
    "preprocessing__degree" : [1, 2],
    "preprocessing__handle_outliers" : [True, False],
    "preprocessing__handle_skew" : [True, False],
    "preprocessing__feature_selection": [False],
    "model__alpha" : [0.1, 1.0, 10.0]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=2,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    error_score="raise"
)

print("Training model...")
grid.fit(x_train, y_train)

best_model = grid.best_estimator_

print("\nBest Parameter:")
print(grid.best_params_)

preds = best_model.predict(x_test)

mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("\nEvaluation Results:")
print("MSE:", mse)
print("R2 Score:", r2)

feature_names = best_model.named_steps["preprocessing"].get_feature_names_out()
print("\nGenerated Features:")
print(feature_names)

# ensure models folder exists
os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/best_model.pkl")
print("\nModel saved to models/best_model.pkl")