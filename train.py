import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from src.preprocessor import SmartPreprocessor

#sample dataset
data = pd.DataFrame({
    "area" : [1000, 1200, 1300, 1500, None],
    "bedrooms" : [2, 3, 3, 4, 3],
    "location" : ["A", "B", "A", "C", "B"],
    "price" : [200000, 250000, 270000, 300000, 260000]
})

x = data.drop("price", axis=1)
y = data["price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

pipeline = Pipeline([
    ("preprocessing", SmartPreprocessor(add_polynomial=True, handle_outliers=True))
    ("model", Ridge())
])

pipeline.fit(x_train, y_train)

preds = pipeline.predict(x_test)

mse = mean_squared_error(y_test, preds)
print("MSE:", mse)
