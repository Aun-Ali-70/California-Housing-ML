from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import (RandomForestRegressor, HistGradientBoostingRegressor)
import joblib

data = datasets.fetch_california_housing()
x = data.data
y = data.target

print(x.shape)

poly = PolynomialFeatures()
x = poly.fit_transform(x)

print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,
                                                    random_state=432
                                                    )


model = HistGradientBoostingRegressor(
    max_iter=350,
    learning_rate=0.05
)
model.fit(x_train, y_train)
joblib.dump(model, 'my_model.joblib')
prediction = model.predict(x_test)
r2 = r2_score(y_test, prediction)
print(r2)


print(data.feature_names)
print(x_train[0])
print(y_train[0])