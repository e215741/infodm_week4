import datasets
import regression


X,Y = datasets.load_linear_exsample1()

#ver1
#print(X)
#print(Y)

model = regression.LinearRegression()

model.x

#ver2
import importlib
importlib.reload(regression)
model = regression.LinearRegression()
model.fit(X, Y)
#print(model.theta)

#ver3
print(model.predict(X))