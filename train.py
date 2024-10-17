from sklearn.linear_model import LinearRegression
import numpy as np
import joblib
import os

x = np.arange(0,10,0.1)
y = 3*x + 2

x = np.reshape(x, (-1,1))
y = np.reshape(y, (-1,1))

model = LinearRegression()
model.fit(x,y)
joblib.dump(model, os.path.join(os.getcwd(),"model.pkl"))
print("Model Created")
