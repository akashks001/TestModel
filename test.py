import os
import joblib
import numpy as np


model = joblib.load(os.path.join(os.getcwd(),"model.pkl"))
x = 2
x = np.array(x)
x = np.reshape(x, (-1,1))
y = model.predict(x)
print(y)
