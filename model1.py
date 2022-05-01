import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle


dataset = pd.read_csv('Admission_Predict.csv')
x = dataset.iloc[:,1:-1].values
x1=dataset.iloc[:,1:-1]
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

pred=regressor.predict(x_test)

r2_score = regressor.score(x_test,y_test)

pickle.dump(regressor, open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))

