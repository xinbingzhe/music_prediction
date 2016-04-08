# -*- coding: utf-8 -*-
# @Date    	: 2016-04-08 20:26:35
# @Author  	: mr0cheng
# @email	: c15271843451@gmail.com
# refer		:http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#example-linear-model-plot-ols-py
import os,sys,csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from matplotlib.legend_handler import HandlerLine2D
ARTIST_P_D_C=os.path.join(sys.path[0],'artist_p_d_c.txt')

'''
return:
	[]:size for 183
'''
def loadData(artist):
	with open(ARTIST_P_D_C,'r',encoding='utf-8') as fr:
		artist_id=fr.readline().strip('\n')
		while artist_id:
			play=list(map(int,fr.readline().strip('\n').split(',')))
			download=list(map(int,fr.readline().strip('\n').split(',')))
			collect=list(map(int,fr.readline().strip('\n').split(',')))
			if artist==artist_id:
				return play
			artist_id=fr.readline().strip('\n')

diabetes_X=[i for i in range(183)]
diabetes_X=np.array(diabetes_X)
diabetes_X=diabetes_X[:, np.newaxis]
diabetes_Y=loadData('40bbb0da5570702dd6ff3af5e9e3aea6')
print('diabetes_Y',diabetes_Y)
# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_Y[:-30]
diabetes_y_test = diabetes_Y[-30:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The intercept
print('Intercept: \n', regr.intercept_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

# Plot outputs
plt.scatter(diabetes_X_train, diabetes_y_train,  color='black')
plt.scatter(diabetes_X_test, diabetes_y_test,  color='red')
train=plt.plot(diabetes_X_train, regr.predict(diabetes_X_train), color='black',
         linewidth=3)
test=plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='red',
         linewidth=3)
plt.legend([train[0], test[0]], ["train","test"])
plt.show()