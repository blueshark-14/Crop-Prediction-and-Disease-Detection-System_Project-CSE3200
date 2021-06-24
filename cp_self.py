# -*- coding: utf-8 -*-
"""
@author: MD MOSSADEK TOUHID
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read Dataset
dataset = pd.read_csv('cp_self.csv')
X = dataset.iloc[:, [0,1]].values
Y = dataset.iloc[:, 4].values

#Encoding categorical data
#Encoding Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
"""labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])

columnTransformer = ColumnTransformer([('JustName', OneHotEncoder(), [0])], remainder = 'passthrough')
X = np.array(columnTransformer.fit_transform(X), dtype = np.float64)"""

#Encoding Dependent Variable
labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(Y)


#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)"""

#Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(C=1.0, kernel = 'linear', random_state = 0)
classifier.fit(X_train, Y_train)


#Visualizing overall dataset
fig, ax = plt.subplots()
colors = {'Rice':'green', 'Jute':'blue', 'Tobacco':'red','Mango':'yellow','Maize':'black'}

grouped = dataset.groupby('Label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='Temperature', y='Rainfall', label=key, color=colors[key])
plt.show()

#Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green','blue','yellow','black')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('yellow','blue','black','red', 'green'))(i), label = j)
plt.title('Crop Prediction (Training set)')
plt.xlabel('Temperature')
plt.ylabel('Rainfall')
plt.legend()
plt.show()

#Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green','blue','yellow','black')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('yellow', 'blue','black','red','green'))(i), label = j)
plt.title('svm (Test set)')
plt.xlabel('temp')
plt.ylabel('Rainfall')
plt.legend()
plt.show()

#What type of crop will be predicted?
temp = input("Temperature(celsius) : ")
rainfall = input("Rainfall(mm) : ")

in_data_for_prediction = [[temp,rainfall]]

p_res = classifier.predict(in_data_for_prediction)

if p_res[0] == 0:
    print('\nGiven Crop is of type : Jute')
elif p_res[0] == 1:
    print('\nGiven Crop is of type : Maize')
elif p_res[0] == 2:
    print('\nGiven Crop is of type : Mango')
elif p_res[0] == 3:
    print('\nGiven Crop is of type : Rice')
else:
    print('\nGiven Crop is of type : Tobacco')    

#Accuracy Test
print(classifier.score(X_test, Y_test))


#Visualizing overall dataset
"""import seaborn as sns
sns.lmplot('Temperature', 'Rainfall',data=dataset, hue='Label', fit_reg=False)
plt.show()"""



