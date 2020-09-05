import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

data = sns.load_dataset("iris")
data.head()
X = data.iloc[:, :-1]

y = data.iloc[:, -1]

plt.xlabel('Features')

plt.ylabel('Species')

pltX = data.loc[:, 'sepal_length']

pltY = data.loc[:,'species']

plt.scatter(pltX, pltY, color='blue', label='sepal_length')

pltX = data.loc[:, 'sepal_width']

pltY = data.loc[:,'species']

plt.scatter(pltX, pltY, color='green', label='sepal_width')

pltX = data.loc[:, 'petal_length']

pltY = data.loc[:,'species']

plt.scatter(pltX, pltY, color='red', label='petal_length')

pltX = data.loc[:, 'petal_width']

pltY = data.loc[:,'species']

plt.scatter(pltX, pltY, color='black', label='petal_width')

plt.legend(loc=4, prop={'size':8})

plt.show()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()

model.fit(x_train, y_train)

predictions = model.predict(x_test)

print(predictions)

print()

print( classification_report(y_test, predictions) )

print( accuracy_score(y_test, predictions))