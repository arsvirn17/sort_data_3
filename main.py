# %matplotlib inline
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (10, 8)
import collections

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz

def create_df(dic, feature_list):
  out = pd.DataFrame(dic)
  out = pd.concat([out, pd.get_dummies(out[feature_list])], axis=1)
  out.drop(feature_list, axis=1, inplace=True)
  return out

def intersect_features(train, test):
  common_feat = list(set(train.keys()) & set(test.keys()))
  return train[common_feat], test[common_feat]

features = ['Внешность', 'Алкоголь_в_напитке',
            'Уровень_красноречия', 'Потраченные_деньги']

df_train = {}
df_train['Внешность'] = ['приятная', 'приятная', 'приятная', 'отталкивающая',
                         'отталкивающая', 'отталкивающая', 'приятная']
df_train['Алкоголь_в_напитке'] = ['да', 'да', 'нет', 'нет', 'да', 'да', 'да']
df_train['Уровень_красноречия'] = ['высокий', 'низкий', 'средний', 'средний', 'низкий',
                                   'высокий', 'средний']
df_train['Потраченные_деньги'] = ['много', 'мало', 'много', 'мало', 'много',
                                  'много', 'много']
df_train['Поедет'] = LabelEncoder().fit_transform(['+', '-', '+', '-', '-', '+', '+'])

df_train = create_df(df_train, features)
df_train

df_test = {}
df_test['Внешность'] = ['приятная', 'приятная', 'отталкивающая']
df_test['Алкоголь_в_напитке'] = ['нет', 'да', 'да']
df_test['Уровень_красноречия'] = ['средний', 'высокий', 'средний']
df_test['Потраченные_деньги'] = ['много', 'мало', 'много']
df_test = create_df(df_test, features)
df_test

y = df_train['Поедет']
df_train, df_test = intersect_features(train=df_train, test=df_test)
df_train

df_test
tree = DecisionTreeClassifier(random_state=17)
# %%time
tree.fit(df_train, y)

from sklearn.tree import export_graphviz

export_graphviz(
    tree, out_file="tree.dot", feature_names=df_train.columns, filled=True,
)

# !dot -Tpng tree.dot -o tree.png

balls = [1 for i in range(9)] + [0 for i in range(11)]

balls_left  = [1 for i in range(8)] + [0 for i in range(5)]
balls_right = [1 for i in range(1)] + [0 for i in range(6)]

from math import log

def entropy(a_list):
  lst = list(a_list)
  size = len(lst) * 1.0
  entropy = 0
  set_elements = list(set(lst))
  if set_elements in [0,1]:
    return 0
  for i in set(lst):
    occ = lst.count(i)
    entropy -= occ/size * log(occ/size, 2)
  return entropy

print(entropy(balls))
print(entropy(balls_left))
print(entropy(balls_right))
print(entropy([1,2,3,4,5,6]))

def information_gain(root, right, left):
  return entropy(root)- 1.0 * len(left)/len(root) * entropy(left) - 1.0 * len(right)/len(root) * entropy(right)

information_gain(balls, balls_right, balls_left)

def best_feature_to_split(X, y):
  out = []
  for i in X.columns:
    out.append(information_gain(y, y[X[i]==0], y[X[i]==1]))
  return out

best_feature_to_split(df_train, y)

data_train = pd.read_csv('adult_train.csv', sep=';')

data_train.tail()

data_test = pd.read_csv('adult_test.csv', sep=';')

data_test.tail()

data_test = data_test[(data_test['Target'] == ' >50K.') | (data_test['Target'] == ' <=50K.')]

data_train.loc[data_train['Target'] == ' <=50K', 'Target'] = 0
data_train.loc[data_train['Target'] == ' >50K', 'Target'] = 1

data_test.loc[data_test['Target'] == ' <=50K.', 'Target'] = 0
data_test.loc[data_test['Target'] == ' >50K.', 'Target'] = 1

data_test.describe(include='all').T

data_train['Target'].value_counts()

fig = plt.figure(figsize=(25,15))
cols = 5
rows = int(np.ceil(float(data_train.shape[1]) / cols))
for i, column in enumerate(data_train.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    if data_train.dtypes[column] == np.object:
        data_train[column].value_counts().plot(kind="bar", axes=ax)
    else:
        data_train[column].hist(axes=ax)
        plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.7, wspace=0.2)

data_train.dtypes

data_test.dtypes

data_test['Age'] = data_test['Age'].astype(int)

categorical_columns_train = [c for c in data_train.columns
                             if data_train[c].dtype.name == 'object']
numerical_columns_train = [c for c in data_train.columns
                           if data_train[c].dtype.name != 'object']

categorical_columns_test = [c for c in data_test.columns
                            if data_test[c].dtype.name == 'object']
numerical_columns_test = [c for c in data_test.columns
                          if data_test[c].dtype.name != 'object']

print('categorical_columns_test:', categorical_columns_test)
print('categorical_columns_train:', categorical_columns_train)
print('numerical_columns_test:', numerical_columns_test)
print('numerical_columns_train:', numerical_columns_train)

data_train['Workclass'].mode()

# fill nan
for c in categorical_columns_train:
    data_train[c] = data_train[c].fillna(data_train[c].mode())
for c in categorical_columns_test:
    data_test[c] = data_test[c].fillna(data_train[c].mode())

for c in numerical_columns_train:
    data_train[c] = data_train[c].fillna(data_train[c].median())
for c in numerical_columns_test:
    data_test[c] = data_test[c].fillna(data_train[c].median())

data_train = pd.concat([data_train, pd.get_dummies(data_train['Workclass'],
                                                   prefix="Workclass"),
                      pd.get_dummies(data_train['Education'], prefix="Education"),
                      pd.get_dummies(data_train['Martial_Status'], prefix="Martial_Status"),
                      pd.get_dummies(data_train['Occupation'], prefix="Occupation"),
                      pd.get_dummies(data_train['Relationship'], prefix="Relationship"),
                      pd.get_dummies(data_train['Race'], prefix="Race"),
                      pd.get_dummies(data_train['Sex'], prefix="Sex"),
                      pd.get_dummies(data_train['Country'], prefix="Country")],
                     axis=1)

data_test = pd.concat([data_test, pd.get_dummies(data_test['Workclass'], prefix="Workclass"),
                      pd.get_dummies(data_test['Education'], prefix="Education"),
                      pd.get_dummies(data_test['Martial_Status'], prefix="Martial_Status"),
                      pd.get_dummies(data_test['Occupation'], prefix="Occupation"),
                      pd.get_dummies(data_test['Relationship'], prefix="Relationship"),
                      pd.get_dummies(data_test['Race'], prefix="Race"),
                      pd.get_dummies(data_test['Sex'], prefix="Sex"),
                      pd.get_dummies(data_test['Country'], prefix="Country")],
                     axis=1)

data_train.drop(['Workclass', 'Education', 'Martial_Status',
                 'Occupation', 'Relationship', 'Race', 'Sex', 'Country'],
                axis=1, inplace=True)
data_test.drop(['Workclass', 'Education', 'Martial_Status', 'Occupation',
                'Relationship', 'Race', 'Sex', 'Country'],
               axis=1, inplace=True)

data_test.describe(include='all').T

set(data_train.columns) - set(data_test.columns)

data_train.shape, data_test.shape

data_test['Country_ Holand-Netherlands'] = np.zeros([data_test.shape[0], 1], dtype=int)

data_test = data_test.reindex(columns=data_train.columns)

set(data_train.columns) - set(data_test.columns)

data_train.head(2)

data_test.head(2)

X_train = data_train.drop(['Target'], axis=1)
y_train = data_train['Target']

X_test = data_test.drop(['Target'], axis=1)
y_test = data_test['Target']

tree = DecisionTreeClassifier(max_depth=3, random_state = 17)
tree.fit(X_train, y_train)

tree_predictions = tree.predict(X_test)

accuracy_score(y_test,tree_predictions)

tree_params = {"max_depth": list(range(1,15))}

localy_best_tree = GridSearchCV(DecisionTreeClassifier(random_state=17), tree_params, cv=5)
localy_best_tree.fit(X_train, y_train)


print("Best params: ", localy_best_tree.best_params_)
print("Best cross validation score: ", localy_best_tree.best_score_)

tuned_tree = DecisionTreeClassifier(max_depth=9, random_state=17)
tuned_tree.fit(X_train, y_train)

tuned_tree_predictions = tuned_tree.predict(X_test)

accuracy_score(y_test, tuned_tree_predictions)

rf = RandomForestClassifier(n_estimators=100, random_state=17)
rf.fit(X_train, y_train)

rf_predictions = rf.predict(X_test)

accuracy_score(y_test, rf_predictions)

forest_params = {'max_depth': range(10, 21),
                 'max_features': range(5, 105, 10)}

locally_best_forest = GridSearchCV(RandomForestClassifier(random_state=17,
                                                         n_jobs=-1),
                                 forest_params, cv=5)

locally_best_forest.fit(X_train, y_train)

print("Best params:", locally_best_forest.best_params_)
print("Best cross validaton score", locally_best_forest.best_score_)

tuned_forest_predictions = locally_best_forest.predict(X_test)
accuracy_score(y_test,tuned_forest_predictions)