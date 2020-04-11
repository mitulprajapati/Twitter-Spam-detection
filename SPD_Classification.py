# # By inserting The clean Pollutors_legitimate dataset

import pandas as pd
all_data = pd.read_csv('datasets/Clean_SPD_NEW_Tweets_with_class.csv')
all_data.rename(columns={'ID': 'UserID'}, inplace=True)
print(all_data)

# # Importing SPD_Created

SPD_Created = pd.read_csv('datasets/SPD_Created.csv')
SPD_Created['AccountCreated'] = SPD_Created['AccountCreated'].str.extract('(\d\d\d\d)', expand=True)
SPD_Created = SPD_Created.drop(columns=['Unnamed: 2'])
print(SPD_Created.head())

# # Merging in all data

print('all_data', 'SPD_Created', "pd.merge(all_data, all_data, on='UserID')")
all_data = pd.merge(all_data, SPD_Created, on='UserID')

print(all_data.head())

# # Importing SPD_des

SPD_des = pd.read_csv('datasets/SPD_des.csv')
print(SPD_des.head())

# # Merging in all data

print('all_data', 'SPD_des', "pd.merge(all_data, all_data, on='UserID')")

all_data = pd.merge(all_data, SPD_des, on='UserID')
print(all_data.head())

# # Importing SPD_Follower

SPD_Follower = pd.read_csv('datasets/SPD_Follower.csv')
SPD_Follower.head()

# # Merging in all data

print('all_data', 'SPD_Follower', "pd.merge(all_data, all_data, on='UserID')")

all_data = pd.merge(all_data, SPD_Follower, on='UserID')
all_data.head()

# # Importing SPD_Friends

SPD_Friends = pd.read_csv('datasets/SPD_Friends.csv')

SPD_Friends.head()

print('all_data', 'SPD_Friends', "pd.merge(all_data, all_data, on='UserID')")

all_data = pd.merge(all_data, SPD_Friends, on='UserID')
# all_data.to_csv('SPD.csv',index=False)


all_data.head()

# # Importing SPD_scr

SPD_scr = pd.read_csv('datasets/SPD_scr.csv')
SPD_scr = SPD_scr.drop(columns=['Unnamed: 2'])
SPD_scr.head()

print('all_data', 'SPD_scr', "pd.merge(all_data, all_data, on='UserID')")

all_data = pd.merge(all_data, SPD_scr, on='UserID')
all_data.head()

# # Importing SPD_Status

SPD_Status = pd.read_csv('datasets/SPD_Status.csv')
SPD_Status.head()

print('all_data', 'SPD_Status', "pd.merge(all_data, all_data, on='UserID')")

all_data = pd.merge(all_data, SPD_Status, on='UserID')
all_data.head()


# # Extracting date of tweet from Date_time feature
all_data['date_time'] = all_data['date_time'].str.extract('(\d\d\d\d)', expand=True)
print(all_data['date_time'])

all_data = all_data.dropna()
all_data.head()


# # By Calculatig Entropy

import math
import nltk
def entropy(labels):
    freqdist = nltk.FreqDist(labels)
    probs = [freqdist.freq(l) for l in freqdist]
    return -sum(p * math.log(p,2) for p in probs)


all_data['Entropy'] = all_data['Tweet_text'].map(entropy)

print(all_data)


# # Count of each User

count_user = all_data['UserID'].value_counts()
print(count_user)
# all_data.to_csv('abf.csv',index=False)
# # One user per row

all_data_required = all_data
all_data_required = all_data_required.groupby(by=['UserID'], as_index=False).first()
print(all_data_required)

all_data_required = all_data_required.drop(columns=['Tweet_text', 'date_time', 'UserID'])
all_data_required

# # Correlation Heatmap

import seaborn as sns 
import matplotlib.pyplot as plt
# plt.figure(figsize = (18,18))
# sns.heatmap(data = all_data_required.corr(), annot=True, linewidths=.3, fmt="1.2f")
# lt.show()


# # pairplot for all features
# import seaborn as sns
# sns.pairplot(all_data,hue="Class")

all_data_required.describe()

# # Graphs of Features Distribution
# all_data_required.to_csv('datasets/all_data_required.csv')

# # Class Distribution 

import seaborn as sns
import matplotlib.pyplot as plt
data = all_data_required
sns.countplot(data= data, x = "Class")
plt.show()
data.loc[:,"Class"].value_counts()
print(all_data_required)


# # By getting Text features and Class

Class_lable = all_data_required.Class
Text_Features_train = all_data_required.drop(columns=['Class'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Text_Features_train, Class_lable, test_size=0.30, random_state=700)

# # RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier
Ran_For_uni = RandomForestClassifier(n_estimators=100,max_depth=30, random_state=9,max_leaf_nodes=30)
Ran_For_uni = Ran_For_uni.fit(X_train, y_train)
Ran_For_uni

y_pred1 = Ran_For_uni.predict(X_test)
print('Random Forest= {:.2f}'.format(Ran_For_uni.score(X_test, y_test)))

# # Precision, Recall, F1

from sklearn.metrics import classification_report, confusion_matrix


print('\n')
print("Precision, Recall, F1")
print('\n')
CR=classification_report(y_test, y_pred1)
print(CR)
print('\n')

# # ROC CURVE

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_pred1)

roc_auc = auc(fpr, tpr)

plt.figure()

plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC CURVE')

plt.legend(loc="lower right")

plt.show()

# # GradientBoostingClassifier

from sklearn.ensemble import GradientBoostingClassifier

Gr_uni = GradientBoostingClassifier()
Gr_uni = Gr_uni.fit(X_train , y_train)
Gr_uni

y_pred1 = Gr_uni.predict(X_test)
print('Gradient boosting= {:.2f}'.format(Gr_uni.score(X_test, y_test)))

# # Precision, Recall, F1

from sklearn.metrics import classification_report, confusion_matrix
print('\n')
print("Precision, Recall, F1")
print('\n')
CR=classification_report(y_test, y_pred1)
print(CR)
print('\n')


# # ROC Curve

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_pred1)

roc_auc = auc(fpr, tpr)


plt.figure()

plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC CURVE')

plt.legend(loc="lower right")

plt.show()

# # ExtraTreesClassifier

from sklearn.ensemble import ExtraTreesClassifier
extra_tree = ExtraTreesClassifier(n_estimators=150)
Ran_For_uni = extra_tree.fit(X_train , y_train)
extra_tree


y_pred1 = extra_tree.predict(X_test)
print('ExtraTree= {:.2f}'.format(extra_tree.score(X_test, y_test)))

# # Precision, Recall, F1
from sklearn.metrics import classification_report, confusion_matrix

print('\n')
print("Precision, Recall, F1")
print('\n')
CR=classification_report(y_test, y_pred1)
print(CR)
print('\n')


# # ROC CURVE
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_pred1)

roc_auc = auc(fpr, tpr)


plt.figure()

plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC CURVE')

plt.legend(loc="lower right")

plt.show()

# # Comparison of Results

from prettytable import PrettyTable
x = PrettyTable()
print('\n')
print("Deatiled Performance of the all models")
x.field_names = ["Model", "Accuracy"]

x.add_row(["RandomForestClassifier", 0.92])
x.add_row(["GradientBoostingClassifier", 0.93])
x.add_row(["ExtraTreesClassifier", 0.86])
print(x)
print('\n')

x = PrettyTable()
print('\n')
print("Best Model.")
x.field_names = ["Model", "Accuracy"]
x.add_row(["GradientBoostingClassifier",0.93])
print(x)
print('\n')

# # After Data balancing 

from imblearn.over_sampling import SMOTE

smt = SMOTE()
Tf_X, Tf_y = smt.fit_sample(Text_Features_train, Class_lable)

import numpy as np
aa=np.bincount(Tf_y)
aa

sns.countplot(data= data, x = Tf_y)
plt.show()
data.iloc[Tf_y].head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Tf_X, Tf_y, test_size=0.30, random_state=700)

# # RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier
Ran_For_uni = RandomForestClassifier(n_estimators=100,max_depth=30, random_state=9,max_leaf_nodes=130)
Ran_For_uni = Ran_For_uni.fit(X_train , y_train)
Ran_For_uni


y_pred1 = Ran_For_uni.predict(X_test)
print('Accuracy score= {:.2f}'.format(Ran_For_uni.score(X_test, y_test)))

# # Precision, Recall, F1

from sklearn.metrics import classification_report, confusion_matrix


print('\n')
print("Precision, Recall, F1")
print('\n')
CR=classification_report(y_test, y_pred1)
print(CR)
print('\n')


# # ROC CURVE

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_pred1)

roc_auc = auc(fpr, tpr)


plt.figure()

plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC CURVE')

plt.legend(loc="lower right")

plt.show()


# # GradientBoostingClassifier

from sklearn.ensemble import GradientBoostingClassifier

Gr_uni = GradientBoostingClassifier()
Gr_uni = Gr_uni.fit(X_train, y_train)
Gr_uni

y_pred1 = Gr_uni.predict(X_test)
print('Accuracy score= {:.2f}'.format(Gr_uni.score(X_test, y_test)))


# # Precision, Recall, F1

from sklearn.metrics import classification_report, confusion_matrix


print('\n')
print("Precision, Recall, F1")
print('\n')
CR=classification_report(y_test, y_pred1)
print(CR)
print('\n')


# # ROC Curve

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_pred1)

roc_auc = auc(fpr, tpr)


plt.figure()

plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC CURVE')

plt.legend(loc="lower right")

plt.show()


# # ExtraTreesClassifier

from sklearn.ensemble import ExtraTreesClassifier
extra_tree = ExtraTreesClassifier(n_estimators=150)
Ran_For_uni = extra_tree.fit(X_train , y_train)
extra_tree

y_pred1 = extra_tree.predict(X_test)
print('Accuracy score= {:.2f}'.format(extra_tree.score(X_test, y_test)))


# # Precision, Recall, F1

from sklearn.metrics import classification_report, confusion_matrix


print('\n')
print("Precision, Recall, F1")
print('\n')
CR=classification_report(y_test, y_pred1)
print(CR)
print('\n')


# # ROC CURVE

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_pred1)

roc_auc = auc(fpr, tpr)


plt.figure()

plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC CURVE')

plt.legend(loc="lower right")

plt.show()


# # Comparison of Results

from prettytable import PrettyTable
x = PrettyTable()
print('\n')
print("Deatiled Performance of the all models")
x.field_names = ["Model", "Accuracy"]


x.add_row(["RandomForestClassifier", 0.93])
x.add_row(["GradientBoostingClassifier",0.94])
x.add_row(["ExtraTreesClassifier", 0.92])
print(x)
print('\n')


x = PrettyTable()
print('\n')
print("Best Model.")
x.field_names = ["Model", "Accuracy"]
x.add_row(["GradientBoostingClassifier",0.94])
print(x)
print('\n')
