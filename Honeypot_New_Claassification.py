import pandas as pd
import numpy as np

all_data = pd.read_csv('datasets/New_all_data_Pollutors_legitimate.csv')
all_data.rename(columns={'No.1': 'UserID', 'char_count': 'TweetLen'}, inplace=True)
print(all_data)


legitimate_new = pd.read_csv('datasets/Legitimate_New.csv')
print(legitimate_new.head())

polluter_new = pd.read_csv('datasets/Polluter_New12.csv')
print(polluter_new)

legitimate_Polluter_new = pd.concat([legitimate_new, polluter_new], sort=False)
print(legitimate_Polluter_new.head())

all_data = pd.merge(all_data, legitimate_Polluter_new, on='UserID')
print(all_data)
# all_data.to_csv('abcd2.csv',index=False)


Legitimate_created = pd.read_csv('datasets/legitimate_users.csv', usecols=['CreatedAt','UserID'])
# print(Legitimate_created)

Polluter_created = pd.read_csv('datasets/content_polluters.csv', usecols=['CreatedAt','UserID'])
# print(Polluter_created)

legitimate_polluter_created = pd.concat([Legitimate_created,Polluter_created],sort=False)
# print(legitimate_polluter_created)

all_data = pd.merge(all_data, legitimate_polluter_created, on='UserID')
# print(all_data)

# all_data['Followership'] = (all_data['FollowerCount']/all_data['FriendsCount'])

# all_data = pd.merge(all_data, all_data['Followership'], on='UserID')
# print(all_data)


all_data['CreatedAt'] = all_data['CreatedAt'].str.extract('(\d\d\d\d)', expand=True)
# print(all_data['CreatedAt'])

all_data['Date_time'] = all_data['Date_time'].str.extract('(\d\d\d\d)', expand=True)
# print(all_data['Date_time'])
# print(all_data)
# all_data.to_csv('abc11.csv', index=False)


all_data = all_data.dropna()
# print(all_data)

# # By Calculatig Entropy

import math
import nltk


def entropy(labels):
    freqdist = nltk.FreqDist(labels)
    probs = [freqdist.freq(l) for l in freqdist]
    return -sum(p * math.log(p, 2) for p in probs)


all_data['Entropy'] = all_data['Tweet_text'].map(entropy)
# all_data.to_csv('Honeypot-New.csv', index=False)
# print(all_data)

# # Count of each User

count_user = all_data['UserID'].value_counts()
print(count_user)

all_data = all_data.drop(columns=['UserID','Tweet_text'])
# # Correlation Heatmap

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 15))
sns.heatmap(data=all_data.corr(), annot=True, linewidths=.3, fmt="1.2f")
plt.show()

print(all_data.describe())

# # Class Distribution

import seaborn as sns
import matplotlib.pyplot as plt

data = all_data
sns.countplot(data=data, x="Class")
plt.show()
data.loc[:, "Class"].value_counts()


Class_lable = all_data.Class
Text_Features_train = all_data.drop(columns=['Class'])
# print(all_data)

# all_data.to_csv('reg.csv',index=False)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(Text_Features_train, Class_lable, test_size=0.30, random_state=700)

# # RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

Ran_For_uni = RandomForestClassifier(n_estimators=200, max_depth=30, random_state=9, max_leaf_nodes=30)
Ran_For_uni = Ran_For_uni.fit(X_train, y_train)
print(Ran_For_uni)

y_pred1 = Ran_For_uni.predict(X_test)
print('Random Forest= {:.2f}'.format(Ran_For_uni.score(X_test, y_test)))

# # Precision, Recall, F1

from sklearn.metrics import classification_report

print('\n')
print("Precision, Recall, F1")
print('\n')
CR = classification_report(y_test, y_pred1)
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
print('Gradient boosting = {:.2f}'.format(Gr_uni.score(X_test, y_test)))

# # Precision, Recall, F1

from sklearn.metrics import classification_report

print('\n')
print("Precision, Recall, F1")
print('\n')
CR = classification_report(y_test, y_pred1)
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
Ran_For_uni = extra_tree.fit(X_train, y_train)
extra_tree

y_pred1 = extra_tree.predict(X_test)
print('ExtraTree = {:.2f}'.format(extra_tree.score(X_test, y_test)))

# # Precision, Recall, F1

from sklearn.metrics import classification_report

print('\n')
print("Precision, Recall, F1")
print('\n')
CR = classification_report(y_test, y_pred1)
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


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Text_Features_train, Class_lable , test_size=0.3,
                                                    stratify=Class_lable, random_state=1)

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear',cache_size=4000)
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print('SVM= {:.2f}'.format(svclassifier.score(X_test, y_test)))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_pred)

roc_auc = auc(fpr, tpr)

plt.figure()

plt.plot(fpr, tpr, color='darkorange', lw=3, label='SVM (area = %0.2f)' % roc_auc)

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

x.add_row(["RandomForestClassifier", 0.88])
x.add_row(["GradientBoostingClassifier", 0.88])
x.add_row(["ExtraTreesClassifier", 0.98])
print(x)
print('\n')


