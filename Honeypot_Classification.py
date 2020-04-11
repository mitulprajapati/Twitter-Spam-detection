# # By inserting The clean Pollutors_legitimate dataset

import pandas as pd

all_data = pd.read_csv('datasets/all_data_Pollutors_legitimate.csv')
all_data.rename(columns={'No.1': 'UserID', 'char_count': 'TweetLen'}, inplace=True)
print(all_data)

# # Importing legitimate_FollowerCount
legitimate_FollowerCount = pd.read_csv('datasets/legitimate_FollowerCount.csv')
print(legitimate_FollowerCount.head())

# # Importing the Polluter_FollowerCount

Polluter_FollowerCount = pd.read_csv('datasets/Polluter_FollowerCount.csv')
print(Polluter_FollowerCount.head())

# # Cancatenation of legitimate_FollowerCount and Polluter_FollowerCount

legitimate_Polluter_FollowerCount = pd.concat([legitimate_FollowerCount, Polluter_FollowerCount])
print(legitimate_Polluter_FollowerCount.head())

# # Merging the all_data and data3

aa = print('all_data', 'legitimate_Polluter_FollowerCount', "pd.merge(all_data, all_data, on='UserID')")

all_data = pd.merge(all_data, legitimate_Polluter_FollowerCount, on='UserID')
print(all_data.head())

# # Importing legitimate_users details

legitimate_users = pd.read_csv('datasets/legitimate_users.csv')
print(legitimate_users.head())

# # Importing content_polluters details

content_polluters = pd.read_csv('datasets/content_polluters.csv')
print(content_polluters.head())

# # Cancatenation of legitimate_users details and content_polluters details

legitimate_Polluters_users_details = pd.concat([legitimate_users, content_polluters])
print(legitimate_Polluters_users_details.head())

# # Merging the assigning_class and data3

print('all_data', 'legitimate_Polluters_users_details', "pd.merge(all_data, all_data, on='UserID')")
all_data = pd.merge(all_data, legitimate_Polluters_users_details, on='UserID')
print(all_data.head())

# # Importing legitimate_friends_counts

# legitimate_friends_counts = pd.read_csv('datasets/legitimate_friends_counts.csv')
# legitimate_friends_counts.rename(columns={'FriendsCount': 'friends_count'}, inplace=True)
# print(legitimate_friends_counts.head())

# # Importing Polluter_friends_counts
# Polluter_friends_counts = pd.read_csv('datasets/Polluter_friends_counts.csv')
# Polluter_friends_counts.head()

# # Cancatenation of legitimate_Polluters_friends_count

# legitimate_Polluters_friends_count = pd.concat([legitimate_friends_counts, Polluter_friends_counts])
# legitimate_Polluters_friends_count.head()

# # Merging the in all data

# print('all_data', 'legitimate_Polluters_friends_count', "pd.merge(all_data, all_data, on='UserID')")

# all_data = pd.merge(all_data, legitimate_Polluters_friends_count, on='UserID')
# all_data.head()

# # Importing legitimate_Location

legitimate_Location = pd.read_csv('datasets/legitimate_Location.csv')
legitimate_Location.head()

# # Importing Polluter_Location

Polluter_Location = pd.read_csv('datasets/Polluter_Location.csv')
Polluter_Location.head()

# # Cancatenation of legitimate_Polluters_location

legitimate_Polluters_location = pd.concat([legitimate_Location, Polluter_Location])
legitimate_Polluters_location = legitimate_Polluters_location.drop(columns=['Location'])
legitimate_Polluters_location.head()

# # Merging the in all data

print('all_data', 'legitimate_Polluters_location', "pd.merge(all_data, all_data, on='UserID')")

all_data = pd.merge(all_data, legitimate_Polluters_location, on='UserID')

all_data.head()

# # Importing Legitimate_statuses_counts

Legitimate_statuses_counts = pd.read_csv('datasets/Legitimate_statuses_counts.csv')
Legitimate_statuses_counts.rename(columns={'StatusesCount': 'Statuses_count'}, inplace=True)
Legitimate_statuses_counts.head()

# # Importing Polluter_statuses_counts


Polluter_statuses_counts = pd.read_csv('datasets/Polluter_statuses_counts.csv')
Polluter_statuses_counts.head()

# # Cancatenation of legitimate_Polluters_Stauses

legitimate_Polluters_Stauses = pd.concat([Legitimate_statuses_counts, Polluter_statuses_counts])
legitimate_Polluters_Stauses.head()

# # Merging the in all data

print('all_data', 'legitimate_Polluters_Stauses', "pd.merge(all_data, all_data, on='UserID')")

all_data = pd.merge(all_data, legitimate_Polluters_Stauses, on='UserID')

all_data.head()

# # Importing Legitimate_username


Legitimate_username = pd.read_csv('datasets/Legitimate_username.csv')
Legitimate_username['UserNameLen'] = Legitimate_username['Name'].apply(lambda x: len(str(x).split(" ")))
print(Legitimate_username.head())
Legitimate_username = Legitimate_username.drop(columns=['Name'])

# # Importing Polluter_statuses_counts

Polluter_UserName = pd.read_csv('datasets/Polluter_UserName.csv')
Polluter_UserName['UserNameLen'] = Polluter_UserName['UserName'].apply(lambda x: len(str(x).split(" ")))
print(Polluter_UserName.head())
Polluter_UserName = Polluter_UserName.drop(columns=['UserName'])

# # Cancatenation of legitimate_Polluters_Stauses

legitimate_Polluters_userName = pd.concat([Legitimate_username, Polluter_UserName])
legitimate_Polluters_userName.head()

# # Merging the in all data

print('all_data', 'legitimate_Polluters_userName', "pd.merge(all_data, all_data, on='UserID')")

all_data = pd.merge(all_data, legitimate_Polluters_userName, on='UserID')

print('Data',all_data.head())

# # Extracting years count from CreatedAt for Account Age feature

all_data['CreatedAt'] = all_data['CreatedAt'].str.extract('(\d\d\d\d)', expand=True)
print(all_data['CreatedAt'])

# # Extracting date of tweet from Date_time feature

all_data['Date_time'] = all_data['Date_time'].str.extract('(\d\d\d\d)', expand=True)
print(all_data['Date_time'])

# all_data.to_csv('Honeypot-Old.csv', index=False)

all_data = all_data.dropna()
print(all_data)

# print(all_data)

# # By Calculating Entropy

import math
import nltk


def entropy(labels):
    freqdist = nltk.FreqDist(labels)
    probs = [freqdist.freq(l) for l in freqdist]
    return -sum(p * math.log(p, 2) for p in probs)


all_data['Entropy'] = all_data['Tweet_text'].map(entropy)
# all_data.to_csv('Honeypot-Old1.csv', index=False)
print(all_data)

# # Count of each User

count_user = all_data['UserID'].value_counts()
print(count_user)

# # One user per row

all_data_required = all_data
all_data_required = all_data_required.groupby(by=['UserID'], as_index=False).first()
print(all_data_required)

all_data_required = all_data_required.drop(columns=['No.2','Tweet_text','CollectedAt','FollowerCount','UserID',
                                                    'CreatedAt','Statuses_count'])
# print(all_data_required)
# all_data_required.to_csv('abc.csv',index=False)
# # Correlation Heatmap

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 15))
sns.heatmap(data=all_data_required.corr(), annot=True, linewidths=.3, fmt="1.2f")
plt.show()

# # pairplot for all features

# sns.pairplot(all_data_required, hue='Class')

# all_data_required.describe()

# # Graphs of Features Distribution
all_data_required.hist(figsize=(16, 13), bins=15, color="#107009AA")
plt.title("Features Distribution")
plt.show()

# all_data_required.to_csv('datasets/all_data_required.csv')

# # Class Distribution

import seaborn as sns
import matplotlib.pyplot as plt

data = all_data_required
sns.countplot(data=data, x="Class")
plt.show()
data.loc[:, "Class"].value_counts()

# # By getting Text features and Class

Class_lable = all_data_required.Class
Text_Features_train = all_data_required.drop(columns=['Class'])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(Text_Features_train, Class_lable, test_size=0.30, random_state=700)

# # RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

Ran_For_uni = RandomForestClassifier(n_estimators=100, max_depth=30, random_state=9, max_leaf_nodes=30)
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

plt.plot(fpr, tpr, color='darkorange', lw=1, label='RandomForest (area = %0.2f)' % roc_auc)

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

plt.plot(fpr, tpr, color='darkorange', lw=1, label='GradientBoosting (area = %0.2f)' % roc_auc)

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

plt.plot(fpr, tpr, color='darkorange', lw=1, label='ExtraTrees (area = %0.2f)' % roc_auc)

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
svclassifier = SVC(kernel='linear',cache_size=2000)
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

x.add_row(["RandomForestClassifier", 0.95])
x.add_row(["GradientBoostingClassifier", 0.96])
x.add_row(["ExtraTreesClassifier", 0.95])
x.add_row(['SVM', 0.86])
print(x)
print('\n')

x = PrettyTable()
print('\n')
print("Best Model.")
x.field_names = ["Model", "Accuracy"]
x.add_row(["GradientBoostingClassifier", 0.96])
print(x)
print('\n')
