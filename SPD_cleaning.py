import numpy as np
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas
import seaborn as sns 
import matplotlib.pyplot as plt

# # legitimate_users_tweets

# # Loading dataset

data = pd.read_csv('datasets/SPD_NEW_Tweets.csv') 
print(data)

# # Dataset analysis

print(data[data.isnull().any(axis=1)].head())


import numpy as np
np.sum(data.isnull().any(axis=1))

data.isnull().any(axis=0)

print(data.info())
print(data.describe())

neg = data
neg_string = []
for t in neg:
    neg_string.append(t)
neg_string = pandas.Series(neg_string).str.cat(sep=' ')


wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(neg_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

data['word_count'] = data['Tweet_text'].apply(lambda x: len(str(x).split(" ")))
data[['Tweet_text','word_count']].head()

data['TweetLen'] = data['Tweet_text'].str.len() ## this also includes spaces
data[['Tweet_text','TweetLen']].head()

# # Cleaning dataset

# # Data cleaning script

import re

pat_1 = r"(?:\@|https?\://)\S+"
pat_2 = r'#\w+ ?'
part_3= r'br'
combined_pat = r'|'.join((pat_1, pat_2,part_3))
www_pat = r'www.[^ ]+'
html_tag = r'<[^>]+>'
negations_ = {"isn't":"is not", "can't":"can not","couldn't":"could not", "hasn't":"has not",
                "hadn't":"had not","won't":"will not",
                "wouldn't":"would not","aren't":"are not",
                "haven't":"have not", "doesn't":"does not","didn't":"did not",
                 "don't":"do not","shouldn't":"should not","wasn't":"was not", "weren't":"were not",
                "mightn't":"might not",
                "mustn't":"must not"}
negation_pattern = re.compile(r'\b(' + '|'.join(negations_.keys()) + r')\b')
from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()
def data_cleaner(text):
    try:
        stripped = re.sub(combined_pat, '', text)
        stripped = re.sub(www_pat, '', stripped)
        cleantags = re.sub(html_tag, '', stripped)
        lower_case = cleantags.lower()
        neg_handled = negation_pattern.sub(lambda x: negations_[x.group()], lower_case)
        letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
        tokens = tokenizer.tokenize(letters_only)
        return (" ".join(tokens)).strip()
    except:
        return 'NC'

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
def post_process(data, n=1048575):
    data = data.head(n)
    data['Tweet_text'] = data['Tweet_text'].progress_map(data_cleaner)  
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data

data = post_process(data)


print(data['Tweet_text'])

for letter in '@â€”.¦!)(':
    data['Tweet_text']= data['Tweet_text'].str.replace(letter,'')
data.Tweet_text.head()


data['word_count'] = data['Tweet_text'].apply(lambda x: len(str(x).split(" ")))
data[['Tweet_text','word_count']].head()


data['TweetLen'] = data['Tweet_text'].str.len() ## this also includes spaces
data[['Tweet_text','TweetLen']].head()
print(data.head(20))

aa=data.ID.unique()
print(len(aa))

# data.to_csv("datasets/Clean_SPD_NEW_Tweets.csv", index=False)
data.head()

import pandas as pd
data = pd.read_csv('datasets/Clean_SPD_NEW_Tweets.csv') 
data.head()

import pandas as pd
data1 = pd.read_csv('datasets/arehuslagenheter.csv')
data1.head()

bb=data1.ID.unique()
print(len(bb))

print('data', 'data1', "pd.merge(data, data, on='UserID')")
assigning_class =pd.merge(data, data1, on='ID')

import pandas
assigning_class.rename(columns={'Unnamed: 1': 'Class'}, inplace=True)
# assigning_class.to_csv("datasets/Clean_SPD_NEW_Tweets_with_class.csv", index=False)
assigning_class.head(4)

