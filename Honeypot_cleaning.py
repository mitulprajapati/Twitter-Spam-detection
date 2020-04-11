import numpy as np
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas
import seaborn as sns 
import matplotlib.pyplot as plt

# # content_polluters_tweets

# # Loading datasets

data = pd.read_excel (r'datasets/content_polluters_tweets.xlsx') 
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

data['char_count'] = data['Tweet_text'].str.len() ## this also includes spaces
data[['Tweet_text','char_count']].head()

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
print(data.Tweet_text.head(20))


for letter in 'Â':
    data['Tweet_text']= data['Tweet_text'].str.replace(letter,'')

print(data.Tweet_text.head(20))
# # After Cleaning data


data['word_count'] = data['Tweet_text'].apply(lambda x: len(str(x).split(" ")))
data[['Tweet_text','word_count']].head(15)

data['char_count'] = data['Tweet_text'].str.len() ## this also includes spaces
data[['Tweet_text','char_count']].head(30)

print(data.head(20))

import pandas
content_polluters_tweets=data
content_polluters_tweets['Class']='1'
# content_polluters_tweets.to_csv("datasets/Clean_content_polluters_tweets.csv", index=False)
content_polluters_tweets.head()

# # legitimate_users_tweets

# # Loading dataset
data = pd.read_excel (r'datasets/legitimate_users_tweets.xlsx')
print(data.head())

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

data['char_count'] = data['Tweet_text'].str.len() ## this also includes spaces
data[['Tweet_text','char_count']].head()

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
print(data.Tweet_text.head())

for letter in 'Â':
    data['Tweet_text']= data['Tweet_text'].str.replace(letter,'')
print(data.Tweet_text.head())

# # After Cleaning data

data['word_count'] = data['Tweet_text'].apply(lambda x: len(str(x).split(" ")))
data[['Tweet_text','word_count']].head()

data['char_count'] = data['Tweet_text'].str.len() ## this also includes spaces
data[['Tweet_text','char_count']].head()
print(data.head(20))

import pandas
legitimate_users_tweets=data
legitimate_users_tweets['Class'] = '0'
# legitimate_users_tweets.to_csv("datasets/Clean_legitimate_users_tweets.csv", index=False)
legitimate_users_tweets.head()


all_data = pd.concat([content_polluters_tweets, legitimate_users_tweets])
# all_data.to_csv("datasets/all_data_Pollutors_legitimate.csv", index=False)
print(all_data)

