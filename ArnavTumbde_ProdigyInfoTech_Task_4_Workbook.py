#!/usr/bin/env python
# coding: utf-8

# ## Arnav Tumbde
# ## Domain : Data Science
# ## Task : 4

# In[3]:


import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
import string
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


import nltk
nltk.download('stopwords')


# In[6]:


data = pd.read_csv("D:\\Semester - IV\\Prodigy Internship\\Task 4\\archive (1)\\twitter_training.csv")
v_data = pd.read_csv("D:\\Semester - IV\\Prodigy Internship\\Task 4\\archive (1)\\twitter_validation.csv")


# In[7]:


data


# In[8]:


v_data


# In[9]:


data.columns = ['id', 'game', 'sentiment', 'text']
v_data.columns = ['id', 'game', 'sentiment', 'text']


# In[10]:


data


# In[11]:


v_data


# In[12]:


data.shape


# In[13]:


data.columns


# In[14]:


data.describe(include='all')


# In[15]:


id_types = data['id'].value_counts()
id_types


# In[16]:


plt.figure(figsize=(12,7))
sns.barplot(y=id_types.index, x=id_types.values)  
plt.xlabel('Type')
plt.ylabel('Count')
plt.title('# of Id vs Count')
plt.show()


# In[17]:


game_types = data['game'].value_counts()
game_types


# In[18]:


plt.figure(figsize=(14,10))

sns.barplot(x=game_types.values,y=game_types.index)  
plt.title('# of Games and their count')
plt.ylabel('Type')
plt.xlabel('Count')

plt.show()


# In[19]:


sns.catplot(x="game",hue="sentiment", kind="count",height=10, aspect=3, data=data)


# In[20]:


sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[21]:


total_null=data.isnull().sum().sort_values(ascending=False)
percent = ((data.isnull().sum()/data.isnull().count())*100).sort_values(ascending = False)
print("Total records = ", data.shape[0])
missing_data = pd.concat([total_null,percent.round(2)],axis=1,keys=['Total Missing','In Percent'])
missing_data.head(10)


# In[22]:


data.dropna(subset=['text'],inplace=True)

total_null=data.isnull().sum().sort_values(ascending=False)
percent = ((data.isnull().sum()/data.isnull().count())*100).sort_values(ascending = False)
print("Total records = ", data.shape[0])
missing_data = pd.concat([total_null,percent.round(2)],axis=1,keys=['Total Missing','In Percent'])
missing_data.head(10)


# In[23]:


train0=data[data['sentiment']=="Negative"]
train1=data[data['sentiment']=="Positive"]
train2=data[data['sentiment']=="Irrelevant"]
train3=data[data['sentiment']=="Neutral"]


# In[24]:


train0.shape, train1.shape, train2.shape, train3.shape


# In[25]:


train0=train0[:int(train0.shape[0]/12)]
train1=train1[:int(train1.shape[0]/12)]
train2=train2[:int(train2.shape[0]/12)]
train3=train3[:int(train3.shape[0]/12)]


# In[26]:


train0.shape, train1.shape, train2.shape, train3.shape


# In[27]:


data=pd.concat([train0,train1,train2,train3],axis=0)
data


# In[28]:


id_types = data['id'].value_counts()
id_types


# In[29]:


plt.figure(figsize=(12,7))
sns.barplot(x=id_types.values,y=id_types.index)

plt.xlabel('Type')
plt.ylabel('Count')
plt.title('# of TV shows vs Movies')
plt.show()


# In[30]:


game_types = data['game'].value_counts()
game_types


# In[31]:


plt.figure(figsize=(12,7))
sns.barplot(x=game_types.values,y=game_types.index)

plt.xlabel('Type')
plt.ylabel('Count')
plt.title('# of TV shows vs Movies')
plt.show()


# In[32]:


sentiment_types = data['sentiment'].value_counts()
sentiment_types


# In[33]:


plt.figure(figsize=(12,7))
plt.pie(x=sentiment_types.values, labels=sentiment_types.index, autopct='%.3f%%', explode=[0.1, 0.1,0,0])
plt.title('The Difference in the Type of Contents')
plt.show()


# In[34]:


sns.catplot(x='game',hue='sentiment',kind='count',height=7,aspect=2,data=data)


# In[35]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
data['sentiment']=label_encoder.fit_transform(data['sentiment'])
data['game']=label_encoder.fit_transform(data['game'])
v_data['sentiment']=label_encoder.fit_transform(v_data['sentiment'])
v_data['game']=label_encoder.fit_transform(v_data['game'])


# In[36]:


data = data.drop(['id'],axis=1)

data


# In[37]:


data.nunique()


# In[38]:


v_data.nunique()


# In[ ]:





# In[ ]:




