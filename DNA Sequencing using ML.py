#!/usr/bin/env python
# coding: utf-8

# In[1]:


#load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


human_data = pd.read_table('human_data.txt')
human_data.head()


# In[3]:


chimp_data = pd.read_table('chimp_data.txt')
chimp_data.head()


# In[4]:


dog_data = pd.read_table('dog_data.txt')
dog_data.head()


# In[5]:


def Kmers_funct(seq, size=6):
    return [seq[x:x+size].lower() for x in range(len(seq) - size + 1)]


# In[6]:


#Now we can convert our training data sequences into short overlapping k-mers of legth 6.
human_data['words'] = human_data.apply(lambda x: Kmers_funct(x["sequence"]), axis=1)
human_data = human_data.drop("sequence", axis=1)


# In[7]:


chimp_data['words'] = chimp_data.apply(lambda x: Kmers_funct(x["sequence"]), axis=1)
chimp_data = chimp_data.drop("sequence", axis=1)


# In[8]:


dog_data['words'] = dog_data.apply(lambda x: Kmers_funct(x["sequence"]), axis=1)
dog_data = dog_data.drop("sequence", axis=1)


# In[9]:


human_data.head()


# In[10]:


#Since we are going to use scikit-learn natural language processing tools to do the k-mer counting,
#we need to now convert the lists of k-mers for each gene into #string sentences of words that the count vectorizer can use. 
#We can also make a y variable to hold the class labels.
human_texts = list(human_data['words'])
for item in range(len(human_texts)):
    human_texts[item] = ' '.join(human_texts[item])
#separate labels
y_human = human_data.iloc[:, 0].values         # y_human for human_dna


# In[11]:


#Now let's do the same for chimp and dog.
chimp_texts = list(chimp_data['words'])
for item in range(len(chimp_texts)):
    chimp_texts[item] = ' '.join(chimp_texts[item])
#separate labels
y_chim = chimp_data.iloc[:, 0].values # y_chim for chimp_dna


dog_texts = list(dog_data['words'])
for item in range(len(dog_texts)):
    dog_texts[item] = ' '.join(dog_texts[item])
#separate labels
y_dog = dog_data.iloc[:, 0].values  # y_dog for dog_dna


# In[12]:


# Creating the Bag of Words model using CountVectorizer()
# This is equivalent to k-mer counting
# The n-gram size of 4 was previously determined by testing
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(4,4))
X = cv.fit_transform(human_texts)
X_chimp = cv.transform(chimp_texts)
X_dog = cv.transform(dog_texts)


# In[13]:


print(X.shape)
print(X_chimp.shape)
print(X_dog.shape)


# In[14]:


#ploting class 
human_data['class'].value_counts().sort_index().plot.bar()


# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y_human,test_size = 0.20,random_state=42)
print(X_train.shape)
print(X_test.shape)


# In[16]:


# Create Naive Bayes Classifier model to predict gene functioning of other species 
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)


# In[17]:


y_pred = classifier.predict(X_test)


# In[18]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print("Confusion matrix\n")
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))

def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))


# In[19]:


#predicting chimp DNA functioning
y_pred_chimp = classifier.predict(X_chimp)


# In[20]:


print("Confusion matrix\n")
print(pd.crosstab(pd.Series(y_chim, name='Actual'), pd.Series(y_pred_chimp, name='Predicted')))
accuracy, precision, recall, f1 = get_metrics(y_chim, y_pred_chimp)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))


# In[21]:


# Now same for dog's functioning
y_pred_dog = classifier.predict(X_dog)
print("Confusion matrix\n")
print(pd.crosstab(pd.Series(y_dog, name='Actual'), pd.Series(y_pred_dog, name='Predicted')))
accuracy, precision, recall, f1 = get_metrics(y_dog, y_pred_dog)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))


# In[22]:


#Conclusion:
# Here, we compared the DNA functioning of humans, dogs & chimpanzees; for their similarities in genes. 
# As a result, we got to know that humans & chimps share same DNA ancestors of about 98% similarities whereas humans & dogs doesn't have such higher similarities in functioning.

