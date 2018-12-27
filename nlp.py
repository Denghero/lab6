
# coding: utf-8

# # Weeks 13-14
# 
# ## Natural Language Processing
# ***

# Read in some packages.

# In[2]:


# Import pandas to read in data
import numpy as np
import pandas as pd

# Import models and evaluation functions
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
from sklearn import cross_validation

# Import vectorizers to turn text into numeric
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Import plotting
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Basic Feature Engineering
# We have examined two ways of dealing with categorical (i.e. text based) data: binarizing/dummy variables and numerical scaling. 
# 
# See the following examples for implementation in sklearn to start:

# In[3]:


data = pd.read_csv("data/categorical.csv")


# In[4]:


data


# ### Binarizing
# Get a list of features you want to binarize, go through each feature and create new features for each level.

# In[5]:


features_to_binarize = ["Gender", "Marital"]

# Go through each feature
for feature in features_to_binarize:
    # Go through each level in this feature (except the last one!)
    for level in data[feature].unique()[0:-1]:
        # Create new feature for this level
        data[feature + "_" + level] = pd.Series(data[feature] == level, dtype=int)
    # Drop original feature
    data = data.drop([feature], 1)


# In[6]:


data


# ### Numeric scaling
# We can also replace text levels with some numeric mapping we create

# In[7]:


data['Satisfaction'] = data['Satisfaction'].replace(['Very Low', 'Low', 'Neutral', 'High', 'Very High'], 
                                                    [-2, -1, 0, 1, 2])


# In[8]:


data


# ## Text classification
# We are going to look at some Amazon reviews and classify them into positive or negative.

# ### Data
# The file `data/books.csv` contains 2,000 Amazon book reviews. The data set contains two features: the first column (contained in quotes) is the review text. The second column is a binary label indicating if the review is positive or negative.
# 
# Let's take a quick look at the file.

# In[9]:


get_ipython().system('head -3 data/books.csv')


# Let's read the data into a pandas data frame. You'll notice two new attributed in `pd.read_csv()` that we've never seen before. The first, `quotechar` is tell us what is being used to "encapsulate" the text fields. Since our review text is surrounding by double quotes, we let pandas know. We use a `\` since the quote is also used to surround the quote. This backslash is known as an escape character. We also let pandas now this.

# In[22]:


data = pd.read_csv("data/books.csv", quotechar="\"", escapechar="\\")


# In[23]:


data.head()


# ### Text as a set of features
# Going from text to numeric data is very easy. Let's take a look at how we can do this. We'll start by separating out our X and Y data.

# In[24]:


X_text = data['review_text']
Y = data['positive']


# In[25]:


# look at the first few lines of X_text
X_text.head()


# Do the same for Y

# In[26]:


# your code here
Y.head()


# Next, we will turn `X_text` into just `X` -- a numeric representation that we can use in our algorithms or for queries...
# 
# Text preprocessing, tokenizing and filtering of stopwords are all included in CountVectorizer, which builds a dictionary of features and transforms documents to feature vectors. 
# 
# The result of the following is a matrix with each row a file and each column a word. The matrix is sparse because most words only appear a few times. The values are 1 if a word appears in a document and 1 otherwise.

# In[27]:


# Create a vectorizer that will track text as binary features
binary_vectorizer = CountVectorizer(binary=True)

# Let the vectorizer learn what tokens exist in the text data
binary_vectorizer.fit(X_text)

# Turn these tokens into a numeric matrix
X = binary_vectorizer.transform(X_text)


# In[28]:


# Dimensions of X:
X.shape


# There are 2000 documents (each row) and 22,743 words/tokens.
# 
# Can look at some of the words by querying the binary vectorizer:

# In[29]:


# List of the 20 features (words) in column 10,000
features = binary_vectorizer.get_feature_names()
features[10000:10020]


# Spend some time to look at the binary vectoriser.
# 
# Examine the structure of X. Look at some the rows and columns values.

# In[30]:


# see the density of 0s and 1s in X
import scipy.sparse as sps
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.spy(X.toarray())
plt.show()


# Look at the sparse matrix above. Notice how some columns are quite dark (i.e. the words appear in almost every file). 
# 
# What are the 5 most common words?

# In[31]:


# your code here
value = X.toarray().sum(axis=0)
re = pd.Series(value)
re.index = features
print re.sort_values(ascending=False)[0:5]


# Your answer here

# Write a function that takes the sparse matrix X, and gets the feature list from the vectoriser, and a document index (1 - 2000) and returns a list of the words in the file that corresponds to the index (the list should be obtained from the sparse matrix / bag of words representation NOT from the original data file). 

# In[32]:


# complete the function 
# returns vector of words / features
def getWords(bag_of_words, file_index_row, features_list):
    Spare_matrix_array = bag_of_words.toarray()
    test = Spare_matrix_array[file_index_row,:]
    result = []
    for i in range (0,len(test)):
        if(test[i]==1):
            result.append(features_list[i])
    return result
getWords(X, 1, features)


# ### Modeling
# We have a 22743 features, let's use them in some different models.

# In[34]:


# Create a model
logistic_regression = LogisticRegression()

# Use this model and our data to get 5-fold cross validation accuracy
acc = cross_validation.cross_val_score(logistic_regression, X, Y, scoring="accuracy", cv=5)

# Print out the average accuracy rounded to three decimal points
print ("Mean accuracy of our classifier is " + str(round(np.mean(acc), 3)) )


# In[ ]:


Use the above classifier to classify a new example (new review below):


# In[38]:


new_review = """"
really bad book!
"""

# your code here ...
temp = new_review.split(' ')
matrix = []
for i in range(0,len(features)):
    if features[i] in temp:
        matrix.append(1)
    else:
        matrix.append(0)
logistic_regression = logistic_regression.fit(X.toarray(),Y)
# predit function needs a array with two dimensions.
predict = logistic_regression.predict([matrix])
print predict
    


# Let's try using full counts instead of a binary representation (i.e. each time a word appears use the raw count value). 

# In[40]:


# Create a vectorizer that will track text as binary features
count_vectorizer = CountVectorizer()

# Let the vectorizer learn what tokens exist in the text data
count_vectorizer.fit(X_text)

# Turn these tokens into a numeric matrix
X = count_vectorizer.transform(X_text)

# Create a model
logistic_regression = LogisticRegression()

# Use this model and our data to get 5-fold cross validation accuracy
acc = cross_validation.cross_val_score(logistic_regression, X, Y, scoring="accuracy", cv=5)

# Print out the average AUC rounded to three decimal points
print( "Accuracy for our classifier is " + str(round(np.mean(acc), 3)) )


# Now try using TF-IDF:

# In[41]:


# Create a vectorizer that will track text as binary features
tfidf_vectorizer = TfidfVectorizer()

# Let the vectorizer learn what tokens exist in the text data
tfidf_vectorizer.fit(X_text)

# Turn these tokens into a numeric matrix
X = tfidf_vectorizer.transform(X_text)

# Create a model
logistic_regression = LogisticRegression()

# Use this model and our data to get 5-fold cross validation AUCs
acc = cross_validation.cross_val_score(logistic_regression, X, Y, scoring="accuracy", cv=5)

# Print out the average AUC rounded to three decimal points
print( "Accuracy for our classifier is " + str(round(np.mean(acc), 3)) )


# Use the tfidf classifier to classify some online book reviews from here: https://www.amazon.com/
# 
# Hint: You can copy and paste a review from the online site into a multiline string literal with 3 quotes: 
# ```
# """
# copied and pasted
# multiline
# string...
# """
# ```

# In[ ]:


# your code here
sample = "I loved Reed and Charlotte Everything that happened with the both of them made them that much more real There were also some laugh out loud moments so watch out for those if you are reading in a public place or else people will look at you like you are crazy"
sample_temp = sample.split(' ')
sample_matrix = []
for i in range(0,len(features)):
    if features[i] in sample_temp:
        sample_matrix.append(1)
    else:
        sample_matrix.append(0)
logistic_regression = logistic_regression.fit(X.toarray(),Y)
# predit function needs a array with two dimensions.
sample_predict = logistic_regression.predict([sample_matrix])
print sample_predict
    


# ### Extending the implementation
# #### Features
# Tfidf is looking pretty good! How about adding n-grams? Stop words? Lowercase transforming?
# 
# We saw that the most common words include "the" and others above - start by making these stop words.
# 
# N-grams are conjunctions of words (e.g. a 2-gram adds all sequences of 2 words)
# 
# 
# Look at the docs: `CountVectorizer()` and `TfidfVectorizer()` can be modified to handle all of these things. Work in groups and try a few different combinations of these settings for anything you want: binary counts, numeric counts, tf-idf counts. Here is how you would use these settings:
# 
# - "`ngram_range=(1,2)`": would include unigrams and bigrams (ie including combinations of words in sequence)
# - "`stop_words="english"`": would use a standard set of English stop words
# - "`lowercase=False`": would turn off lowercase transformation (it is actually on by default)!
# 
# You can use some of these like this:
# 
# `tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), lowercase=False)`
# 
# #### Models
# Next swap out the line creating a logistic regression with one making a naive Bayes or support vector machines (SVM). SVM have been shown to be very effective in text classification. Naive Bayes has been used a lot also.
# 
# For example see: http://www.cs.cornell.edu/home/llee/papers/sentiment.pdf
# 

# In[ ]:


# Try different features, models, or both!
# What is the highest accuracy you can get?
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), lowercase=False)
tfidf_vectorizer.fit(X_text)
numeric_X = tfidf_vectorizer.transform(X_text)

LinearSVC  = LinearSVC()
MultinomialNB = MultinomialNB()
acc_SVC = cross_validation.cross_val_score(LinearSVC, numeric_X, Y, scoring="accuracy", cv=5)
acc_NB = cross_validation.cross_val_score(MultinomialNB, numeric_X, Y, scoring="accuracy", cv=5)

print( "Accuracy for our classifier by using SVM is " + str(round(np.mean(acc_SVC), 3)) )
print( "Accuracy for our classifier by using Native Bayes is " + str(round(np.mean(acc_NB), 3)) )

