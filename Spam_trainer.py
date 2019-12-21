import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

headers = ['label', 'sms_message']
df = pd.read_csv ('spam.csv', names = headers)
df ['label'] = df['label'].map({'ham': 0, 'spam': 1})
df ["sms_message"]= df["sms_message"].str.lower().str.replace('[^\w\s]','')

count_vector = CountVectorizer()
y = count_vector.fit_transform(df['sms_message'])
doc_array = y.toarray()

fm = pd.DataFrame(doc_array, columns = count_vector.get_feature_names())
fm

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], df['label'], random_state=1)

count_vector = CountVectorizer()
# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)
# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)
# Instantiate our model
naive_bayes = MultinomialNB()
# Fit our model to the training data
naive_bayes.fit(training_data, y_train)

# Predict on the test data
predictions = naive_bayes.predict(testing_data)

# Score our model
print ('Naive Bayes')
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))



from sklearn.ensemble import BaggingClassifier
# Instantiate a BaggingClassifier with:
# 200 weak learners (n_estimators) and everything else as default values
model_BC = BaggingClassifier(n_estimators=200)
# Fit your BaggingClassifier to the training data
model_BC.fit(training_data, y_train)
# Predict using BaggingClassifier on the test data
y_pred_BC = model_BC.predict(testing_data)
# Score our model
print ('Bagging:')
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))


# In[80]:


from sklearn.ensemble import RandomForestClassifier
# Instantiate a RandomForestClassifier with:
# 200 weak learners (n_estimators) and everything else as default values
model_RF = RandomForestClassifier(n_estimators=200)
# Fit your RandomForestClassifier to the training data
model_RF.fit(training_data, y_train)
# Predict using AdaBoostClassifier on the test data
y_pred_RF = model_RF.predict(testing_data)
print ('RandomForest:')
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))


# In[81]:


from sklearn.ensemble import AdaBoostClassifier
# Instantiate an a AdaBoostClassifier with:
# With 300 weak learners (n_estimators) and a learning_rate of 0.2
model_AB = AdaBoostClassifier(n_estimators=200, learning_rate = 0.2)
# Fit your AdaBoostClassifier to the training data
model_AB.fit(training_data, y_train)
# Predict using AdaBoostClassifier on the test data
y_pred_AB = model_AB.predict(testing_data)
print ('RandomForest:')
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))


# In[ ]:
