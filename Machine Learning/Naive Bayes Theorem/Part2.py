import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Read the data
df = pd.read_csv('spam.csv')

df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
df.drop('Category', axis='columns', inplace=True)

# Split the data into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(df.Message, df.spam, test_size=0.25)

v=CountVectorizer()
X_train_count = v.fit_transform(X_train.values)

model = MultinomialNB()
model.fit(X_train_count, Y_train)

emails = [
    'Hey mohan, can we get together to watch football game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]

emails_count = v.transform(emails)
print(model.predict(emails_count))
X_test_count = v.transform(X_test)
print(model.score(X_test_count, Y_test))

clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

clf.fit(X_train, Y_train)
print(clf.score(X_test, Y_test))
