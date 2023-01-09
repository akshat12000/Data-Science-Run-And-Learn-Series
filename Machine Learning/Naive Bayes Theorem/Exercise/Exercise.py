import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_wine
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline


# Read the data
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

print(df)

# Split the data into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(df.drop('target', axis='columns'), df.target, test_size=0.25)

# Create a model
model = GaussianNB()
model.fit(X_train, Y_train)

# Test the model
print(model.score(X_test, Y_test))

# Create a model
clf = MultinomialNB()
clf.fit(X_train, Y_train)
print(clf.score(X_test, Y_test))


