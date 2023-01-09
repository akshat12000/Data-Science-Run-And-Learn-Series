import pandas as pd
from sklearn import svm, datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score,RandomizedSearchCV

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['flower'] = iris.target
df['flower'] = df.flower.apply(lambda x: iris.target_names[x])

X = df.drop(['flower'], axis='columns')
Y = df.flower

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = svm.SVC(kernel='rbf',C=30,gamma='auto')
model.fit(X_train, Y_train)
print(model.score(X_test, Y_test))

# K-Fold Cross Validation
print(cross_val_score(svm.SVC(kernel='rbf',C=30,gamma='auto'), iris.data, iris.target, cv=5))

# Hyper Parameter Tuning
clf = GridSearchCV(svm.SVC(gamma='auto'), {
    'C': [1, 10, 20],
    'kernel': ['rbf', 'linear']
}, cv=5, return_train_score=False)
clf.fit(iris.data, iris.target)
res_df = pd.DataFrame(clf.cv_results_)
print(res_df)
print(clf.best_params_)
print(clf.best_score_)
print(clf.best_estimator_)
print(clf.best_index_)

# Randomized Search CV
rs = RandomizedSearchCV(svm.SVC(gamma='auto'), {
    'C': [1, 10, 20],
    'kernel': ['rbf', 'linear']
}, cv=5, return_train_score=False, n_iter=2)
rs.fit(iris.data, iris.target)
res_df = pd.DataFrame(rs.cv_results_)
print(res_df)
print(rs.best_params_)
print(rs.best_score_)
print(rs.best_estimator_)
print(rs.best_index_)

# Choosing Best Model
model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params': {
            'C': [1, 10, 20],
            'kernel': ['rbf', 'linear']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [1, 5, 10]
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(solver='liblinear', multi_class='auto'),
        'params': {
            'C': [1, 5, 10]
        }
    }
}

scores = []
for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(iris.data, iris.target)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })

df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
print(df)

