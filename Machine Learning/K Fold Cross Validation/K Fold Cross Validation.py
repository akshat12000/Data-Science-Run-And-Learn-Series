from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

digits = load_digits()

X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, test_size=0.3)

# K Fold Cross Validation
kf = KFold(n_splits=3)
for train_index, test_index in kf.split([1,2,3,4,5,6,7,8,9]):
    print(train_index, test_index)

def get_score(model, X_train, X_test, Y_train, Y_test):
    model.fit(X_train, Y_train)
    return model.score(X_test, Y_test)

folds = StratifiedKFold(n_splits=3)

scores_l = []
scores_svm = []
scores_rf = []

for train_index, test_index in folds.split(digits.data, digits.target):
    X_train, X_test, Y_train, Y_test = digits.data[train_index], digits.data[test_index], digits.target[train_index], digits.target[test_index]
    scores_l.append(get_score(LogisticRegression(solver="lbfgs",class_weight="balanced",max_iter=10000), X_train, X_test, Y_train, Y_test))
    scores_svm.append(get_score(SVC(gamma="auto"), X_train, X_test, Y_train, Y_test))
    scores_rf.append(get_score(RandomForestClassifier(n_estimators=40), X_train, X_test, Y_train, Y_test))

print(scores_l)
print(scores_svm)
print(scores_rf)

# Cross Validation Score
print(cross_val_score(LogisticRegression(solver="lbfgs",class_weight="balanced",max_iter=10000), digits.data, digits.target))
print(cross_val_score(SVC(gamma="auto"), digits.data, digits.target))
print(cross_val_score(RandomForestClassifier(n_estimators=40), digits.data, digits.target))





