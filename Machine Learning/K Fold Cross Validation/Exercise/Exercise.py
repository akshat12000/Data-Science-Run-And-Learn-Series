from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

iris = load_iris()

# Cross Validation Score
print(cross_val_score(LogisticRegression(solver="lbfgs",class_weight="balanced",max_iter=10000), iris.data, iris.target))
print(cross_val_score(SVC(gamma="auto"), iris.data, iris.target))
print(cross_val_score(RandomForestClassifier(n_estimators=40), iris.data, iris.target))
