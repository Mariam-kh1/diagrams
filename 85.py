from sklearn.tree import DecisionTreeClassifier
X_train = [[1], [2], [3], [4], [5], [6], [7]]
y_train = [0, 0, 0, 1, 1, 1, 0]
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
X_new = [[3], [5.5], [7]]
y_new = [0, 1, 1]
predictions = clf.predict(X_new)
print("New data:", X_new)
print("Predictions:", predictions)
print("Actual labels:", y_new)
print("Accuracy on new data:", clf.score (X_new, y_new))

from sklearn import tree
import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
tree.plot_tree(clf, feature_names=["Hours"], class_names=["Fail", "Pass"], filled=True)
plt.show()