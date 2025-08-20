from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5]] 
y = [40, 50, 60, 70, 80]
model = LinearRegression()
model.fit(X, y)
prediction = model.predict([[6]])
print("Predicted score for 6 hours studied:", prediction[0])

print("R^2 score: ", model.score(X, y))

import matplotlib.pyplot as plt

plt.scatter(X, y, color="blue", label="Data points")
plt.plot(X,model.predict(X), color="red", label="Regression line")
plt.scatter(6, prediction, color="green", s=100, marker="*", label="Prediction (6h)")

plt.xlabel("Hours studied")
plt.ylabel("Exam score")
plt.legend()
plt.show()