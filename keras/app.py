import numpy as np
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


def draw(x1, x2):
    """
    :param x: x
    :param y: y
    :return: Point in graph
    """
    ln = plt.plot(x1, x2)


def plot_decision_boundary(X, y, model):
    """
    This function will predict by model.predict the Sigmoid for each point
    in graph.
    :param X: All points
    :param y: Labels
    :param model: The model
    :return:
    """
    x_span = np.linspace(min(X[:,0]) - 1, max(X[:,0]) + 1)
    y_span = np.linspace(min(X[:,1]) - 1, max(X[:,1]) + 1)
    xx, yy = np.meshgrid(x_span, y_span)
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    pred_func = model.predict(grid)
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx, yy, z)


n_pts = 500
np.random.seed(0)
# Xa - All points for 0 group
Xa = np.array([np.random.normal(13, 2, n_pts),
               np.random.normal(12, 2, n_pts)]).T
# Xb - All points for 1 group
Xb = np.array([np.random.normal(8, 2, n_pts),
               np.random.normal(6, 2, n_pts)]).T

# Stack arrays in vertically (row wise).
# Xa1
# Xa2...
# Xb1
# Xb2...
X = np.vstack((Xa, Xb))

# Let's label 2 groups:
# top_region - label zero: np.zeros(n_pts)
# bottom_region - label one: np.ones(n_pts)
Y = np.matrix(np.append(np.zeros(n_pts), np.ones(n_pts))).T

plt.scatter(X[:n_pts, 0], X[:n_pts, 1])
plt.scatter(X[n_pts:, 0], X[n_pts:, 1])

model = Sequential()
model.add(Dense(units=1, input_shape=(2,), activation='sigmoid'))
# lr=learning rate
adam = Adam(lr=0.1)
# binary_crossentropy(loss): Accuracy of our Linear Model
model.compile(adam, loss='binary_crossentropy', metrics=['accuracy'])
h = model.fit(x=X, y=Y, verbose=1, batch_size=50, epochs=100, shuffle='true')
plt.plot(h.history['loss'])
plt.title("loss")
plt.xlabel("epoch")
plt.legend(["loss"])
# plt.show()

'''
Now, what's left is to get the Linear Model of loss: 0.0676 - accuracy: 0.9770 which is best Model got in 100 loops
'''
plot_decision_boundary(X, y, model)
plt.scatter(X[:n_pts, 0], X[:n_pts, 1])
plt.scatter(X[n_pts:, 0], X[n_pts:, 1])
x = 7.5
y = 5

point = np.array([[x, y]])
prediction = model.predict(point)
plt.plot([x], [y], marker='o', markersize=10, color="red")
print("prediction is: ", prediction)
plt.show()
