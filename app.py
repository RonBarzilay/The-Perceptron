import numpy as np
import matplotlib.pyplot as plt

"""
This Code Should Get More Documentation 
"""


def draw(x1, x2):
    """
    :param x: x
    :param y: y
    :return: Point in graph
    """
    ln = plt.plot(x1, x2)
    plt.pause(0.0001)
    ln[0].remove()


def sigmoid(score):
    """
    This function gets score of a point and returns the percent of that point being related to group no.'1'
    Percentage = (1)/(1 + e^(1/score))
    :param score: The result of: m*x1 + b - x2 = {score}
    :return: Percentage of a scored point being related to group no.'1'
    """
    return 1 / (1 + np.exp(-score))


def calculate_error(line_parameters, points, y):
    """
    This function calculates how accurate the Linear Model from line_parameters and the sigmoid of points.
    This function uses sigmoid for calculating error.
    :param line_parameters: Checked Linear Model: w1, w2, b
    :param points: All points
    :param y: Group no. 1/0
    :return: How accurate the Linear Model is
    """
    num_of_points = len(points)
    print("number of points: ", num_of_points)

    # The percentage for each point being related to group no.'1'
    percentage = sigmoid(points * line_parameters)

    # -[1/num_of_points] * [y*ln(p1, p2, ... pn) + (1-y)ln(1-p11, 1-p22, ... 1-pnn)]
    cross_entropy = -(1 / num_of_points) * (np.log(percentage).T * y + np.log(1 - percentage).T * (1 - y))
    return cross_entropy


def gradient_descent(line_parameters, points, y, alpha):
    """
    This function aim to minimize the Linear Model Error
    :param line_parameters: Checked Linear Model: w1, w2, b
    :param points: All points
    :param y: Group no. 1/0
    :param alpha: lr=learning rate - Const multiplying for small fixes in Linear Model
    :return: More accurate Linear Model
    """
    num_of_points = len(points)
    for i in range(2000):
        p = sigmoid(points*line_parameters)
        # gradient = ((points * (P - y))*alpha)/n
        gradient = points.T*(p-y)*(alpha/num_of_points)
        # gradient - fix the Linear Model by: line_parameters - gradient
        line_parameters = line_parameters - gradient

        w1 = line_parameters.item(0)
        w2 = line_parameters.item(1)
        b = line_parameters.item(2)

        # x1, x2 ?
        x1 = np.array([points[:, 0].min(), points[:, 0].max()])
        x2 = -b/w2 + x1*(-w1/w2)
        draw(x1, x2)
        print(calculate_error(line_parameters, points, y))


# Number of points for each color/group
n_pts = 100
np.random.seed(0)

# Array of ones with length be the given n_points
# The bias is the factor of b in : m*x1 + bias*b - x2 = 0
# Which in this case the factor is 1
bias = np.ones(n_pts)

# Create top_region array of X: (center: 10, distance 2, n_points) Y: (center: 12, distance 2, n_points)
top_region = np.array([np.random.normal(10, 2, n_pts), np.random.normal(12, 2, n_pts), bias]).T
print("top region/group: ", top_region)

# Create bottom_region array of X: (center: 5, distance 2, n_points) Y: (center: 6, distance 2, n_points)
bottom_region = np.array([np.random.normal(5, 2, n_pts), np.random.normal(6, 2, n_pts), bias]).T
print("bottom region/group: ", bottom_region)

all_points = np.vstack((top_region, bottom_region))

# # The slope (m)
# w1 = -0.2
# # The intercept (b)
# w2 = -0.15
# # The bias is the factor of b: b*bias
# bias = 3

# All line_parameters in transpose matrix

line_parameters = np.matrix([np.zeros(3)]).T

# x1 = np.array([bottom_region[:, 0].min(), top_region[:, 0].max()])
# x2 = -bias / w2 + (x1 * (-w1 / w2))

y = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts * 2, 1)
_, ax = plt.subplots(figsize=(4, 4))

# np slicing: [1:4] slice idx 1 to idx 4
# np slicing: [:4] slice start idx to idx 4 (not included 0123)
# np slicing: [4:] slice idx 4 to end idx
# np slicing: [-5:-3] slice idx 5 from end to idx 3 from end
ax.scatter(top_region[:, 0], top_region[:, 1], color='r')
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color='b')
gradient_descent(line_parameters, all_points, y, 0.06)
plt.show()

