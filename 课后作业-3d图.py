import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


def forward(x, b):
    return x * w + b


def loss(x, y,b):
    y_pred = forward(x, b)
    return (y_pred - y) ** 2


# 穷举法
w_list = []
mse_list = []
b_list = []
for w in np.arange(0.0, 4.1, 0.1):
    for b in np.arange(-2.0, 2.1, 0.1):
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(x_val, b)
            loss_val = loss(x_val, y_val,b)
            l_sum += loss_val
        mse_list.append(l_sum / 3)
        # print(mse_list)
    w_list.append(w)


fig = plt.figure()
ax = Axes3D(fig)

x = np.array(w_list)
y = np.arange(-2.0, 2.1, 0.1)
print(x.shape)
print(y.shape)
z = np.reshape(mse_list,(x.__len__(), y.__len__()))
x, y = np.meshgrid(x, y)
# print(z)

ax.plot_surface(x, y, z)
plt.show()
