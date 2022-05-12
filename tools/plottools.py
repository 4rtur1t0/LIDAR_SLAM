import matplotlib.pyplot as plt
import numpy as np

def plot(x, title='Untitled', block=True):
    plt.figure()
    x = np.array(x)
    plt.plot(x)
    plt.title(title)
    plt.show(block=block)


def plot_vars(q_path, title='UNTITLED'):
    plt.figure()
    q_path = np.array(q_path)
    sh = q_path.shape
    for i in range(0, sh[1]):
        plt.plot(q_path[:, i], label='q' + str(i + 1), linewidth=4)
    plt.legend()
    plt.title(title)
    plt.show(block=True)


def plot_xy(x, y, title='UNTITLED'):
    plt.figure()
    x = np.array(x)
    y = np.array(y)
    x = x.flatten()
    y = y.flatten()
    plt.plot(x, y, linewidth=4, marker='.')
    # plt.xlabel('qi')
    # plt.ylabel('dw')
    plt.legend()
    plt.title(title)
    plt.show(block=True)


def plot3d(x, y, z, title='3D'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.xlim([0, 0.5])
    plt.title(title)
    plt.show(block=True)


def plot_path(odo_icp, odo_gt, title='UNTITLED'):
    plt.figure()
    plt.plot(odo_icp[:, 0], odo_icp[:, 1], color='blue', linestyle='dashed', marker='o', markersize=12)
    plt.plot(odo_gt[:, 0], odo_gt[:, 1], color='red', marker='o', markerfacecolor='blue', markersize=12)
    plt.legend()
    plt.title(title)
    plt.show(block=True)


def plot_state(x_gt, odo, title='UNTITLED'):
    plt.figure()
    plt.plot(odo[:, 0], odo[:, 1], color='blue', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=12)
    plt.plot(x_gt[:, 0], x_gt[:, 1], color='red', marker='o', markerfacecolor='red', markersize=12)
    plt.title(title)
    plt.show(block=False)


def plot_initial(x_gt, odo, edges, title='UNTITLED'):
    plt.figure()
    plt.plot(x_gt[:, 0], x_gt[:, 1], color='red', marker='o', markerfacecolor='red', markersize=12)
    plt.plot(odo[:, 0], odo[:, 1], color='blue', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=12)
    for edge in edges:
        i = int(edge[0])
        j = int(edge[1])
        x = [odo[i, 0], odo[j, 0]]
        y = [odo[i, 1], odo[j, 1]]
        plt.plot(x, y, color='black', linestyle='dotted', marker='o', markerfacecolor='black', markersize=12)
    plt.title(title)
    plt.show(block=False)


def plot_x(x, title='UNTITLED'):
    N = int(len(x)/3)
    sol = np.zeros((N, 3))
    for i in range(N):
        sol[i][0]=x[i*3]
        sol[i][1] = x[i * 3 + 1]
        sol[i][2] = x[i * 3 + 2]
    plt.figure()
    plt.plot(sol[:, 0], sol[:, 1], color='blue', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=12)
    # plt.plot(x_gt[:, 0], x_gt[:, 1], color='red', marker='o', markerfacecolor='red', markersize=12)
    plt.title(title)
    plt.show(block=False)
