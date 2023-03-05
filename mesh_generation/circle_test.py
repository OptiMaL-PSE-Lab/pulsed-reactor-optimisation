from os import X_OK
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import math


def rotate_z(x0, y0, z0, r_z):
    """
    Rotate a point cloud about the z axis
    :param x0: x coordinates
    :param y0: y coordinates
    :param z0: z coordinates
    :param r_z: rotation angle
    :return: rotated point cloud
    """
    x = np.array(x0) - np.mean(x0)
    y = np.array(y0) - np.mean(y0)
    x_new = x * np.cos(r_z) - y * np.sin(r_z)
    y_new = x * np.sin(r_z) + y * np.cos(r_z)
    x_new += np.mean(x0)
    y_new += np.mean(y0)
    return x_new, y_new, z0


def rotate_x(x0, y0, z0, r_x):
    """
    Rotate a point cloud about the x axis
    :param x0: x coordinates
    :param y0: y coordinates
    :param z0: z coordinates
    :param r_x: rotation angle
    :return: rotated point cloud
    """
    y = np.array(y0) - np.mean(y0)
    z = np.array(z0) - np.mean(z0)
    y_new = y * np.cos(r_x) - z * np.sin(r_x)
    z_new = y * np.sin(r_x) - z * np.cos(r_x)
    y_new += np.mean(y0)
    z_new += np.mean(z0)
    return x0, y_new, z_new


def rotate_y(x0, y0, z0, r_y):
    """
    Rotate a point cloud about the y axis
    :param x0: x coordinates
    :param y0: y coordinates
    :param z0: z coordinates
    :param r_y: rotation angle
    :return: rotated point cloud
    """
    x = np.array(x0) - np.mean(x0)
    z = np.array(z0) - np.mean(z0)
    z_new = z * np.cos(r_y) - x * np.sin(r_y)
    x_new = z * np.sin(r_y) - x * np.cos(r_y)
    x_new += np.mean(x0)
    z_new += np.mean(z0)
    return x_new, y0, z_new


def cylindrical_convert(r, theta, z):
    """
    Convert cylindrical coordinates to cartesian coordinates
    :param r: radius
    :param theta: angle
    :param z: z coordinate
    :return: x, y, z coordinates
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = z
    return x, y, z


def create_circle(rho, theta, r, z):
    """
    Create a circle from cylindrical coordinates
    :param rho: radius
    :param theta: angle
    :param r: radius of circle
    :param z: z coordinate
    :return: x, y, z coordinates
    """

    # Convert to cartesian coordinates
    c_x, c_y, c_z = cylindrical_convert(rho, theta, z)
    # Create circle
    alpha = np.linspace(0, 2 * np.pi, 100)
    y = r * np.cos(alpha) + c_y
    x = r * np.sin(alpha) + c_x
    z = [c_z for i in range(len(x))]
    return x, y, z


def rotation_matrix_from_vectors(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (
        vec2 / np.linalg.norm(vec2)
    ).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix


def plot_v(c1, c2, ax, c):
    ax.plot3D([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]], c=c)
    return


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def create_circle(r1, t1, z1, r2, t2, z2, rad):
    c_x1, c_y1, c_z1 = cylindrical_convert(r1, t1, z1)
    c_x2, c_y2, c_z2 = cylindrical_convert(r2, t2, z2)
    alpha = np.linspace(0, 2 * np.pi, 100)
    y1 = rad * np.cos(alpha) + c_y1
    x1 = rad * np.sin(alpha) + c_x1
    z1 = [c_z1 for i in range(len(x1))]

    y2 = rad * np.cos(alpha) + c_y2
    x2 = rad * np.sin(alpha) + c_x2
    z2 = [c_z2 for i in range(len(x2))]

    c1 = np.mean([x1, y1, z1], axis=1)
    c2 = np.mean([x2, y2, z2], axis=1)

    x1, y1, z1 = np.array([x1, y1, z1]) - np.array([c1 for i in range(len(x1))]).T

    x2, y2, z2 = np.array([x2, y2, z2]) - np.array([c1 for i in range(len(x1))]).T

    u = np.eye(3)

    v = c2 - c1

    a_z = angle_between([0, 0, 1], v)

    xv = [v[0], v[1]]
    b = [0, 1]

    a_x = math.atan2(xv[1] * b[0] - xv[0] * b[1], xv[0] * b[0] + xv[1] * b[1])

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot3D(x1, y1, z1, c="k")
    ax.plot3D(x2, y2, z2, c="k", ls="dashed")

    for i in u:
        plot_v([0, 0, 0], i, ax, "k")

    plot_v([0, 0, 0], v, ax, "r")
    ax.plot3D(x2, y2, z2, c="r", ls="dashed")
    x2p, y2p, z2p = rotate_x(x2, y2, z2, (-a_z))
    ax.plot3D(x2p, y2p, z2p, c="g", ls="dashed")
    x2p, y2p, z2p = rotate_z(x2p, y2p, z2p, a_x)
    ax.plot3D(x2p, y2p, z2p, c="b", ls="dashed")

    ax.set_xlim(-1, 3)
    ax.set_ylim(-1, 3)

    plt.show()
    return


rad = 0.5
r1 = 1
z1 = 0.5
t1 = 0.7
r2 = 2
z2 = 3
t2 = 1

create_circle(r1, t1, z1, r2, t2, z2, rad)
