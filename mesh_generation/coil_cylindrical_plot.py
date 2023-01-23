from re import A, I
import numpy as np
from scipy.interpolate import interp1d
from classy_blocks.classes.primitives import Edge
from classy_blocks.classes.block import Block
from classy_blocks.classes.mesh import Mesh
import shutil
import os
import matplotlib.pyplot as plt
import math
import imageio
import os


def rotate_z(x0, y0, z0, r_z):
    # rotation of points around z axis by r_z radians
    x = np.array(x0) - np.mean(x0)
    y = np.array(y0) - np.mean(y0)
    x_new = x * np.cos(r_z) - y * np.sin(r_z)
    y_new = x * np.sin(r_z) + y * np.cos(r_z)
    x_new += np.mean(x0)
    y_new += np.mean(y0)
    return x_new, y_new, z0


def rotate_x(x0, y0, z0, r_x):
    # rotation of points around x axis by r_x radians
    y = np.array(y0) - np.mean(y0)
    z = np.array(z0) - np.mean(z0)
    y_new = y * np.cos(r_x) - z * np.sin(r_x)
    z_new = y * np.sin(r_x) - z * np.cos(r_x)
    y_new += np.mean(y0)
    z_new += np.mean(z0)
    return x0, y_new, z_new


def rotate_y(x0, y0, z0, r_y):
    # rotation of points around y axis by r_y radians
    x = np.array(x0) - np.mean(x0)
    z = np.array(z0) - np.mean(z0)
    z_new = z * np.cos(r_y) - x * np.sin(r_y)
    x_new = z * np.sin(r_y) - x * np.cos(r_y)
    x_new += np.mean(x0)
    z_new += np.mean(z0)
    return x_new, y0, z_new


def cylindrical_convert(r, theta, z):
    # conversion to cylindrical coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = z
    return x, y, z


def unit_vector(vector):
    # returns a unit vector
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    # angle between two vectors
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def create_circle(d1, d2):
    # takes 2 cylindrical coordinates
    # and rotates the location of the second to be orthogonal
    # to the vector between the centre of the two circles

    # circle_test.py provides an example
    r1, t1, z1, rad1 = d1
    r2, t2, z2, rad2 = d2
    c_x1, c_y1, c_z1 = cylindrical_convert(r1, t1, z1)
    c_x2, c_y2, c_z2 = cylindrical_convert(r2, t2, z2)
    alpha = np.linspace(0, 2 * np.pi, 100)
    y1 = rad1 * np.cos(alpha) + c_y1
    x1 = rad1 * np.sin(alpha) + c_x1
    z1 = [c_z1 for i in range(len(x1))]
    y2 = rad2 * np.cos(alpha) + c_y2
    x2 = rad2 * np.sin(alpha) + c_x2
    z2 = [c_z2 for i in range(len(x2))]
    c1 = np.mean([x1, y1, z1], axis=1)
    c2 = np.mean([x2, y2, z2], axis=1)
    x1, y1, z1 = np.array([x1, y1, z1]) - np.array([c1 for i in range(len(x1))]).T
    x2, y2, z2 = np.array([x2, y2, z2]) - np.array([c1 for i in range(len(x1))]).T
    v = c2 - c1
    a_z = angle_between([0, 0, 1], v)
    xv = [v[0], v[1]]
    b = [0, 1]
    a_x = math.atan2(xv[1] * b[0] - xv[0] * b[1], xv[0] * b[0] + xv[1] * b[1])
    x2p, y2p, z2p = rotate_x(x2, y2, z2, (-a_z))
    x2p, y2p, z2p = rotate_z(x2p, y2p, z2p, a_x)
    return x2p + c1[0], y2p + c1[1], z2p + c1[2]


def interpolate(y, f, kind):
    # interpolates between a set of points by a factor of f

    x = np.linspace(0, len(y), len(y))
    x_new = np.linspace(0, len(y), len(y) * f)
    f = interp1d(x, y, kind=kind)
    y_new = f(x_new)
    # plot interpolation if needed

    # plt.figure()
    # plt.scatter(x, y)
    # plt.plot(x_new,y_new)
    # plt.show()

    return y_new


def parse_inputs(NB, f):
    x = interpolate(NB, f, "quadratic")
    return x


def create_mesh(ax, data):

    # factor to interpolate between control points
    interpolation_factor = data["factor"]  # interpolate 10 times the points between

    keys = ["rho", "theta", "z", "tube_rad"]

    # do interpolation between points
    vals = {}
    # calculating real values from differences and initial conditions
    for k in keys:
        vals[k] = parse_inputs(data[k], interpolation_factor)

    le = len(vals["z"])
    data = vals

    for p in range(1, le - 1):
        # get proceeding circle (as x,y,z samples)
        x2, y2, z2 = create_circle(
            [data[keys[i]][p - 1] for i in range(len(keys))],
            [data[keys[i]][p] for i in range(len(keys))],
        )
        # get next circle (as x,y,z samples)

        x1, y1, z1 = create_circle(
            [data[keys[i]][p] for i in range(len(keys))],
            [data[keys[i]][p + 1] for i in range(len(keys))],
        )
        # plot for reference
        ax.plot3D(x2, y2, z2, c="k", alpha=0.75, lw=0.25)
        for i in np.linspace(0, len(x1) - 1, 10):
            i = int(i)
            ax.plot3D(
                [x1[i], x2[i]],
                [y1[i], y2[i]],
                [z1[i], z2[i]],
                c="k",
                alpha=0.5,
                lw=0.25,
            )

    # save matlpotlib plot for easy debugging
    ax.set_box_aspect(
        [ub - lb for lb, ub in (getattr(ax, f"get_{a}lim")() for a in "xyz")]
    )
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(0, 40)

    return


coils = 4  # number of coils
coil_rad_max = 10  # max coil radius
h = 40  # max height
N = 2 * np.pi * coils  # angular turns (radians)
n = 4  # points per coil to use
f = 30
# example data for cylindrical coordinates (normally optimized for)
# as close as possible to the paper
data = {}
data["rho"] = [5 for i in range(n)]
data["theta"] = np.linspace(0, N, n)
data["z"] = np.linspace(0, 40, n)
data["tube_rad"] = [0.75 for i in range(n)]
data["factor"] = f


for im in range(40):
    fig = plt.figure(figsize=(6, 2))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    # First subplot
    i = 1
    vals = {}
    # calculating real values from differences and initial conditions
    for k in data.keys():
        if k != "factor":
            vals[k] = parse_inputs(data[k], data["factor"])

    ylims = [[-5, 30], [-2, 15]]
    for key in ["theta", "rho"]:
        ax = fig.add_subplot(1, 3, i)
        ax.scatter(data["z"], data[key], c="k")
        ax.plot(vals["z"], vals[key], c="k")
        ax.set_xlim(-5, 45)
        ax.set_xlabel(r"$z$")
        ax.set_ylim(ylims[i - 1][0], ylims[i - 1][1])
        if i == 1:
            ax.set_ylabel(r"$\theta$")
        else:
            ax.set_ylabel(r"$\rho$")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        i += 1

    ax = fig.add_subplot(1, 3, 3, projection="3d")
    create_mesh(ax, data)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    data["theta"] += 2 * np.sin(data["z"] / 4 + im / 2)
    data["rho"] += 1 * np.cos(data["z"] / 4 + im / 2)
    plt.savefig("mesh_generation/output_images/" + str(im) + ".png")


images = []  # creating image array

for im in range(40):  # iterating over images

    images.append(
        imageio.imread("mesh_generation/output_images/" + str(im) + ".png")
    )  # adding each image to the array
    # note see how this follows the standard naming convention

    os.remove(
        "mesh_generation/output_images/" + str(im) + ".png"
    )  # this then deletes the image file from the folder

imageio.mimsave(
    "mesh_generation/output_images/cylindrical.gif", images
)  # this then saves theA
