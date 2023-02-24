import numpy as np
from scipy.interpolate import interp1d
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from mesh_generation.classy_blocks.classes.primitives import Edge
from mesh_generation.classy_blocks.classes.block import Block
from mesh_generation.classy_blocks.classes.mesh import Mesh
import shutil

import matplotlib.pyplot as plt
import math


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


def interpolate(y, fac_interp, kind, name):

    x = np.linspace(0, len(y), len(y))
    x_new = np.linspace(0, len(y), len(y) * fac_interp)
    f = interp1d(x, y, kind=kind)
    y_new = f(x_new)
    y = y_new 

    fac = 2
    cutoff = 0.1
    x_start = np.linspace(0,int(len(y)*cutoff),int(len(y)*cutoff))
    x_start_new = np.linspace(0,int(len(y)*cutoff),int(len(y)*cutoff)*fac)
    f = interp1d(x_start, y[:int(len(y)*cutoff)], kind=kind)
    y_start_new = f(x_start_new)
    x_end = np.linspace(0,int(len(y)*cutoff),int(len(y)*cutoff))
    x_end_new = np.linspace(0,int(len(y)*cutoff),int(len(y)*cutoff)*fac)
    f = interp1d(x_end, y[len(y)-int(len(y)*cutoff):], kind=kind)
    y_end_new = f(x_end_new)
    y_new = np.concatenate((y_start_new,y[int(len(y)*cutoff):int(len(y)*(1-cutoff))],y_end_new))

    # fig,ax = plt.subplots(1,1,figsize=(4,3))
    # fig.tight_layout()
    # ax.scatter(x,y,c='k')
    # ax.set_ylabel(name)
    # ax.plot(x_new,y_new,c='k')
    # plt.savefig('outputs/'+name+'.png')


    return y_new


def parse_inputs(NB, f, name):
    x = interpolate(NB, f, "quadratic", name)
    return x


def create_mesh(data, path, n_interp, nominal_data):
    # factor to interpolate between control points
    interpolation_factor = int(
        np.rint(data["fid_axial"])
    )  # interpolate x times the points between
    fid_radial = int(np.rint(data["fid_radial"]))

    keys = ["rho", "theta", "z", "tube_rad"]

    # do interpolation between points
    vals = {}
    # calculating real values from differences and initial conditions
    for i in range(n_interp):
        nominal_data["rho_" + str(i)] += data["rho_" + str(i)]
        nominal_data["z_" + str(i)] += data["z_" + str(i)]

    data = nominal_data

    for k in keys:
        data[k] = [data[k + "_" + str(i)] for i in range(n_interp)]
        vals[k] = parse_inputs(data[k], interpolation_factor, k)

    le = len(vals["z"])
    data = vals
    data["fid_radial"] = fid_radial
    mesh = Mesh()
    fig, axs = plt.subplots(1, 3, figsize=(9, 4), subplot_kw=dict(projection="3d"))

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
        for ax in axs:
            ax.plot3D(x2, y2, z2, c="k", lw=0.5)
            ax.plot3D(x1, y1, z1, c="k", alpha=0.75, lw=0.25)
            for i in np.linspace(0, len(x1) - 1, 10):
                i = int(i)
                ax.plot3D([x1[i], x2[i]], [y1[i], y2[i]], [z1[i], z2[i]], c="k", lw=0.5)

        # l defines the indices of 4 equally spaced points on circle
        l = np.linspace(0, len(x1), 5)[:4].astype(int)

        # centre of circle 1
        centre1 = np.mean(
            np.array([[x1[i], y1[i], z1[i]] for i in range(len(x1))]), axis=0
        )
        # centre of circle 2
        centre2 = np.mean(
            np.array([[x2[i], y2[i], z2[i]] for i in range(len(x1))]), axis=0
        )

        b = [0, 1, 2, 3, 0]

        for k in range(4):
            # moving around the 4 coordinates
            i = b[k]
            j = b[k + 1]
            # identify corners of blocks in O-topology
            block_points = [
                [x1[l[i]], y1[l[i]], z1[l[i]]],
                [x1[l[j]], y1[l[j]], z1[l[j]]],
                list(([x1[l[j]], y1[l[j]], z1[l[j]]] + centre1) / 2),
                list(([x1[l[i]], y1[l[i]], z1[l[i]]] + centre1) / 2),
            ] + [
                [x2[l[i]], y2[l[i]], z2[l[i]]],
                [x2[l[j]], y2[l[j]], z2[l[j]]],
                list(([x2[l[j]], y2[l[j]], z2[l[j]]] + centre2) / 2),
                list(([x2[l[i]], y2[l[i]], z2[l[i]]] + centre2) / 2),
            ]

            if l[j] == 0:
                v = l[j - 1] + (l[j - 1] - l[j - 2])
                a = int(l[i] + (v - l[i]) / 2)
            else:
                a = int(l[i] + (l[j] - l[i]) / 2)
            # add circlular arc edges for top of block (quarter of approx cylinder)
            block_edges = [
                Edge(0, 1, [x1[a], y1[a], z1[a]]),  # arc edges
                Edge(4, 5, [x2[a], y2[a], z2[a]]),
                Edge(2, 3, None),
                Edge(6, 7, None),
            ]

            # add to mesh
            block = Block.create_from_points(block_points, block_edges)
            # defined curved top as the wall
            block.set_patch(["front"], "walls")

            # partition block
            block.chop(0, count=data["fid_radial"])
            block.chop(1, count=data["fid_radial"])
            block.chop(2, count=1)

            # if at the start or end then state this
            if p == 1:
                block.set_patch("top", "inlet")
            if p == le - 2:
                block.set_patch("bottom", "outlet")

            mesh.add_block(block)

        # add final square block in O-topology
        block_points = [
            list(([x1[l[k]], y1[l[k]], z1[l[k]]] + centre1) / 2) for k in [0, 1, 2, 3]
        ] + [list(([x2[l[k]], y2[l[k]], z2[l[k]]] + centre2) / 2) for k in [0, 1, 2, 3]]

        block_edges = [
            Edge(0, 1, None),  # arc edges
            Edge(4, 5, None),
            Edge(2, 3, None),  # spline edges
            Edge(6, 7, None),
        ]
        block = Block.create_from_points(block_points, block_edges)
        if p == 1:
            block.set_patch("top", "inlet")
        if p == le - 2:
            block.set_patch("bottom", "outlet")

        block.chop(0, count=data["fid_radial"])
        block.chop(1, count=data["fid_radial"])
        block.chop(2, count=1)

        mesh.add_block(block)

    # save matlpotlib plot for easy debugging
    for ax in axs:
        ax.set_box_aspect(
            [ub - lb for lb, ub in (getattr(ax, f"get_{a}lim")() for a in "xyz")]
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    # axs[0].set_xlabel("x")
    # axs[0].set_zlabel("z")

    # axs[1].set_ylabel("y")
    # axs[1].set_zlabel("z")

    # axs[2].set_ylabel("y")
    # axs[2].set_xlabel("x")

    axs[0].view_init(0, 270)
    axs[1].view_init(0, 180)
    axs[2].view_init(270, 0)

    # plt.subplots_adjust(left=0.01,right=0.99,wspace=0,top=1)
    # plt.show()
    # copy existing base mesh folder
    try:
        os.mkdir("visualiation/website_gif/")
    except:
        print("file already exists...")
    plt.tight_layout()
    plt.savefig(str(path) + "/single_img.png", dpi=600)
    return
