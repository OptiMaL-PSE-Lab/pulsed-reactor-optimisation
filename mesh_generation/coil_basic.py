import numpy as np
from scipy.interpolate import interp1d
from classy_examples.classy_blocks.classes.primitives import Edge
from classy_examples.classy_blocks.classes.block import Block
from classy_examples.classy_blocks.classes.mesh import Mesh
import shutil
import os
import matplotlib.pyplot as plt


def rotate_z(x, y, z, r_z):
    # rotation of cartesian coordinates around z axis by r_z radians
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    x_new = x * np.cos(r_z) - y * np.sin(r_z)
    y_new = x * np.sin(r_z) + y * np.cos(r_z)
    return x_new, y_new, z


def create_circle(d):
    # from a centre, radius, and z rotation,
    #  create the points of a circle
    c_x, c_y, t, r, c_z = d
    alpha = np.linspace(0, 2 * np.pi, 100)
    z = r * np.cos(alpha) + c_z
    x = r * np.sin(alpha) + c_x
    y = [c_y for i in range(len(z))]
    x -= c_x
    y -= c_y
    x, y, z = rotate_z(x, y, z, t)
    x += c_x
    y += c_y
    return x, y, z


def interpolate(y, f, kind):
    # interpolate between points y, by a factor of f
    x = np.linspace(0, len(y), len(y))
    x_new = np.linspace(0, len(y), len(y) * f)
    f = interp1d(x, y, kind=kind)
    y_new = f(x_new)

    # # plot if you want to
    # plt.figure()
    # plt.scatter(x, y)
    # plt.plot(x_new,y_new)
    # plt.show()
    return y_new


def parse_inputs(x0, x, f):
    # transform initial conditions and differences to values
    x = np.cumsum(np.append([x0], x))
    x = interpolate(x, f, "quadratic")
    return x


def create_mesh(coil_rad, tube_rad, coils, h, path):

    n = coils * 8  # 8 interpolation points per rotation
    interpolation_factor = 10  # interpolate 4 times the points between
    keys = ["x", "y", "t", "r", "z"]
    initial_vals = [0, 0, 0, tube_rad, 0]  # initial values for keys

    data = {}
    for i in range(len(keys)):
        data[keys[i]] = {}
        data[keys[i]]["0"] = initial_vals[i]

    # for the basic coil these are calculated within the function

    # x and y values around a circle
    data["x"]["diff"] = np.diff(
        [(coil_rad * np.cos(x_y)) for x_y in np.linspace(0, 2 * coils * np.pi, n)]
    )
    data["y"]["diff"] = np.diff(
        [(coil_rad * np.sin(x_y)) for x_y in np.linspace(0, 2 * coils * np.pi, n)]
    )

    # rotations around z are defined by number of coils
    data["t"]["diff"] = np.diff(np.linspace(0, 2 * coils * np.pi, n))
    # no change in radius for now
    data["r"]["diff"] = [0 for i in range(n)]
    # height is linear
    data["z"]["diff"] = [h / n for i in range(n)]

    # calculating real values from differences and initial conditions
    for k in keys:
        data[k]["vals"] = parse_inputs(
            data[k]["0"], data[k]["diff"], interpolation_factor
        )

    le = len(data[keys[0]]["vals"])
    mesh = Mesh()
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for p in range(le - 1):

        # obtaining two circles
        x1, y1, z1 = create_circle([data[keys[i]]["vals"][p] for i in range(len(keys))])
        x2, y2, z2 = create_circle(
            [data[keys[i]]["vals"][p + 1] for i in range(len(keys))]
        )

        ax.plot3D(x2, y2, z2, c="k", alpha=0.75, lw=0.25)
        ax.plot3D(x1, y1, z1, c="k", alpha=0.75, lw=0.25)
        for i in np.linspace(0,len(x1)-1,10):
                i = int(i)
                ax.plot3D([x1[i],x2[i]],[y1[i],y2[i]],[z1[i],z2[i]],c='k',alpha=0.5,lw=0.25)



        l = np.linspace(0, len(x1), 5)[:4].astype(int)

        centre1 = np.mean(
            np.array([[x1[i], y1[i], z1[i]] for i in range(len(x1))]), axis=0
        )
        centre2 = np.mean(
            np.array([[x2[i], y2[i], z2[i]] for i in range(len(x1))]), axis=0
        )

        b = [0, 1, 2, 3, 0]

        for k in range(4):

            # O-Topology for what is effectively
            # a slanted cylinder
            i = b[k]
            j = b[k + 1]

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

            # add circular curves at top of cylindrical quadrant
            block_edges = [
                Edge(0, 1, [x1[a], y1[a], z1[a]]),  # arc edges
                Edge(4, 5, [x2[a], y2[a], z2[a]]),
                Edge(2, 3, None),
                Edge(6, 7, None),
            ]

            block = Block.create_from_points(block_points, block_edges)

            block.set_patch(["front"], "walls")

            # partition block
            block.chop(0, count=10)
            block.chop(1, count=10)
            block.chop(2, count=1)

            if p == 0:
                block.set_patch("bottom", "inlet")
            if p == le - 2:
                block.set_patch("top", "outlet")

            mesh.add_block(block)

        # add centre rectangular block
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
        if p == 0:
            block.set_patch("bottom", "inlet")
        if p == le - 2:
            block.set_patch("top", "outlet")

        block.chop(0, count=10)
        block.chop(1, count=10)
        block.chop(2, count=1)

        mesh.add_block(block)
        # copy template folder
    ax.set_box_aspect(
        [ub - lb for lb, ub in (getattr(ax, f"get_{a}lim")() for a in "xyz")]
    )
    plt.savefig("output_images/pre_render_basic.png", dpi=1000)
    try:
        shutil.copytree("base", path)
    except:
        print("file already exists...")

    # run script to create mesh
    mesh.write(output_path=os.path.join(path, "system", "blockMeshDict"), geometry=None)
    os.system(path + "/Allrun.mesh")


# coil radius
coil_rad = 10
# tube radius
tube_rad = 1
# number of coils
coils = 4
# coil height
h = 20

# create coil
create_mesh(coil_rad, tube_rad, coils, h, path="coil_basic")
