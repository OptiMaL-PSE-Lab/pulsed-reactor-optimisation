import numpy as np
from scipy.interpolate import interp1d
from classy_blocks.classes.primitives import Edge
from classy_blocks.classes.block import Block
from classy_blocks.classes.mesh import Mesh
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


def parse_inputs( x, f):
    # transform initial conditions and differences to values
    x = interpolate(x, f, "quadratic")
    return x


def create_mesh(coil_rad, tube_rad, pitch, length, path):


    
    coils = length/(2*np.pi*coil_rad)

    h = pitch * coils 
    n = int(coils * 8)  # 8 interpolation points per rotation
    interpolation_factor = 10  # interpolate 4 times the points between
    keys = ["x", "y", "t", "r", "z"]

    data = {}

    # x and y values around a circle
    data["x"] = [(coil_rad * np.cos(x_y)) for x_y in np.linspace(0, 2 * coils * np.pi, n)]
    data["y"] = [(coil_rad * np.sin(x_y)) for x_y in np.linspace(0, 2 * coils * np.pi, n)]
    # rotations around z are defined by number of coils
    data["t"] = np.linspace(0, 2 * coils * np.pi, n)
    # no change in radius for now
    data["r"] = [tube_rad for i in range(n)]
    # height is linear
    data["z"] = np.linspace(0,h,n)

    port_len = tube_rad*5
    start_dx =  data['x'][0] + port_len * np.sin(0) 
    start_dy =  data['y'][0] - port_len * np.cos(0) 
    end_dx = data['x'][-1] - port_len * np.sin(2 * coils * np.pi)
    end_dy = data['y'][-1] + port_len * np.cos(2 * coils * np.pi)

    data['x'] = np.append(np.append([start_dx],data['x']),[end_dx]) 
    data['y'] = np.append(np.append([start_dy],data['y']),[end_dy]) 
    data['t'] = np.append(np.append([data['t'][0]],data['t']),[data['t'][-1]])
    data['z'] = np.append(np.append([data['z'][0]],data['z']),[data['z'][-1]])
    data['r'] = np.append(np.append([data['r'][0]],data['r']),[data['r'][-1]])


    # calculating real values from differences and initial conditions
    for k in keys:
        data[k] = parse_inputs(data[k], interpolation_factor)


    le = len(data[keys[0]])
    mesh = Mesh()
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for p in range(le - 1):

        # obtaining two circles
        x1, y1, z1 = create_circle([data[keys[i]][p] for i in range(len(keys))])
        x2, y2, z2 = create_circle(
            [data[keys[i]][p + 1] for i in range(len(keys))]
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
            if p == 0:
                block.set_patch("bottom", "inlet")
            if p == le - 2:
                block.set_patch("top", "outlet")


            block.chop(0, count=10)
            block.chop(1, count=10)
            block.chop(2, count=1)

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
        # partition block
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
    ax.set_title('Pitch: '+str(np.round(pitch,2))+' Coil Radius: '+str(np.round(coil_rad,2)))
    plt.savefig("output_images/"+path+".png", dpi=1000)
    try:
        shutil.copytree("base", path)
    except:
        print("file already exists...")

    # run script to create mesh
    mesh.write(output_path=os.path.join(path, "system", "blockMeshDict"), geometry=None)
    os.system(path + "/Allrun.mesh")


tube_rad = 0.5
length = 50


coil_rad = 3
pitch = 1.5
create_mesh(coil_rad, tube_rad, pitch, length, path="coil_basic")
