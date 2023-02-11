from re import A
from venv import create
import numpy as np
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from mesh_generation.classy_blocks.classes.primitives import Edge
from mesh_generation.classy_blocks.classes.block import Block
from mesh_generation.classy_blocks.classes.mesh import Mesh
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def rotate_z(x, y, z, r_z):
    # rotation of cartesian coordinates around z axis by r_z radians
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    x_new = x * np.cos(r_z) - y * np.sin(r_z)
    y_new = x * np.sin(r_z) + y * np.cos(r_z)
    return x_new, y_new, z


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


def create_circle(d, flip):
    # from a centre, radius, and z rotation,
    #  create the points of a circle
    c_x, c_y, t, t_x, r, c_z = d
    if flip is False:
        alpha = np.linspace(0, 2 * np.pi, 100)
    else:
        alpha = np.linspace(2 * np.pi, 0, 100)
    z = r * np.cos(alpha) + c_z
    x = r * np.sin(alpha) + c_x
    y = [c_y for i in range(len(z))]
    x -= c_x
    y -= c_y
    (
        x,
        y,
        z,
    ) = rotate_x(x, y, z, t_x)
    x, y, z = rotate_z(x, y, z, t)
    x += c_x
    y += c_y
    x, y, z = rotate_z(x, y, z, 3 * np.pi / 2)
    return x, y, z


def cylindrical_convert(r, theta, z):
    # conversion to cylindrical coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = z
    return x, y, z


def interpolate(y, v, kind):
    # interpolates between a set of points by a factor of f

    x = np.linspace(0, len(y), len(y))
    x_new = np.linspace(0, len(y), v + 1)
    f = interp1d(x, y, kind=kind)
    y_new = f(x_new)
    return y_new[:-1]


def create_mesh(x: dict, length: float, tube_rad: float, path: str):

    coil_rad = x["coil_rad"]
    pitch = x["pitch"]
    inversion_loc = x["inversion_loc"]
    try:
        os.mkdir(path)
    except:
        print("already exists")

    x["fid_radial"] = int(x["fid_radial"])
    x["fid_axial"] = int(x["fid_axial"])

    coils = length / (2 * np.pi * coil_rad)
    h = coils * pitch
    keys = ["x", "y", "t", "t_x", "r", "z"]
    data = {}
    points = 20 + x["fid_radial"]

    t_x = -np.arctan(h / length)
    if inversion_loc < 0.05 or inversion_loc > 0.95:
        inversion_loc = None
    if inversion_loc is None:
        il = 1
        n = int(points)
        print("No inversion location specified")
    else:
        il = inversion_loc
        n = int(points * il)  # 8 interpolation points per rotation
    coil_vals = np.linspace(0, 2 * coils * np.pi * il, n)
    # x and y values around a circle
    data["x"] = [(coil_rad * np.cos(x_y)) for x_y in coil_vals]
    data["y"] = [(coil_rad * np.sin(x_y)) for x_y in coil_vals]
    # rotations around z are defined by number of coils
    data["t"] = list(coil_vals)
    data["t_x"] = [t_x for i in range(n)]
    # no change in radius for now
    data["r"] = [tube_rad for i in range(n)]
    # height is linear
    data["z"] = list(np.linspace(0, h * il, n))
    orig_len = len(data["t"])

    if inversion_loc is not None:

        il = inversion_loc
        n = int(points * (1 - il))
        new_tv = 2 * coils * np.pi * il + np.pi
        new_end = 2 * coils * np.pi + np.pi
        dt = new_end - new_tv
        coil_vals = np.linspace(new_end - dt, new_tv - dt, n)

        dx = data["x"][-1] * 2
        dy = data["y"][-1] * 2

        new_x = ([(coil_rad * (np.cos(x))) for x in coil_vals] + dx)[1:]
        new_y = ([(coil_rad * (np.sin(x))) for x in coil_vals] + dy)[1:]
        new_t = list(coil_vals)[1:]
        new_t_x = [-t_x for i in range(n)]
        new_r = [tube_rad for i in range(n)][1:]
        new_z = list(np.linspace(h * il, h, n))[1:]
        for i in range(len(new_x)):
            data["x"].append(new_x[i])
            data["y"].append(new_y[i])
            # rotations around z are defined by number of coils
            data["t"].append(new_t[i])
            data["t_x"].append(new_t_x[i])
            # no change in radius for now
            data["r"].append(new_r[i])
            # height is linear
            data["z"].append(new_z[i])

    port_len = coil_rad + tube_rad

    start_dx = data["x"][0]
    start_dy = data["y"][0] - port_len
    if inversion_loc is not None:
        end_theta = new_tv - dt
        end_dx = data["x"][-1] + port_len * np.sin(end_theta)
        end_dy = data["y"][-1] - port_len * np.cos(end_theta)
    else:
        end_theta = 2 * np.pi * coils
        end_dx = data["x"][-1] - port_len * np.sin(end_theta)
        end_dy = data["y"][-1] + port_len * np.cos(end_theta)

    print("Adding start and end ports")
    n_x = int(points / 6)

    inlet_x = np.linspace(start_dx, data["x"][0], n_x + 1)[:-1]
    inlet_y = np.linspace(start_dy, data["y"][0], n_x + 1)[:-1]
    outlet_x = np.linspace(data["x"][-1], end_dx, n_x + 1)[1:]
    outlet_y = np.linspace(data["y"][-1], end_dy, n_x + 1)[1:]

    mid = int(n_x / 2)

    hf_inlet_x = np.linspace(inlet_x[:mid][0], inlet_x[:mid][-1], n_x)
    hf_inlet_y = np.linspace(inlet_y[:mid][0], inlet_y[:mid][-1], n_x)

    inlet_x = np.append(hf_inlet_x, inlet_x[mid:])
    inlet_y = np.append(hf_inlet_y, inlet_y[mid:])

    hf_outlet_x = np.linspace(outlet_x[mid:][0], outlet_x[mid:][-1], n_x)
    hf_outlet_y = np.linspace(outlet_y[mid:][0], outlet_y[mid:][-1], n_x)
    outlet_x = np.append(outlet_x[:mid], hf_outlet_x)
    outlet_y = np.append(outlet_y[:mid], hf_outlet_y)

    n_x = len(inlet_x)
    print(n_x)

    data["x"] = np.append(np.append(inlet_x, data["x"]), outlet_x)
    data["y"] = np.append(np.append(inlet_y, data["y"]), outlet_y)
    data["t"] = np.append(
        np.append([data["t"][0] for i in range(n_x)], data["t"]),
        [data["t"][-1] for i in range(n_x)],
    )
    if inversion_loc is not None:
        data["t_x"] = np.append(
            np.append(
                -np.geomspace(0.0001, abs(data["t_x"][0]), n_x + 1)[:-1], data["t_x"]
            ),
            np.geomspace(abs(data["t_x"][-1]), 0.0001, n_x + 1)[1:],
        )
    else:
        data["t_x"] = np.append(
            np.append(
                -np.geomspace(0.0001, abs(data["t_x"][0]), n_x + 1)[:-1], data["t_x"]
            ),
            -np.geomspace(abs(data["t_x"][-1]), 0.0001, n_x + 1)[1:],
        )
    data["z"] = np.append(
        np.append([data["z"][0] for i in range(n_x)], data["z"]),
        [data["z"][-1] for i in range(n_x)],
    )
    data["r"] = np.append(
        np.append([data["r"][0] for i in range(n_x)], data["r"]),
        [data["r"][-1] for i in range(n_x)],
    )

    # calculating real values from differences and initial conditions

    le = len(data[keys[0]])
    mesh = Mesh()

    fig, axs = plt.subplots(1, 3, figsize=(10, 4), subplot_kw=dict(projection="3d"))

    orig_len += n_x - 1
    print("Creating mesh of coil")

    v = 6
    m = int(v / 2)

    d_z = [
        data["z"][-(n_x + m)],
        (data["z"][-(n_x + m)] + data["z"][-(n_x - m)] * 3) / 4,
        data["z"][-(n_x - m)],
    ]
    d_z = interpolate(d_z, v, "quadratic")
    data["z"][-(n_x + m) : -(n_x - m)] = d_z

    d_z = [
        data["z"][n_x - m],
        (data["z"][n_x + m] + data["z"][n_x - m] * 3) / 4,
        data["z"][n_x + m],
    ]
    d_z = interpolate(d_z, v, "quadratic")
    data["z"][n_x - m : n_x + m] = d_z

    for p in tqdm(range(1, le - 2)):

        # obtaining two circles
        if inversion_loc is not None:
            if p < orig_len:
                flip_1 = False
                flip_2 = False
            elif p == orig_len:
                flip_1 = False
                flip_2 = True

            else:
                flip_1 = True
                flip_2 = True

        if inversion_loc is None:
            flip_1 = False
            flip_2 = False

        if p == 0 or p == le - 2:
            x1, y1, z1 = create_circle(
                [data[keys[i]][p] for i in range(len(keys))], flip_1
            )
            x2, y2, z2 = create_circle(
                [data[keys[i]][p + 1] for i in range(len(keys))], flip_2
            )
        else:
            x2, y2, z2 = create_circle(
                [data[keys[i]][p] for i in range(len(keys))], flip_1
            )
            x1, y1, z1 = create_circle(
                [data[keys[i]][p + 1] for i in range(len(keys))], flip_2
            )

        for ax in axs:
            ax.plot3D(x2, y2, z2, color="k", alpha=0.75, lw=0.5)
            ax.plot3D(x1, y1, z1, color="k", alpha=0.75, lw=0.5)

            for i in np.linspace(0, len(x1) - 1, 20):
                i = int(i)
                ax.plot3D(
                    [x1[i], x2[i]],
                    [y1[i], y2[i]],
                    [z1[i], z2[i]],
                    c="k",
                    alpha=0.5,
                    lw=0.5,
                )

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_box_aspect(
            [ub - lb for lb, ub in (getattr(ax, f"get_{a}lim")() for a in "xyz")]
        )
    axs[0].set_xlabel("x")
    axs[0].set_zlabel("z")

    axs[1].set_ylabel("y")
    axs[1].set_zlabel("z")

    axs[2].set_ylabel("y")
    axs[2].set_xlabel("x")

    axs[0].view_init(0, 270)
    axs[1].view_init(0, 180)
    axs[2].view_init(270, 0)

    plt.subplots_adjust(left=0.01, right=0.99, wspace=0.05, top=0.99, bottom=0.01)
    # plt.savefig(path+"/pre-render.png", dpi=600)
    plt.show()

    return


def demo_inversion(x: dict, length: float, tube_rad: float, path: str):

    coil_rad = x["coil_rad"]
    pitch = x["pitch"]

    il_vals = [0, 0.15, 0.3, 0.6,0.75]
    fig, axs = plt.subplots(
        1, len(il_vals), figsize=(10, 4), subplot_kw=dict(projection="3d")
    )
    for k in range(len(il_vals)):
        inversion_loc = il_vals[k]

        x["fid_radial"] = int(x["fid_radial"])
        x["fid_axial"] = int(x["fid_axial"])

        coils = length / (2 * np.pi * coil_rad)
        h = coils * pitch
        keys = ["x", "y", "t", "t_x", "r", "z"]
        data = {}
        points = 20 + x["fid_radial"]

        t_x = -np.arctan(h / length)
        if inversion_loc < 0.05 or inversion_loc > 0.95:
            inversion_loc = None
        if inversion_loc is None:
            il = 1
            n = int(points)
            print("No inversion location specified")
        else:
            il = inversion_loc
            n = int(points * il)  # 8 interpolation points per rotation
        coil_vals = np.linspace(0, 2 * coils * np.pi * il, n)
        # x and y values around a circle
        data["x"] = [(coil_rad * np.cos(x_y)) for x_y in coil_vals]
        data["y"] = [(coil_rad * np.sin(x_y)) for x_y in coil_vals]
        # rotations around z are defined by number of coils
        data["t"] = list(coil_vals)
        data["t_x"] = [t_x for i in range(n)]
        # no change in radius for now
        data["r"] = [tube_rad for i in range(n)]
        # height is linear
        data["z"] = list(np.linspace(0, h * il, n))
        orig_len = len(data["t"])

        if inversion_loc is not None:

            il = inversion_loc
            n = int(points * (1 - il))
            new_tv = 2 * coils * np.pi * il + np.pi
            new_end = 2 * coils * np.pi + np.pi
            dt = new_end - new_tv
            coil_vals = np.linspace(new_end - dt, new_tv - dt, n)

            dx = data["x"][-1] * 2
            dy = data["y"][-1] * 2

            new_x = ([(coil_rad * (np.cos(x))) for x in coil_vals] + dx)[1:]
            new_y = ([(coil_rad * (np.sin(x))) for x in coil_vals] + dy)[1:]
            new_t = list(coil_vals)[1:]
            new_t_x = [-t_x for i in range(n)]
            new_r = [tube_rad for i in range(n)][1:]
            new_z = list(np.linspace(h * il, h, n))[1:]
            for i in range(len(new_x)):
                data["x"].append(new_x[i])
                data["y"].append(new_y[i])
                # rotations around z are defined by number of coils
                data["t"].append(new_t[i])
                data["t_x"].append(new_t_x[i])
                # no change in radius for now
                data["r"].append(new_r[i])
                # height is linear
                data["z"].append(new_z[i])

        port_len = coil_rad + tube_rad

        start_dx = data["x"][0]
        start_dy = data["y"][0] - port_len
        if inversion_loc is not None:
            end_theta = new_tv - dt
            end_dx = data["x"][-1] + port_len * np.sin(end_theta)
            end_dy = data["y"][-1] - port_len * np.cos(end_theta)
        else:
            end_theta = 2 * np.pi * coils
            end_dx = data["x"][-1] - port_len * np.sin(end_theta)
            end_dy = data["y"][-1] + port_len * np.cos(end_theta)

        print("Adding start and end ports")
        n_x = int(points / 6)

        inlet_x = np.linspace(start_dx, data["x"][0], n_x + 1)[:-1]
        inlet_y = np.linspace(start_dy, data["y"][0], n_x + 1)[:-1]
        outlet_x = np.linspace(data["x"][-1], end_dx, n_x + 1)[1:]
        outlet_y = np.linspace(data["y"][-1], end_dy, n_x + 1)[1:]

        mid = int(n_x / 2)

        hf_inlet_x = np.linspace(inlet_x[:mid][0], inlet_x[:mid][-1], n_x)
        hf_inlet_y = np.linspace(inlet_y[:mid][0], inlet_y[:mid][-1], n_x)

        inlet_x = np.append(hf_inlet_x, inlet_x[mid:])
        inlet_y = np.append(hf_inlet_y, inlet_y[mid:])

        hf_outlet_x = np.linspace(outlet_x[mid:][0], outlet_x[mid:][-1], n_x)
        hf_outlet_y = np.linspace(outlet_y[mid:][0], outlet_y[mid:][-1], n_x)
        outlet_x = np.append(outlet_x[:mid], hf_outlet_x)
        outlet_y = np.append(outlet_y[:mid], hf_outlet_y)

        n_x = len(inlet_x)

        data["x"] = np.append(np.append(inlet_x, data["x"]), outlet_x)
        data["y"] = np.append(np.append(inlet_y, data["y"]), outlet_y)
        data["t"] = np.append(
            np.append([data["t"][0] for i in range(n_x)], data["t"]),
            [data["t"][-1] for i in range(n_x)],
        )
        if inversion_loc is not None:
            data["t_x"] = np.append(
                np.append(
                    -np.geomspace(0.0001, abs(data["t_x"][0]), n_x + 1)[:-1],
                    data["t_x"],
                ),
                np.geomspace(abs(data["t_x"][-1]), 0.0001, n_x + 1)[1:],
            )
        else:
            data["t_x"] = np.append(
                np.append(
                    -np.geomspace(0.0001, abs(data["t_x"][0]), n_x + 1)[:-1],
                    data["t_x"],
                ),
                -np.geomspace(abs(data["t_x"][-1]), 0.0001, n_x + 1)[1:],
            )
        data["z"] = np.append(
            np.append([data["z"][0] for i in range(n_x)], data["z"]),
            [data["z"][-1] for i in range(n_x)],
        )
        data["r"] = np.append(
            np.append([data["r"][0] for i in range(n_x)], data["r"]),
            [data["r"][-1] for i in range(n_x)],
        )

        # calculating real values from differences and initial conditions

        le = len(data[keys[0]])
        mesh = Mesh()

        orig_len += n_x - 1
        print("Creating mesh of coil")

        v = 6
        m = int(v / 2)

        d_z = [
            data["z"][-(n_x + m)],
            (data["z"][-(n_x + m)] + data["z"][-(n_x - m)] * 3) / 4,
            data["z"][-(n_x - m)],
        ]
        d_z = interpolate(d_z, v, "quadratic")
        data["z"][-(n_x + m) : -(n_x - m)] = d_z

        d_z = [
            data["z"][n_x - m],
            (data["z"][n_x + m] + data["z"][n_x - m] * 3) / 4,
            data["z"][n_x + m],
        ]
        d_z = interpolate(d_z, v, "quadratic")
        data["z"][n_x - m : n_x + m] = d_z

        for p in tqdm(range(1, le - 2)):

            # obtaining two circles
            if inversion_loc is not None:
                if p < orig_len:
                    flip_1 = False
                    flip_2 = False
                elif p == orig_len:
                    flip_1 = False
                    flip_2 = True

                else:
                    flip_1 = True
                    flip_2 = True

            if inversion_loc is None:
                flip_1 = False
                flip_2 = False

            if p == 0 or p == le - 2:
                x1, y1, z1 = create_circle(
                    [data[keys[i]][p] for i in range(len(keys))], flip_1
                )
                x2, y2, z2 = create_circle(
                    [data[keys[i]][p + 1] for i in range(len(keys))], flip_2
                )
            else:
                x2, y2, z2 = create_circle(
                    [data[keys[i]][p] for i in range(len(keys))], flip_1
                )
                x1, y1, z1 = create_circle(
                    [data[keys[i]][p + 1] for i in range(len(keys))], flip_2
                )

            axs[k].set_axis_off()
            axs[k].plot3D(x2, y2, z2, color="k", alpha=0.75, lw=0.5)
            axs[k].plot3D(x1, y1, z1, color="k", alpha=0.75, lw=0.5)

            for i in np.linspace(0, len(x1) - 1, 20):
                i = int(i)
                axs[k].plot3D(
                    [x1[i], x2[i]],
                    [y1[i], y2[i]],
                    [z1[i], z2[i]],
                    c="k",
                    alpha=0.5,
                    lw=0.5,
                )

            axs[k].set_xticks([])
            axs[k].set_yticks([])
            axs[k].set_zticks([])
            axs[k].set_box_aspect(
                [
                    ub - lb
                    for lb, ub in (getattr(axs[k], f"get_{a}lim")() for a in "xyz")
                ]
            )
        axs[k].view_init(0,270)

    plt.tight_layout()
    plt.savefig("outputs/inversion_demo.png", dpi=800)
    return


def demo_pitch(x: dict, length: float, tube_rad: float, path: str):

    coil_rad = x["coil_rad"]
    inversion_loc = x["inversion_loc"]

    p_vals = np.linspace(0.0075, 0.015, 5)
    fig, axs = plt.subplots(
        1, len(p_vals), figsize=(10, 4), subplot_kw=dict(projection="3d")
    )
    for k in range(len(p_vals)):
        pitch = p_vals[k]

        x["fid_radial"] = int(x["fid_radial"])
        x["fid_axial"] = int(x["fid_axial"])

        coils = length / (2 * np.pi * coil_rad)
        h = coils * pitch
        keys = ["x", "y", "t", "t_x", "r", "z"]
        data = {}
        points = 20 + x["fid_radial"]

        t_x = -np.arctan(h / length)
        inversion_loc = None
        il = 1
        n = int(points)
        print("No inversion location specified")
        coil_vals = np.linspace(0, 2 * coils * np.pi * il, n)
        # x and y values around a circle
        data["x"] = [(coil_rad * np.cos(x_y)) for x_y in coil_vals]
        data["y"] = [(coil_rad * np.sin(x_y)) for x_y in coil_vals]
        # rotations around z are defined by number of coils
        data["t"] = list(coil_vals)
        data["t_x"] = [t_x for i in range(n)]
        # no change in radius for now
        data["r"] = [tube_rad for i in range(n)]
        # height is linear
        data["z"] = list(np.linspace(0, h * il, n))
        orig_len = len(data["t"])

        if inversion_loc is not None:

            il = inversion_loc
            n = int(points * (1 - il))
            new_tv = 2 * coils * np.pi * il + np.pi
            new_end = 2 * coils * np.pi + np.pi
            dt = new_end - new_tv
            coil_vals = np.linspace(new_end - dt, new_tv - dt, n)

            dx = data["x"][-1] * 2
            dy = data["y"][-1] * 2

            new_x = ([(coil_rad * (np.cos(x))) for x in coil_vals] + dx)[1:]
            new_y = ([(coil_rad * (np.sin(x))) for x in coil_vals] + dy)[1:]
            new_t = list(coil_vals)[1:]
            new_t_x = [-t_x for i in range(n)]
            new_r = [tube_rad for i in range(n)][1:]
            new_z = list(np.linspace(h * il, h, n))[1:]
            for i in range(len(new_x)):
                data["x"].append(new_x[i])
                data["y"].append(new_y[i])
                # rotations around z are defined by number of coils
                data["t"].append(new_t[i])
                data["t_x"].append(new_t_x[i])
                # no change in radius for now
                data["r"].append(new_r[i])
                # height is linear
                data["z"].append(new_z[i])

        port_len = coil_rad + tube_rad

        start_dx = data["x"][0]
        start_dy = data["y"][0] - port_len
        if inversion_loc is not None:
            end_theta = new_tv - dt
            end_dx = data["x"][-1] + port_len * np.sin(end_theta)
            end_dy = data["y"][-1] - port_len * np.cos(end_theta)
        else:
            end_theta = 2 * np.pi * coils
            end_dx = data["x"][-1] - port_len * np.sin(end_theta)
            end_dy = data["y"][-1] + port_len * np.cos(end_theta)

        print("Adding start and end ports")
        n_x = int(points / 6)

        inlet_x = np.linspace(start_dx, data["x"][0], n_x + 1)[:-1]
        inlet_y = np.linspace(start_dy, data["y"][0], n_x + 1)[:-1]
        outlet_x = np.linspace(data["x"][-1], end_dx, n_x + 1)[1:]
        outlet_y = np.linspace(data["y"][-1], end_dy, n_x + 1)[1:]

        mid = int(n_x / 2)

        hf_inlet_x = np.linspace(inlet_x[:mid][0], inlet_x[:mid][-1], n_x)
        hf_inlet_y = np.linspace(inlet_y[:mid][0], inlet_y[:mid][-1], n_x)

        inlet_x = np.append(hf_inlet_x, inlet_x[mid:])
        inlet_y = np.append(hf_inlet_y, inlet_y[mid:])

        hf_outlet_x = np.linspace(outlet_x[mid:][0], outlet_x[mid:][-1], n_x)
        hf_outlet_y = np.linspace(outlet_y[mid:][0], outlet_y[mid:][-1], n_x)
        outlet_x = np.append(outlet_x[:mid], hf_outlet_x)
        outlet_y = np.append(outlet_y[:mid], hf_outlet_y)

        n_x = len(inlet_x)

        data["x"] = np.append(np.append(inlet_x, data["x"]), outlet_x)
        data["y"] = np.append(np.append(inlet_y, data["y"]), outlet_y)
        data["t"] = np.append(
            np.append([data["t"][0] for i in range(n_x)], data["t"]),
            [data["t"][-1] for i in range(n_x)],
        )
        if inversion_loc is not None:
            data["t_x"] = np.append(
                np.append(
                    -np.geomspace(0.0001, abs(data["t_x"][0]), n_x + 1)[:-1],
                    data["t_x"],
                ),
                np.geomspace(abs(data["t_x"][-1]), 0.0001, n_x + 1)[1:],
            )
        else:
            data["t_x"] = np.append(
                np.append(
                    -np.geomspace(0.0001, abs(data["t_x"][0]), n_x + 1)[:-1],
                    data["t_x"],
                ),
                -np.geomspace(abs(data["t_x"][-1]), 0.0001, n_x + 1)[1:],
            )
        data["z"] = np.append(
            np.append([data["z"][0] for i in range(n_x)], data["z"]),
            [data["z"][-1] for i in range(n_x)],
        )
        data["r"] = np.append(
            np.append([data["r"][0] for i in range(n_x)], data["r"]),
            [data["r"][-1] for i in range(n_x)],
        )

        # calculating real values from differences and initial conditions

        le = len(data[keys[0]])
        mesh = Mesh()

        orig_len += n_x - 1
        print("Creating mesh of coil")

        v = 6
        m = int(v / 2)

        d_z = [
            data["z"][-(n_x + m)],
            (data["z"][-(n_x + m)] + data["z"][-(n_x - m)] * 3) / 4,
            data["z"][-(n_x - m)],
        ]
        d_z = interpolate(d_z, v, "quadratic")
        data["z"][-(n_x + m) : -(n_x - m)] = d_z

        d_z = [
            data["z"][n_x - m],
            (data["z"][n_x + m] + data["z"][n_x - m] * 3) / 4,
            data["z"][n_x + m],
        ]
        d_z = interpolate(d_z, v, "quadratic")
        data["z"][n_x - m : n_x + m] = d_z

        for p in tqdm(range(1, le - 2)):

            # obtaining two circles
            if inversion_loc is not None:
                if p < orig_len:
                    flip_1 = False
                    flip_2 = False
                elif p == orig_len:
                    flip_1 = False
                    flip_2 = True

                else:
                    flip_1 = True
                    flip_2 = True

            if inversion_loc is None:
                flip_1 = False
                flip_2 = False

            if p == 0 or p == le - 2:
                x1, y1, z1 = create_circle(
                    [data[keys[i]][p] for i in range(len(keys))], flip_1
                )
                x2, y2, z2 = create_circle(
                    [data[keys[i]][p + 1] for i in range(len(keys))], flip_2
                )
            else:
                x2, y2, z2 = create_circle(
                    [data[keys[i]][p] for i in range(len(keys))], flip_1
                )
                x1, y1, z1 = create_circle(
                    [data[keys[i]][p + 1] for i in range(len(keys))], flip_2
                )

            axs[k].set_axis_off()
            axs[k].plot3D(x2, y2, z2, color="k", alpha=0.75, lw=0.5)
            axs[k].plot3D(x1, y1, z1, color="k", alpha=0.75, lw=0.5)

            for i in np.linspace(0, len(x1) - 1, 20):
                i = int(i)
                axs[k].plot3D(
                    [x1[i], x2[i]],
                    [y1[i], y2[i]],
                    [z1[i], z2[i]],
                    c="k",
                    alpha=0.5,
                    lw=0.5,
                )

            axs[k].set_xticks([])
            axs[k].set_yticks([])
            axs[k].set_zticks([])
            axs[k].set_box_aspect(
                [
                    ub - lb
                    for lb, ub in (getattr(axs[k], f"get_{a}lim")() for a in "xyz")
                ]
            )


    plt.tight_layout()
    plt.savefig("outputs/inversion_demo.png", dpi=800)
    return


def demo_radius(x: dict, length: float, tube_rad: float, path: str):

    inversion_loc = x["inversion_loc"]
    pitch = x["pitch"]

    r_vals = np.linspace(0.003, 0.0125, 5)
    fig, axs = plt.subplots(
        1, len(r_vals), figsize=(10, 4), subplot_kw=dict(projection="3d")
    )
    for k in range(len(r_vals)):
        coil_rad = r_vals[k]

        x["fid_radial"] = int(x["fid_radial"])
        x["fid_axial"] = int(x["fid_axial"])

        coils = length / (2 * np.pi * coil_rad)
        h = coils * pitch
        keys = ["x", "y", "t", "t_x", "r", "z"]
        data = {}
        points = 20 + x["fid_radial"]

        t_x = -np.arctan(h / length)
        inversion_loc = None
        il = 1
        n = int(points)
        print("No inversion location specified")
        coil_vals = np.linspace(0, 2 * coils * np.pi * il, n)
        # x and y values around a circle
        data["x"] = [(coil_rad * np.cos(x_y)) for x_y in coil_vals]
        data["y"] = [(coil_rad * np.sin(x_y)) for x_y in coil_vals]
        # rotations around z are defined by number of coils
        data["t"] = list(coil_vals)
        data["t_x"] = [t_x for i in range(n)]
        # no change in radius for now
        data["r"] = [tube_rad for i in range(n)]
        # height is linear
        data["z"] = list(np.linspace(0, h * il, n))
        orig_len = len(data["t"])

        if inversion_loc is not None:

            il = inversion_loc
            n = int(points * (1 - il))
            new_tv = 2 * coils * np.pi * il + np.pi
            new_end = 2 * coils * np.pi + np.pi
            dt = new_end - new_tv
            coil_vals = np.linspace(new_end - dt, new_tv - dt, n)

            dx = data["x"][-1] * 2
            dy = data["y"][-1] * 2

            new_x = ([(coil_rad * (np.cos(x))) for x in coil_vals] + dx)[1:]
            new_y = ([(coil_rad * (np.sin(x))) for x in coil_vals] + dy)[1:]
            new_t = list(coil_vals)[1:]
            new_t_x = [-t_x for i in range(n)]
            new_r = [tube_rad for i in range(n)][1:]
            new_z = list(np.linspace(h * il, h, n))[1:]
            for i in range(len(new_x)):
                data["x"].append(new_x[i])
                data["y"].append(new_y[i])
                # rotations around z are defined by number of coils
                data["t"].append(new_t[i])
                data["t_x"].append(new_t_x[i])
                # no change in radius for now
                data["r"].append(new_r[i])
                # height is linear
                data["z"].append(new_z[i])

        port_len = coil_rad + tube_rad

        start_dx = data["x"][0]
        start_dy = data["y"][0] - port_len
        if inversion_loc is not None:
            end_theta = new_tv - dt
            end_dx = data["x"][-1] + port_len * np.sin(end_theta)
            end_dy = data["y"][-1] - port_len * np.cos(end_theta)
        else:
            end_theta = 2 * np.pi * coils
            end_dx = data["x"][-1] - port_len * np.sin(end_theta)
            end_dy = data["y"][-1] + port_len * np.cos(end_theta)

        print("Adding start and end ports")
        n_x = int(points / 6)

        inlet_x = np.linspace(start_dx, data["x"][0], n_x + 1)[:-1]
        inlet_y = np.linspace(start_dy, data["y"][0], n_x + 1)[:-1]
        outlet_x = np.linspace(data["x"][-1], end_dx, n_x + 1)[1:]
        outlet_y = np.linspace(data["y"][-1], end_dy, n_x + 1)[1:]

        mid = int(n_x / 2)

        hf_inlet_x = np.linspace(inlet_x[:mid][0], inlet_x[:mid][-1], n_x)
        hf_inlet_y = np.linspace(inlet_y[:mid][0], inlet_y[:mid][-1], n_x)

        inlet_x = np.append(hf_inlet_x, inlet_x[mid:])
        inlet_y = np.append(hf_inlet_y, inlet_y[mid:])

        hf_outlet_x = np.linspace(outlet_x[mid:][0], outlet_x[mid:][-1], n_x)
        hf_outlet_y = np.linspace(outlet_y[mid:][0], outlet_y[mid:][-1], n_x)
        outlet_x = np.append(outlet_x[:mid], hf_outlet_x)
        outlet_y = np.append(outlet_y[:mid], hf_outlet_y)

        n_x = len(inlet_x)

        data["x"] = np.append(np.append(inlet_x, data["x"]), outlet_x)
        data["y"] = np.append(np.append(inlet_y, data["y"]), outlet_y)
        data["t"] = np.append(
            np.append([data["t"][0] for i in range(n_x)], data["t"]),
            [data["t"][-1] for i in range(n_x)],
        )
        if inversion_loc is not None:
            data["t_x"] = np.append(
                np.append(
                    -np.geomspace(0.0001, abs(data["t_x"][0]), n_x + 1)[:-1],
                    data["t_x"],
                ),
                np.geomspace(abs(data["t_x"][-1]), 0.0001, n_x + 1)[1:],
            )
        else:
            data["t_x"] = np.append(
                np.append(
                    -np.geomspace(0.0001, abs(data["t_x"][0]), n_x + 1)[:-1],
                    data["t_x"],
                ),
                -np.geomspace(abs(data["t_x"][-1]), 0.0001, n_x + 1)[1:],
            )
        data["z"] = np.append(
            np.append([data["z"][0] for i in range(n_x)], data["z"]),
            [data["z"][-1] for i in range(n_x)],
        )
        data["r"] = np.append(
            np.append([data["r"][0] for i in range(n_x)], data["r"]),
            [data["r"][-1] for i in range(n_x)],
        )

        # calculating real values from differences and initial conditions

        le = len(data[keys[0]])
        mesh = Mesh()

        orig_len += n_x - 1
        print("Creating mesh of coil")

        v = 6
        m = int(v / 2)

        d_z = [
            data["z"][-(n_x + m)],
            (data["z"][-(n_x + m)] + data["z"][-(n_x - m)] * 3) / 4,
            data["z"][-(n_x - m)],
        ]
        d_z = interpolate(d_z, v, "quadratic")
        data["z"][-(n_x + m) : -(n_x - m)] = d_z

        d_z = [
            data["z"][n_x - m],
            (data["z"][n_x + m] + data["z"][n_x - m] * 3) / 4,
            data["z"][n_x + m],
        ]
        d_z = interpolate(d_z, v, "quadratic")
        data["z"][n_x - m : n_x + m] = d_z

        for p in tqdm(range(1, le - 2)):

            # obtaining two circles
            if inversion_loc is not None:
                if p < orig_len:
                    flip_1 = False
                    flip_2 = False
                elif p == orig_len:
                    flip_1 = False
                    flip_2 = True

                else:
                    flip_1 = True
                    flip_2 = True

            if inversion_loc is None:
                flip_1 = False
                flip_2 = False

            if p == 0 or p == le - 2:
                x1, y1, z1 = create_circle(
                    [data[keys[i]][p] for i in range(len(keys))], flip_1
                )
                x2, y2, z2 = create_circle(
                    [data[keys[i]][p + 1] for i in range(len(keys))], flip_2
                )
            else:
                x2, y2, z2 = create_circle(
                    [data[keys[i]][p] for i in range(len(keys))], flip_1
                )
                x1, y1, z1 = create_circle(
                    [data[keys[i]][p + 1] for i in range(len(keys))], flip_2
                )

            axs[k].set_axis_off()
            axs[k].plot3D(x2, y2, z2, color="k", alpha=0.75, lw=0.5)
            
            axs[k].plot3D(x1, y1, z1, color="k", alpha=0.75, lw=0.5)

            for i in np.linspace(0, len(x1) - 1, 20):
                i = int(i)
                axs[k].plot3D(
                    [x1[i], x2[i]],
                    [y1[i], y2[i]],
                    [z1[i], z2[i]],
                    c="k",
                    alpha=0.5,
                    lw=0.5,
                )

            axs[k].set_xticks([])
            axs[k].set_yticks([])
            axs[k].set_zticks([])
            axs[k].set_box_aspect(
                [
                    ub - lb
                    for lb, ub in (getattr(axs[k], f"get_{a}lim")() for a in "xyz")
                ]
            )
        axs[k].view_init(0, 270)

    plt.tight_layout()
    plt.savefig("outputs/radius_demo.png", dpi=800)
    return


# demo_radius(
#     {"inversion_loc": 0.001, "pitch": 0.0075, "fid_radial": 40, "fid_axial": 1},
#     0.0753,
#     0.0025,
#     "outputs/test",
# )
# demo_pitch({'inversion_loc':0.001,'coil_rad':0.0075,'fid_radial':40,'fid_axial':1},0.0753,0.0025,"outputs/test")
demo_inversion({'pitch':0.01,'coil_rad':0.0075,'fid_radial':40,'fid_axial':1},0.0753,0.0025,"outputs/test")
