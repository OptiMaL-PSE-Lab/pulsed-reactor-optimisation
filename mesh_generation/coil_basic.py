import numpy as np
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from mesh_generation.classy_blocks.classes.primitives import Edge
from mesh_generation.classy_blocks.classes.block import Block
from mesh_generation.classy_blocks.classes.mesh import Mesh
import shutil
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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

def create_circle(d,flip):
    # from a centre, radius, and z rotation,
    #  create the points of a circle
    c_x, c_y, t, r, c_z = d
    if flip is False:
        alpha = np.linspace(0, 2 * np.pi, 100)
    else:
        alpha = np.linspace(2*np.pi,0,100)
    z = r * np.cos(alpha) + c_z
    x = r * np.sin(alpha) + c_x
    y = [c_y for i in range(len(z))]
    x -= c_x
    y -= c_y
    x, y, z = rotate_z(x, y, z, t)
    x += c_x
    y += c_y
    x, y, z = rotate_z(x, y, z, np.pi/2)
    return x, y, z


def create_mesh(coil_rad, tube_rad, pitch, length, inversion_loc, path):

    coils = length/(2*np.pi*coil_rad)
    h = pitch * coils 
    keys = ["x", "y", "t", "r", "z"]
    data = {}
    points = 50

    if inversion_loc is None:
        il = 1
        n = int(coils * points)  # 8 interpolation points per rotation
    else:
        il = inversion_loc
        n = int(coils * points * il)  # 8 interpolation points per rotation

    coil_vals = np.linspace(0, 2 * coils * np.pi*il, n)
    # x and y values around a circle
    data["x"] = [(coil_rad * np.cos(x_y)) for x_y in coil_vals]
    data["y"] = [(coil_rad * np.sin(x_y)) for x_y in coil_vals]
    # rotations around z are defined by number of coils
    data["t"] = list(coil_vals)
    # no change in radius for now
    data["r"] = [tube_rad for i in range(n)]
    # height is linear
    data["z"] = list(np.linspace(0,h*il,n))
    orig_len = len(data['t'])

    if inversion_loc is not None:

        il = inversion_loc
        n = int(coils * points * (1-il))
        new_tv = 2*coils*np.pi*il + np.pi
        new_end = 2*coils*np.pi + np.pi
        dt = new_end-new_tv
        coil_vals = np.linspace(new_end-dt, new_tv-dt , n)

        dx = data['x'][-1]*2
        dy = data['y'][-1]*2
        
        new_x = ([(coil_rad * (np.cos(x))) for x in coil_vals] + dx)[1:]
        new_y = ([(coil_rad * (np.sin(x))) for x in coil_vals] + dy)[1:]
        new_t = list(coil_vals)[1:]
        new_r = [tube_rad for i in range(n)][1:]
        new_z = list(np.linspace(h*il,h,n))[1:]
        for i in range(len(new_x)):
            data["x"].append(new_x[i])
            data["y"].append(new_y[i])
            # rotations around z are defined by number of coils
            data["t"].append(new_t[i])
            # no change in radius for now
            data["r"].append(new_r[i])
            # height is linear
            data["z"].append(new_z[i])

    port_len = tube_rad*10
    start_dx =  data['x'][0] 
    start_dy =  data['y'][0] - port_len 
    if inversion_loc is not None:
        end_theta = new_tv-dt
        end_dx = data['x'][-1] + port_len * np.sin(end_theta)
        end_dy = data['y'][-1] - port_len * np.cos(end_theta) 
    else:
        end_theta = 2*np.pi*coils
        end_dx = data['x'][-1] - port_len * np.sin(end_theta)
        end_dy = data['y'][-1] + port_len * np.cos(end_theta) 


    data['x'] = np.append(np.append([start_dx],data['x']),[end_dx]) 
    data['y'] = np.append(np.append([start_dy],data['y']),[end_dy]) 
    data['t'] = np.append(np.append([data['t'][0]],data['t']),[data['t'][-1]])
    data['z'] = np.append(np.append([data['z'][0]],data['z']),[data['z'][-1]])
    data['r'] = np.append(np.append([data['r'][0]],data['r']),[data['r'][-1]])


    # calculating real values from differences and initial conditions


    le = len(data[keys[0]])
    mesh = Mesh()
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(projection="3d")
    for p in range(le-1):

        # obtaining two circles
        if p < orig_len:
            flip_1 = False
            flip_2 = False
        elif p == orig_len:
            flip_1 = False
            flip_2 = True
        else:
            flip_1 = True
            flip_2 = True

        if p == le - 2 and inversion_loc is None:
            flip_1 = False
            flip_2 = False

        x1, y1, z1 = create_circle([data[keys[i]][p] for i in range(len(keys))],flip_1)
        x2, y2, z2 = create_circle(
            [data[keys[i]][p + 1] for i in range(len(keys))],flip_2)


        ax.plot3D(x2, y2, z2, color='k', alpha=0.75, lw=0.5)
        ax.plot3D(x1, y1, z1, color='k', alpha=0.75, lw=0.5)

        
        for i in np.linspace(0,len(x1)-1,20):
                i = int(i)
                ax.plot3D([x1[i],x2[i]],[y1[i],y2[i]],[z1[i],z2[i]],c='k',alpha=0.5,lw=1)


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
    ax.set_title('Pitch: '+str(np.round(pitch,2))+', Coil Radius: '+str(np.round(coil_rad,2))+', Inversion %: '+str(np.round(il,2)*100))
    #plt.show()
    plt.savefig(path+"/pre-render.png", dpi=400)
    try:
        shutil.copytree("mesh_generation/base", path)
    except:
        print("file already exists...")

    # run script to create mesh
    mesh.write(output_path=os.path.join(path, "system", "blockMeshDict"), geometry=None)
    with open(os.path.join(path,"system", "blockMeshDict"),'r') as file:
         filedata= file.read()
    filedata = filedata.replace('scale   1','scale   0.01')
    with open(os.path.join(path,"system", "blockMeshDict"),'w') as file:
         file.write(filedata)
    os.system(path +"/Allrun.mesh")
    return 

# tube_rad = 0.5
# length = 60
# coil_rad = 3
# pitch = 3
# inversion_loc = 0.5

# create_mesh(coil_rad, tube_rad, pitch, length, inversion_loc, path='coil_basic')
