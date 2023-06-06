from utils import *
import matplotlib as mpl 

def plot_fidelities(path):
    data = read_json(path)
    z_vals = []
    c_vals = []
    n_init = 0 
    for d in data["data"]:
        try:
            flag = d['flag']
        except:
            n_init += 1
        if d["cost"] != "running":
            xv = d["x"]
            c_vals.append(d["cost"])
            zv = []
            for xk in list(xv.keys()):
                if xk.split("_")[0] == "fid":
                    zv.append(xv[xk])
            z_vals.append(zv)
    z_vals = np.array(z_vals)
    if len(z_vals[0]) != 2:
        print('Only 2 fidelities can be plotted...')
        return 


    c_vals = c_vals[n_init:]
    z_vals = z_vals[n_init:, :]


    color = cm.Spectral_r(c_vals)
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    fig.subplots_adjust(hspace=0.05)


    min_z = np.min(z_vals, axis=0)
    max_z = np.max(z_vals, axis=0)+1

    grid = np.zeros((int(max_z[0]-min_z[0]),int(max_z[1]-min_z[1])))

    ind_z = z_vals-min_z

    g_count = np.zeros_like(grid)
    for i in range(len(ind_z[:,0])):
        grid[int(ind_z[i,0]),int(ind_z[i,1])] += c_vals[i]
        g_count[int(ind_z[i,0]),int(ind_z[i,1])] += 1
    
    for i in range(len(grid[:,0])):
        for j in range(len(grid[0,:])):
            if g_count[i,j] > 0:
                grid[i,j] = grid[i,j]/g_count[i,j]
            else:
                grid[i,j] = np.nan
                g_count[i,j] = np.nan


    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="3%", pad=0.05)
    sc = axs[0].imshow(grid.T, cmap='Spectral_r', norm=colors.LogNorm(vmin=np.min(c_vals), vmax=np.max(c_vals)),origin='lower',aspect='auto')
    cb = fig.colorbar(sc, cax=cax, orientation="vertical")
    cax.tick_params(labelsize=14)
    cb.set_label(label="Simulation Cost (s)", size=14)

    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="3%", pad=0.05)
    sc_count = axs[1].imshow(g_count.T, cmap='Spectral_r',origin='lower',aspect='auto')
    cb = fig.colorbar(sc_count, cax=cax, orientation="vertical")
    cax.tick_params(labelsize=14)
    cb.set_label(label="Evaluations", size=14)
    min1 = np.min(z_vals[:,0])
    max1 = np.max(z_vals[:,0])
    min2 = np.min(z_vals[:,1])
    max2 = np.max(z_vals[:,1])

    for ax in axs:
        ax.set_ylabel("Radial Fidelity", fontsize=14)
        ax.set_xlabel("Axial Fidelity", fontsize=14)
        ax.set_xticks(range(int(max1-min1+1)), range(int(min1),int(max1)+1))
        ax.set_yticks(range(int(max2-min2+1)), range(int(min2),int(max2)+1))
        # ax.set_xticks(np.array([1,11,21,31]), np.array([1,11,21,31])+min_z[0])
    fig.tight_layout()
    # fig.subplots_adjust(right=1.05, left=0.075, top=0.95, bottom=0.15)
    # for i, c in zip(range(len(z_vals)-1), color):
    #     axs.plot([z_vals[i,0],z_vals[i+1,0]],[z_vals[i,1],z_vals[i+1,1]],lw=6,color=c,alpha=0.95)
    plt.savefig(path.split("data.json")[0] + "/fidelities.png", dpi=800)
    return


def plot_data_file(path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))

    all_data = read_json(path)["data"]
    data = []
    init_n = 0
    for d in all_data:
        if d["cost"] != "running":
            data.append(d)
            try:
                flag = d["flag"]
            except KeyError:
                init_n += 1
                pass

    data = data[init_n:]
    n = len(data)
    obj_mean = list_from_dict(data, "obj_mean")
    obj_std = list_from_dict(data, "obj_std")
    cost_mean = list_from_dict(data, "cost_mean")
    cost_std = list_from_dict(data, "cost_std")
    pred_s_obj_mean = list_from_dict(data, "pred_s_obj_mean")
    pred_s_cost_mean = list_from_dict(data, "pred_s_cost_mean")
    pred_g_cost_mean = list_from_dict(data, "pred_g_cost_mean")
    pred_s_cost_std = list_from_dict(data, "pred_s_cost_std")
    pred_g_cost_std = list_from_dict(data, "pred_g_cost_std")

    pred_s_obj_mean_unnorm = []
    pred_s_cost_mean_unnorm = []
    max_s = []
    max_g = []
    for i in range(n):
        pred_s_obj_mean_unnorm.append(
            unnormalise(pred_s_obj_mean[i], obj_mean[i], obj_std[i])
        )
        pred_s_cost_mean_unnorm.append(
            unnormalise(pred_s_cost_mean[i], cost_mean[i], cost_std[i])
        )
        max_s.append(
            unnormalise(
                pred_s_cost_mean[i] + 2 * pred_s_cost_std[i], cost_mean[i], cost_std[i]
            )
        )
        max_g.append(
            unnormalise(
                pred_g_cost_mean[i] + 2 * pred_g_cost_std[i], cost_mean[i], cost_std[i]
            )
        )

    obj = list_from_dict(data, "obj")
    cost = list_from_dict(data, "cost")
    time_left = list_from_dict(data, "time_left_at_beginning_of_iteration")

    ms = 30

    ax[0].scatter(
        np.arange(n), pred_s_obj_mean_unnorm, c="tab:red", s=ms, label="Predicted"
    )
    ax[0].scatter(np.arange(n), obj, c="k", s=ms, label="Evaluated")
    for i in range(n):
        ax[0].plot(
            [i, i], [pred_s_obj_mean_unnorm[i], obj[i]], c="k", lw=3, ls="dotted"
        )

    ax[1].scatter(
        np.arange(n), pred_s_cost_mean_unnorm, c="tab:red", s=ms, label="Predicted"
    )
    ax[1].scatter(np.arange(n), cost, c="k", s=ms, label="Evaluated")
    for i in range(n):
        ax[1].plot(
            [i, i], [pred_s_cost_mean_unnorm[i], cost[i]], c="k", lw=3, ls="dotted"
        )

    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Objective")
    ax[0].legend(frameon=False)

    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Simulation Cost (s)")
    ax[1].legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path.split("data.json")[0] + "/predicted_values.png", dpi=800)

    fig, ax = plt.subplots(1,2, figsize=(10, 3))
    max_s = np.array(max_s)
    max_g = np.array(max_g)
    time = np.cumsum(cost)

    lw = 2
    ms = 10
    ax[0].plot(np.arange(n), time_left, c="tab:red", lw=lw, label="Remaining")
    ax[0].plot(np.arange(n), max_s, c="k", lw=lw, label="Max Predicted: Standard")
    ax[0].scatter(np.arange(n), cost, c="k", s=ms, label="Actual")
    ax[0].plot(
        np.arange(n),
        max_s + max_g,
        c="tab:blue",
        lw=lw,
        label="Max Predicted: Standard + Greedy",
    )

    ax[1].plot(time, time_left, c="tab:red", lw=lw, label="Remaining")
    ax[1].plot(time, max_s, c="k", lw=lw, label="Max Predicted: Standard")
    ax[1].scatter(time, cost, c="k", s=ms, label="Actual")
    ax[1].plot(
        time,
        max_s + max_g,
        c="tab:blue",
        lw=lw,
        label="Max Predicted: Standard + Greedy",
    )

    for a in ax:
        a.set_yscale("log")
    ax[0].set_xlabel("Iteration",size=14)
    ax[0].set_ylabel("Time (s)",size=14)
    handles, labels = ax[0].get_legend_handles_labels()
    # fig.legend(handles,labels,frameon=False,bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",ncol=4)
    fig.legend(handles, labels, loc="upper center", frameon=False, ncol=4)
    #fig.subplots_adjust(top=0.875, bottom=0.15, left=0.075, right=0.975)
    ax[1].set_xlabel("Wall-clock time (s)",size=14)
    ax[1].set_ylabel("Time (s)",size=14)
    # plt.tight_layout()
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.2,top = 0.9)
    plt.savefig(path.split("data.json")[0] + "/time_remaining.png", dpi=800)

    return


def plot_results(path):
    data = read_json(path)
    data = data["data"]

    obj = []
    cost = []
    fid = []
    crit = []

    init_n = 0
    for d in data:
        if d["cost"] != "running":
            try:
                flag = d["flag"]
                obj.append(d["obj"])
                cost.append(d["cost"])
                crit.append(
                    unnormalise(d["pred_g_obj_mean"], d["obj_mean"], d["obj_std"])
                )
                x = d["x"]
                fid_vals = []
                for x_k in list(x.keys()):
                    if x_k.split("_")[0] == "fid":
                        fid_vals.append(x[x_k])
                fid.append(fid_vals)
            except KeyError:
                init_n += 1
                pass


    init_data = data[:init_n]
    init_obj = list_from_dict(init_data, "obj")
    init_cost = list_from_dict(init_data, "cost")

    it = np.arange(len(obj))

    time = np.cumsum(init_cost)
    init_time = time - time[-1]
    time = np.cumsum(cost)
    full_time = np.append(init_time, time)

    full_it = np.arange(-init_n, len(obj))
    cost = np.append(init_cost, cost, axis=0)
    obj = np.append(init_obj, obj, axis=0)
    obj = list((np.array(obj) - np.min(obj))/(np.max(obj) - np.min(obj)))



    b = 0
    best_obj = []
    for o in obj:
        if o > b:
            best_obj.append(o)
            b = o
        else:
            best_obj.append(b)
    b = 0
    best_crit = []
    for c in crit:
        if c > b:
            best_crit.append(c)
            b = c
        else:
            best_crit.append(b)
    lw = 3
    ms = 60
    m_alpha = 1
    mar = "o"
    grid_alpha = 0.0
    font_size = 14

    fig, ax = plt.subplots(2,1, figsize=(10, 5))

    for a in ax.ravel():
        # a.spines["right"].set_visible(False)
        # a.spines["top"].set_visible(False)
        a.tick_params(axis="both", which="major", labelsize=font_size - 2)
        #a.set_ylim(0, max(best_crit) + 1)

    from matplotlib import cm

    norm = colors.LogNorm(vmin=np.percentile(cost,5), vmax=np.percentile(cost,95))

    rgba_color = [np.array(cm.Spectral_r(norm(c), bytes=True)) / 255 for c in cost]
    # im1 = ax[0].scatter(
    #     full_it,
    #     obj,
    #     c=rgba_color,
    #     marker=mar,
    #     s=ms,
    #     lw=0,
    #     edgecolor="w",
    #     norm=norm,
    #     alpha=m_alpha,
    # )
    ax[0].plot(full_it, best_obj, c="k", lw=2, zorder=-1, label="Best")
    # ax[0,0].legend(frameon=False,fontsize=14)
    ax[0].set_xlabel(r"Iteration", fontsize=font_size)
    ax[0].set_ylabel(
        "Normalised \nObjective",

        fontsize=font_size,
    )
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Spectral_r), cax=cax, orientation="vertical")
    cb.outline.set_visible(False)
    cax.tick_params(labelsize=14)
    cb.set_label(label="Simulation Cost (s)", size=14)
    for i in range(len(full_it) - 1):
        ax[0].fill_between(
            [full_it[i], full_it[i + 1]],
            [obj[i], obj[i + 1]],
            [0, 0],
            alpha=1,
            color=rgba_color[i + 1],
            linewidth=0,
            zorder=-1,
        )
        ax[0].fill_between(
            [full_it[i], full_it[i + 1]],
            [best_obj[i], best_obj[i + 1]],
            [obj[i], obj[i + 1]],
            fc='k',
            lw=0,
            alpha=0.05,
            zorder=-1,
        )

    # ax[0, 1].scatter(
    #     np.arange(len(crit)), crit, c="k", marker=mar, s=ms, lw=0, alpha=m_alpha
    # )
    # ax[0, 1].plot(np.arange(len(crit)), best_crit, c="k", zorder=-1, lw=2, label="Best")
    # # ax[0,1].legend(frameon=False,fontsize=14)
    # ax[0, 1].set_xlabel(r"Iteration", fontsize=font_size)
    # ax[0, 1].set_ylabel(r"$\max_x \quad \mu_t(x,z^\bullet)$", fontsize=font_size)
    # for i in range(len(it) - 1):
    #     ax[0, 1].fill_between(
    #         [it[i], it[i + 1]],
    #         [best_crit[i], best_crit[i + 1]],
    #         [0, 0],
    #         color="k",
    #         alpha=0.1,
    #         lw=0,
    #         zorder=-1,
    #     )

    # im2 = ax[1].scatter(
    #     full_time,
    #     obj,
    #     c=rgba_color,
    #     marker=mar,
    #     s=ms,
    #     lw=0,
    #     edgecolor="w",
    #     norm=norm,
    #     alpha=m_alpha,
    # )
    ax[1].plot(full_time, best_obj, c="k", zorder=-1, lw=2, label="Best")
    # ax[1,0].legend(frameon=False,fontsize=14)
    ax[1].ticklabel_format(style="sci", axis="x", scilimits=(0, 4))

    ax[1].set_xlabel(r"Wall-clock time (s)", fontsize=font_size)
    ax[1].set_ylabel(
        "Normalised \nObjective",
        fontsize=font_size,
    )
    for i in range(len(full_it) - 1):
        ax[1].fill_between(
            [full_time[i], full_time[i + 1]],
            [obj[i], obj[i + 1]],
            [0, 0],
            alpha=1,
            color=rgba_color[i + 1],
            zorder=-1,
            linewidth=0,
        )
        ax[1].fill_between(
            [full_time[i], full_time[i + 1]],
            [best_obj[i], best_obj[i + 1]],
            [obj[i], obj[i + 1]],
            fc='k',
            lw=0,
            alpha=0.05,
            zorder=-1,
        )

    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Spectral_r), cax=cax, orientation="vertical")
    cb.outline.set_visible(False)
    cax.tick_params(labelsize=14)
    cb.set_label(label="Simulation Cost (s)", size=14)

    # ax[1, 1].scatter(time, crit, c="k", marker=mar, s=ms, lw=0, alpha=m_alpha)
    # ax[1, 1].plot(time, best_crit, c="k", zorder=-1, lw=2, label="Best")
    # # ax[1,1].legend(frameon=False,fontsize=14)
    # ax[1, 1].set_xlabel(r"Wall-clock time (s)", fontsize=font_size)
    # ax[1, 1].ticklabel_format(style="sci", axis="x", scilimits=(0, 4))
    # ax[1, 1].set_ylabel(r"$\max_x \quad \mu_t(x,z^\bullet)$", fontsize=font_size)
    # for i in range(len(it) - 1):
    #     ax[1, 1].fill_between(
    #         [time[i], time[i + 1]],
    #         [best_crit[i], best_crit[i + 1]],
    #         [0, 0],
    #         color="k",
    #         alpha=0.1,
    #         lw=0,
    #         zorder=-1,
    #     )

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.4)
    plt.savefig(path.split("data.json")[0] + "/res.png", dpi=800)

    return

# data_path = 'parameterisation_study/cross_section/data.json'
# plot_results(data_path)
# plot_fidelities(data_path)
# plot_data_file(data_path)