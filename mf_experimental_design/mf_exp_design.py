
from jax import grad, jit, value_and_grad
import jax.numpy as jnp
from utils import *









n = 25

x_bounds = {}
x_bounds["a"] = [0.001, 0.008]
x_bounds["f"] = [2, 8]
x_bounds["re"] = [10, 50]
x_bounds["pitch"] = [0.0075, 0.02]
x_bounds["coil_rad"] = [0.0035, 0.0125]
x_bounds["inversion_loc"] = [0, 1]

z_bounds = {}
z_bounds["fid_axial"] = [20.001, 49.99]
z_bounds["fid_radial"] = [1.001, 5.99]
n_fid = len(z_bounds)

joint_bounds = x_bounds | z_bounds

x_bounds_og = x_bounds.copy()
z_bounds_og = z_bounds.copy()
joint_bounds_og = joint_bounds.copy()

# samples = sample_bounds(joint_bounds,n)
# data_path = 'outputs/mf/data.json'
# data = {'data':[]}
# for sample in samples:
#         for i in range(n_fid):
#                 sample[-(i+1)] = int(sample[-(i+1)])
#         sample_dict = sample_to_dict(sample,joint_bounds)
#         run_info = {'id':'running','x':sample_dict,'cost':'running','obj':'running'}
#         data['data'].append(run_info)
#         save_json(data,data_path)
#         res = eval_cfd(sample_dict)
#         run_info = {'id':res['id'],'x':sample_dict,'cost':res['cost'],'obj':res['obj']}
#         data['data'][-1] = run_info
#         save_json(data,data_path)


data_path = "outputs/mf/data.json"
data = read_json(data_path)

time_left = 72 * 60 * 60

while True:
    # reading data from file format
    data = read_json(data_path)


    inputs, outputs, cost = format_data(data)

    # normalising all data
    j_mean, j_std = mean_std(inputs)
    inputs = normalise(inputs, j_mean, j_std)
    x_mean = j_mean[:-n_fid]
    x_std = j_std[:-n_fid]
    z_mean = j_mean[-n_fid:]
    z_std = j_std[-n_fid:]

    o_mean, o_std = mean_std(outputs)
    outputs = normalise(outputs, o_mean, o_std)
    c_mean, c_std = mean_std(cost)
    cost = normalise(cost, c_mean, c_std)

    x_bounds = normalise_bounds_dict(x_bounds_og, x_mean, x_std)
    z_bounds = normalise_bounds_dict(z_bounds_og, z_mean, z_std)
    joint_bounds = normalise_bounds_dict(joint_bounds_og, j_mean, j_std)

    # training two Gaussian processes:
    print("Training GPs")
    # all inputs and fidelities against objective
    gp = build_gp_dict(*train_gp(inputs, outputs))
    # inputs and fidelities against cost
    cost_gp = build_gp_dict(*train_gp(inputs, cost))

    # optimising the aquisition of inputs, disregarding fidelity
    print("optimising aquisition function")

    gamma = 1.5
    beta = 2.5

    def optimise_aquisition(cost_gp, gp, ms_num, gamma, beta):
        # normalise bounds
        b_list = list(joint_bounds.values())
        fid_high = jnp.array([list(z_bounds.values())[i][1] for i in range(n_fid)])
        # sample and normalise initial guesses
        x_init = jnp.array(sample_bounds(joint_bounds, ms_num))
        f_best = 1e20
        f = value_and_grad(aquisition_function)
        run_store = []
        for i in range(ms_num):
            x = x_init[i]
            res = minimize(
                f,
                x0=x,
                args=(gp, cost_gp, fid_high, gamma, beta),
                method="SLSQP",
                bounds=b_list,
                jac=True,
                options={"disp": False},
            )
            aq_val = res.fun
            x = res.x
            run_store.append(aq_val)
            if aq_val < f_best:
                f_best = aq_val
                x_best = x
        return x_best, aq_val
    
    def optimise_greedy(gp, ms_num):
        # normalise bounds
        b_list = list(x_bounds.values())
        fid_high = jnp.array([list(z_bounds.values())[i][1] for i in range(n_fid)])
        # sample and normalise initial guesses
        x_init = jnp.array(sample_bounds(x_bounds, ms_num))
        f_best = 1e20
        f = value_and_grad(greedy_function)
        run_store = []
        for i in range(ms_num):
            x = x_init[i]
            res = minimize(
                f,
                x0=x,
                args=(gp, fid_high),
                method="SLSQP",
                bounds=b_list,
                jac=True,
                options={"disp": False},
            )
            aq_val = res.fun
            x = res.x
            run_store.append(aq_val)
            if aq_val < f_best:
                f_best = aq_val
                x_best = x
        for i in range(len(fid_high)):
            x_best = np.append(x_best,fid_high[i])

        return x_best, aq_val  
    
    multistart = 1
    
    x_opt, f_opt = optimise_aquisition(cost_gp, gp, multistart, gamma, beta)
    x_greedy,f_greedy = optimise_greedy(gp,multistart)


    mu_standard_obj,var_standard_obj = inference(gp,jnp.array([x_opt]))
    mu_greedy_obj,var_greedy_obj = inference(gp,jnp.array([x_greedy]))

    print("normalised res:", x_opt)

    mu_standard,var_standard = inference(cost_gp,jnp.array([x_opt]))
    mu_greedy,var_greedy = inference(cost_gp,jnp.array([x_greedy]))

    p_c = 2

    max_time_standard = mu_standard + p_c * np.sqrt(var_standard)
    max_time_greedy = mu_greedy + p_c * np.sqrt(var_greedy)

    norm_time_left = (time_left-c_mean)/c_std

    flag = 'NORMAL'
    if max_time_standard + max_time_greedy > norm_time_left:
        x_opt = x_greedy
        flag = "GREEDY"


    x_opt = list(unnormalise(x_opt, j_mean, j_std))
    x_opt = [np.float64(xi) for xi in x_opt]
    print("unnormalised res:", x_opt)

    for i in range(n_fid):
        x_opt[-(i + 1)] = int(x_opt[-(i + 1)])


    sample = sample_to_dict(x_opt, joint_bounds)

    print("Running ", sample)
    run_info = {
        "flag":flag,
        "id": "running",
        "x": sample,
        "cost": 1000 + np.random.uniform(),
        "obj": 6 + np.random.uniform(),
        "joint_mean":list([np.float64(j_mean[i]) for i in range(len(j_mean))]),
        "joint_std":list([np.float64(j_std[i]) for i in range(len(j_std))]),
        "obj_mean":np.float64(o_mean[0]),
        "obj_std":np.float64(o_std[0]),
        "cost_mean":np.float64(c_mean[0]),
        "cost_std":np.float64(c_std[0]),
        "pred_s_obj_mean": np.float64(mu_standard_obj[0]),
        "pred_s_obj_std": np.float64(np.sqrt(var_standard_obj)[0,0]),
        "pred_g_obj_mean": np.float64(mu_greedy_obj[0]),
        "pred_g_obj_std": np.float64(np.sqrt(var_greedy_obj)[0,0]),
        "pred_s_cost_mean": np.float64(mu_standard[0]),
        "pred_s_cost_std": np.float64(np.sqrt(var_standard)[0,0]),
        "pred_g_cost_mean": np.float64(mu_greedy[0]),
        "pred_g_cost_std": np.float64(np.sqrt(var_greedy)[0,0]),
        "norm_time_left_at_beginning_of_iteration": np.float64(norm_time_left[0]),
        "time_left_at_beginning_of_iteration":np.float64(time_left)
    }

    data['data'].append(run_info)
    save_json(data,data_path)

    #res = eval_cfd(sample)
    res = {'id':'test','cost':5000*np.random.uniform(),"obj":30*np.random.uniform()}
    run_info['id'] = res['id']
    run_info['cost'] = res['cost']
    run_info['obj'] = res['obj']

    data["data"][-1] = run_info

    time_left = time_left - run_info['cost']

    save_json(data, data_path)

    plot_results(data,n)
    plot_fidelities(data)
