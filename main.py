from jax import grad, jit, value_and_grad
import jax.numpy as jnp
from utils import *
from utils_plotting import *
from utils_gp import *


def mfbo(f, data_path, x_bounds, z_bounds,time_budget,gamma=1.5, beta=2.5, p_c=2, gp_ms=16,opt_ms=32,sample_initial=True,int_fidelities=False):

    """
    # Multi-fidelity Bayesian Optimisation with stopping criteria and cost adjustment
    ## Inputs
    - f: function to be optimised
    - data_path: path to json file to save data to
    - x_bounds: bounds for the input space as a dictionary 
        - e.g. {"x1": [0, 1], "x2": [0, 1]}
    - z_bounds: bounds for the fidelity space
        - e.g. {"z1": [0, 1], "z2": [0, 1]}
    - time_budget: time budget for the optimisation in seconds 
    - gamma: gamma parameter for the cost-adjusted acquisition function
    - beta: beta parameter for the cost-adjusted acquisition function
    - p_c: p_c parameter for deciding termination
    - gp_ms: number of multi-starts to use when maximising NLL of GP
    - opt_ms: number of multi-starts to use when maximising acquisition function
    - sample_initial: number of initial samples to take, if False then will carry on from file
    - int_fidelities: whether to round fidelity values to nearest integer
    """


    # if carrying on from file
    if sample_initial == False:
        # read file
        data = read_json(data_path)
        # get last datapoint saved
        last_point = data['data'][-1]

        try:
            # if this is from optimisation (aka has a flag)
            flag = last_point['flag'] 
            # set time left to one of these
            if last_point['cost'] != 'running':
                time_left = last_point['time_left_at_end_of_iteration']
            else:
                time_left = last_point['time_left_at_beginning_of_iteration']
        except KeyError:
            # if last point is sampled
            # set time left to just the time budget stated
            time_left = time_budget
    else:
        time_left = time_budget

    # defining joint space bounds
    n_fid = len(z_bounds)
    joint_bounds = x_bounds | z_bounds
    x_bounds_og = x_bounds.copy()
    z_bounds_og = z_bounds.copy()
    joint_bounds_og = joint_bounds.copy()


    if sample_initial != False:
        # perform initial sample of joint space
        samples = sample_bounds(joint_bounds, sample_initial)
        print(samples)
        # intialise data json
        data = {"data": []}
        for sample in samples:
            # for each fidelity value make this an int
            for i in range(n_fid):
                if int_fidelities[i] == True:
                    sample[len(x_bounds)+i] = np.rint(sample[len(x_bounds)+i])
            
            # create sample dict for evaluation
            sample_dict = sample_to_dict(sample, joint_bounds)

            # preliminary run info 
            run_info = {
                "id": "running",
                "x": sample_dict,
                "cost": "running",
                "obj": "running",
            }
            data["data"].append(run_info)
            save_json(data, data_path)

            # perform function evaluation
            res = f(sample_dict)
            run_info = {
                "id": res["id"],
                "x": sample_dict,
                "cost": res["cost"],
                "obj": res["obj"],
            }
            data["data"][-1] = run_info
            # save to file
            save_json(data, data_path)

        for d in data['data']:
            print(d['x']['fid_radial'])
            print(d['x']['fid_axial'])
    data = read_json(data_path) 
    data['gamma'] = gamma
    data['beta'] = beta
    data['p_c'] = p_c
    data['gp_ms'] = gp_ms
    data['opt_ms'] = opt_ms
    save_json(data,data_path)
    

    while True:

        start_time = time.time()
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

        # normalise bounds 
        x_bounds = normalise_bounds_dict(x_bounds_og, x_mean, x_std)
        z_bounds = normalise_bounds_dict(z_bounds_og, z_mean, z_std)
        joint_bounds = normalise_bounds_dict(joint_bounds_og, j_mean, j_std)

        # training two Gaussian processes:
        print("Training GPs")
        # all inputs and fidelities against objective
        gp = build_gp_dict(*train_gp(inputs, outputs, gp_ms))
        # inputs and fidelities against cost
        cost_gp = build_gp_dict(*train_gp(inputs, cost, gp_ms))

        # optimising the aquisition of inputs, disregarding fidelity
        print("Optimising aquisition function")

        def optimise_aquisition(cost_gp, gp, ms_num, gamma, beta, cost_offset):
            # normalise bounds
            b_list = list(joint_bounds.values())
            fid_high = jnp.array([list(z_bounds.values())[i][1] for i in range(n_fid)])
            # sample and normalise initial guesses
            x_init = jnp.array(sample_bounds(joint_bounds, ms_num))
            f_best = 1e20
            # define grad and value for acquisition (jax)
            f = value_and_grad(aquisition_function)
            run_store = []

            # iterate over multistart
            for i in range(ms_num):
                x = x_init[i]
                res = minimize(
                    f,
                    x0=x,
                    args=(gp, cost_gp, fid_high, gamma, beta, cost_offset),
                    method="SLSQP",
                    bounds=b_list,
                    jac=True,
                    tol=1e-8,
                    options={"disp": True},
                )
                aq_val = res.fun
                x = res.x
                run_store.append(aq_val)
                # if this is the best, then store solution
                if aq_val < f_best:
                    f_best = aq_val
                    x_best = x
            # return best solution found
            return x_best, aq_val

        def optimise_greedy(gp, ms_num):
            # normalise bounds
            b_list = list(x_bounds.values())
            fid_high = jnp.array([list(z_bounds.values())[i][1] for i in range(n_fid)])
            # sample and normalise initial guesses
            x_init = jnp.array(sample_bounds(x_bounds, ms_num))
            f_best = 1e20
            # jax magic
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
                    tol=1e-8,
                    options={"disp": True},
                )
                aq_val = res.fun
                x = res.x
                run_store.append(aq_val)
                if aq_val < f_best:
                    f_best = aq_val
                    x_best = x

            # because this is only in x-space append highest fidelities 
            # for complete solution
            for i in range(len(fid_high)):
                x_best = np.append(x_best, fid_high[i])

            return x_best, aq_val



        # this is to ensure cost-adjusted acquisition can't be -inf
        cost_offset = min(cost) * 1.5

        # solve two problems
        x_opt, f_opt = optimise_aquisition(
            cost_gp, gp, opt_ms, gamma, beta, cost_offset
        )
        x_greedy, f_greedy = optimise_greedy(gp, opt_ms)


        # get predicted values for these solutions 
        mu_standard_obj, var_standard_obj = inference(gp, jnp.array([x_opt]))
        mu_greedy_obj, var_greedy_obj = inference(gp, jnp.array([x_greedy]))

        mu_standard, var_standard = inference(cost_gp, jnp.array([x_opt]))
        mu_greedy, var_greedy = inference(cost_gp, jnp.array([x_greedy]))

        # work out different predicted times for evaluation
        max_time_standard = mu_standard + p_c * np.sqrt(var_standard)
        max_time_greedy = mu_greedy + p_c * np.sqrt(var_greedy)
        # this is the time left normalised
        norm_time_left = (time_left - c_mean) / c_std

        flag = "NORMAL" # assume we do a normal evaluation

        # if there's not enough time left for both, do a greedy evaluation
        if max_time_standard + max_time_greedy > norm_time_left:
            x_opt = x_greedy
            flag = "GREEDY"

        # unnormalise solution ready for evaluation
        x_opt = list(unnormalise(x_opt, j_mean, j_std))
        x_opt = [np.float64(xi) for xi in x_opt]
        print("unnormalised res:", x_opt)

        for i in range(n_fid):
            if int_fidelities[i] == True:
                x_opt[len(x_bounds)+i] = int(x_opt[len(x_bounds)+i])

        sample = sample_to_dict(x_opt, joint_bounds)

        run_info = {
            "flag": flag,
            "id": "running",
            "x": sample,
            "cost": "running",
            "obj": "running",
            "joint_mean": list([np.float64(j_mean[i]) for i in range(len(j_mean))]),
            "joint_std": list([np.float64(j_std[i]) for i in range(len(j_std))]),
            "obj_mean": np.float64(o_mean[0]),
            "obj_std": np.float64(o_std[0]),
            "cost_mean": np.float64(c_mean[0]),
            "cost_std": np.float64(c_std[0]),
            "pred_s_obj_mean": np.float64(mu_standard_obj[0]),
            "pred_s_obj_std": np.float64(np.sqrt(var_standard_obj)[0, 0]),
            "pred_g_obj_mean": np.float64(mu_greedy_obj[0]),
            "pred_g_obj_std": np.float64(np.sqrt(var_greedy_obj)[0, 0]),
            "pred_s_cost_mean": np.float64(mu_standard[0]),
            "pred_s_cost_std": np.float64(np.sqrt(var_standard)[0, 0]),
            "pred_g_cost_mean": np.float64(mu_greedy[0]),
            "pred_g_cost_std": np.float64(np.sqrt(var_greedy)[0, 0]),
            "norm_time_left_at_beginning_of_iteration": np.float64(norm_time_left[0]),
            "time_left_at_beginning_of_iteration": np.float64(time_left),
        }
        data['data'].append(run_info)
        save_json(data,data_path)

        end_time = time.time()

        other_time = (end_time - start_time)

        # perform evaluation
        res = f(sample)
        for k,v in res.items():
            run_info[k] = v

        time_left = time_left - run_info["cost"] - other_time
        run_info["time_left_at_end_of_iteration"] = time_left

        # make last thing in data list this evaluation and not the placeholder
        data["data"][-1] = run_info

        # do all plotting you desire
        save_json(data, data_path)
        plot_results(data_path)
        plot_fidelities(data_path)
        plot_data_file(data_path)

