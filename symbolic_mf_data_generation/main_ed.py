
import sys 
import os 
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from jax import grad, jit, value_and_grad
import jax.numpy as jnp
from utils import *
from utils_plotting import *
from utils_gp import *
from tqdm import tqdm 

def mfed(f, data_path, x_bounds, z_bounds,time_budget,gamma=1.5, gp_ms=4,ms_num=4,sample_initial=True,int_fidelities=False,eval_error=True):

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

    data = read_json(data_path) 
    data['gamma'] = gamma
    data['gp_ms'] = gp_ms
    save_json(data,data_path)
    
    iteration = len(data['data'])-1
    while iteration < time_budget:

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
        cost_gp = build_gp_dict(*train_gp(inputs, cost, gp_ms))

        if eval_error == True:
            n_test = 500
            x_test = sample_bounds(x_bounds_og,n_test)
            y_true = []
            y_test = []
            print('Evaluting model (never use this for an actual problem)')
            for x in tqdm(x_test):
                x_cond = {}
                x_keys = list(x_bounds_og.keys())
                for i in range(len(x_keys)):
                    x_cond[x_keys[i]] = x[i]
                for k,v in z_bounds_og.items():
                    x_cond[k] = v[1]
                y_true.append(f(x_cond)['obj'])

                x_cond_vec = list(x_cond.values())
                x_cond_vec = normalise(np.array(x_cond_vec),j_mean,j_std)
                m,v = inference(gp, jnp.array([x_cond_vec]))
                y_test.append(m)
            y_test = unnormalise(np.array(y_test),o_mean,o_std)
            error = 0 
            for i in range(n_test):
                error += (y_test[i] - y_true[i])**2
            error /= n_test
            error = error[0] 




        # if len(x_mean) == 1:
        #     xk = list(x_bounds.keys())[0]
        #     zk = list(z_bounds.keys())[0]
        #     x_sample = np.linspace(x_bounds[xk][0],x_bounds[xk][1], 100)
        #     mean = []
        #     cov = []
        #     for x in (x_sample):
        #         conditioned_sample = jnp.array([[x,z_bounds[zk][1]]])
        #         mean_v, cov_v = inference(gp, conditioned_sample)
        #         mean.append(mean_v)
        #         cov.append(cov_v)

        #     y = []
        #     c = []
        #     x = np.linspace(x_bounds_og[xk][0],x_bounds_og[xk][1], 100)
        #     x_sample = {}
        #     for xi in x:
        #             x_sample[xk] = xi
        #             x_sample[zk] = z_bounds_og[zk][1]
        #             e = f(x_sample)
        #             y.append(e['obj'])
        #             c.append(e['cost'])
        #     mean = unnormalise(np.array(mean),o_mean,o_std)[:,0]
        #     var = unnormalise(np.sqrt(np.array(cov)),o_mean,o_std)[:,0,0]
        #     plt.figure()
        #     plt.plot(x,y,c='k',lw=3,label='Highest Fidelity Function')
        #     plt.title('Current MSE: '+str(error))
        #     plt.plot(x,mean,c='k',ls='--',lw=3,label='Highest Fidelity Model')
        #     plt.fill_between(x,mean+var,mean-var,alpha=0.1,color='k',lw=0,label='Model Variance')
        #     plt.legend()
        #     plt.savefig('symbolic_mf_data_generation/toy/1d_vis/iteration_'+str(iteration)+'.png')

        iteration += 1
        # optimising the aquisition of inputs, disregarding fidelity
        print("Optimising aquisition function")

        def optimise_aquisition(cost_gp, gp, ms_num, gamma, cost_offset):
            # normalise bounds
            b_list = list(joint_bounds.values())
            fid_high = jnp.array([list(z_bounds.values())[i][1] for i in range(n_fid)])
            # sample and normalise initial guesses
            x_init = jnp.array(sample_bounds(joint_bounds, ms_num))
            f_best = 1e20
            # define grad and value for acquisition (jax)
            f = value_and_grad(exp_design_function)
            run_store = []

            # iterate over multistart
            for i in range(ms_num):
                x = x_init[i]
                res = minimize(
                    f,
                    x0=x,
                    args=(gp, cost_gp, fid_high, gamma, cost_offset),
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

        # this is to ensure cost-adjusted acquisition can't be -inf
        cost_offset = min(cost) * 1.5

        x_opt,_ = optimise_aquisition(cost_gp, gp, ms_num, gamma, cost_offset)
        # get predicted values for these solutions 
        mu_standard_obj, var_standard_obj = inference(gp, jnp.array([x_opt]))

        mu_standard, var_standard = inference(cost_gp, jnp.array([x_opt]))
        # this is the time left normalised
        norm_time_left = (time_left - c_mean) / c_std

        flag = "NORMAL" # assume we do a normal evaluation

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
            "pred_s_cost_mean": np.float64(mu_standard[0]),
            "pred_s_cost_std": np.float64(np.sqrt(var_standard)[0, 0]),
            "norm_time_left_at_beginning_of_iteration": np.float64(norm_time_left[0]),
            "time_left_at_beginning_of_iteration": np.float64(time_left),
        }
        try:
            run_info['MSE'] = np.float64(error)
        except:
            print('No Error Calculation')

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
 



def ed_hf(f, data_path, x_bounds, z_bounds,time_budget,sample_initial=True,gp_ms=4,ms_num=4,gamma=1.5,int_fidelities=False,eval_error=True):

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
        samples = sample_bounds(x_bounds, sample_initial)
        # intialise data json
        data = {"data": []}
        for sample in samples:
            # create sample dict for evaluation
            sample_dict = sample_to_dict(sample, joint_bounds)
            for zk,zv in z_bounds.items():
                sample_dict[zk] = zv[1]

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

    data = read_json(data_path) 
    data['gamma'] = gamma
    data['gp_ms'] = gp_ms
    save_json(data,data_path)
    
    iteration = len(data['data'])-1
    while iteration < time_budget:

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
        gp = build_gp_dict(*train_gp(inputs[:,:-n_fid], outputs, gp_ms))

        if eval_error == True:
            n_test = 500
            x_test = sample_bounds(x_bounds_og,n_test)
            y_true = []
            y_test = []
            print('Evaluting model (never use this for an actual problem)')
            for x in tqdm(x_test):
                x_cond = {}
                x_keys = list(x_bounds_og.keys())
                for i in range(len(x_keys)):
                    x_cond[x_keys[i]] = x[i]
                for k,v in z_bounds_og.items():
                    x_cond[k] = v[1]
                y_true.append(f(x_cond)['obj'])
                x_cond_vec = list(x_cond.values())[:-n_fid]
                x_cond_vec = normalise(np.array(x_cond_vec),x_mean,x_std)
                m,v = inference(gp, jnp.array([x_cond_vec]))
                y_test.append(m)
            y_test = unnormalise(np.array(y_test),o_mean,o_std)
            error = 0 
            for i in range(n_test):
                error += (y_test[i] - y_true[i])**2
            error /= n_test
            error = error[0] 




        # if len(x_mean) == 1:
        #     xk = list(x_bounds.keys())[0]
        #     x_sample = np.linspace(x_bounds[xk][0],x_bounds[xk][1], 100)
        #     mean = []
        #     cov = []
        #     for x in (x_sample):
        #         conditioned_sample = jnp.array([[x]])
        #         mean_v, cov_v = inference(gp, conditioned_sample)
        #         mean.append(mean_v)
        #         cov.append(cov_v)

        #     y = []
        #     c = []
        #     x = np.linspace(x_bounds_og[xk][0],x_bounds_og[xk][1], 100)
        #     x_sample = {}
        #     for xi in x:
        #             x_sample[xk] = xi
        #             for k,v in z_bounds_og.items():
        #                 x_sample[k] = v[1]
        #             e = f(x_sample)
        #             y.append(e['obj'])
        #             c.append(e['cost'])
        #     mean = unnormalise(np.array(mean),o_mean,o_std)[:,0]
        #     var = unnormalise(np.sqrt(np.array(cov)),o_mean,o_std)[:,0,0]
        #     plt.figure()
        #     plt.plot(x,y,c='k',lw=3,label='Highest Fidelity Function')
        #     plt.title('Current MSE: '+str(error))
        #     plt.plot(x,mean,c='k',ls='--',lw=3,label='Highest Fidelity Model')
        #     plt.fill_between(x,mean+var,mean-var,alpha=0.1,color='k',lw=0,label='Model Variance')
        #     plt.legend()
        #     plt.savefig('symbolic_mf_data_generation/toy/1d_vis_high/iteration_'+str(iteration)+'.png')
        iteration += 1

        # optimising the aquisition of inputs, disregarding fidelity
        print("Optimising aquisition function")

        def optimise_aquisition(gp, ms_num):
            # normalise bounds
            b_list = list(joint_bounds.values())
            # sample and normalise initial guesses
            x_init = jnp.array(sample_bounds(joint_bounds, ms_num))
            f_best = 1e20
            # define grad and value for acquisition (jax)
            f = value_and_grad(exp_design_hf)
            run_store = []

            # iterate over multistart
            for i in range(ms_num):
                x = x_init[i]
                res = minimize(
                    f,
                    x0=x,
                    args=(gp),
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

        # this is to ensure cost-adjusted acquisition can't be -inf

        x_opt,_ = optimise_aquisition( gp, ms_num)
        # get predicted values for these solutions 
        mu_standard_obj, var_standard_obj = inference(gp, jnp.array([x_opt]))

        # this is the time left normalised
        norm_time_left = (time_left - c_mean) / c_std

        flag = "NORMAL" # assume we do a normal evaluation

        # unnormalise solution ready for evaluation
        x_opt = list(unnormalise(x_opt, j_mean[:-n_fid], j_std[:-n_fid]))
        x_opt = [np.float64(xi) for xi in x_opt]
        print("unnormalised res:", x_opt)

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
            "norm_time_left_at_beginning_of_iteration": np.float64(norm_time_left[0]),
            "time_left_at_beginning_of_iteration": np.float64(time_left),
        }
        try:
            run_info['MSE'] = np.float64(error)
        except:
            print('No Error Calculation')

        data['data'].append(run_info)
        save_json(data,data_path)

        end_time = time.time()

        other_time = (end_time - start_time)

        # perform evaluation
        for zk,zv in z_bounds_og.items():
            sample[zk] = zv[1]
        print(sample)
        res = f(sample)
        for k,v in res.items():
            run_info[k] = v

        time_left = time_left - run_info["cost"] - other_time
        run_info["time_left_at_end_of_iteration"] = time_left

        # make last thing in data list this evaluation and not the placeholder
        data["data"][-1] = run_info

        # do all plotting you desire
        save_json(data, data_path)
 

