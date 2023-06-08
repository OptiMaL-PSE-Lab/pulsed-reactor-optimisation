import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils import *
from utils_gp import * 
import matplotlib.pyplot as plt 
import pandas as pd 

def read_cum_cost(path):
	data = read_json(path)['data']
	d_list = []
	for d in data:
		if d['cost'] != 'running':
			d_list.append(d)
	cost = [d['cost'] for d in d_list]
	cum_cost = np.cumsum(cost)
	return cum_cost

def plot_cum_cost(path_names,fig_path):
	l_ls = ['solid','dashed','dotted']
	fig,ax=plt.subplots(1,1,figsize=(6,4))
	for i in range(len(path_names.keys())):
		key = list(path_names.keys())[i]
		cum_cost = read_cum_cost(path_names[key])/3600
		ax.plot(np.arange(len(cum_cost)),cum_cost,label=key,lw=2,ls=l_ls[i],c='k')
	ax.set_ylabel('Cumulative Time (hr)')
	ax.set_xlabel('Data Collected')
	# ax.grid(alpha=0.5)
	ax.legend(frameon=False)
	# #turn off right and top axis 
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	fig.savefig(fig_path,dpi=300,bbox_inches='tight')


def gen_data(f, data_path, x_bounds, fid, n_w):

        try:
             data = read_json(data_path)
        except FileNotFoundError:
                # defining joint space bounds
                x_bounds_og = x_bounds.copy()
                samples = sample_bounds(x_bounds, n_w) 
                data = {"data": []}
                for sample in samples:
                        # create sample dict for evaluation
                        sample_dict = sample_to_dict(sample, x_bounds)

                        # preliminary run info 
                        run_info = {
                        "id": "running",
                        "x": sample_dict,
                        "z": fid,
                        "cost": "running",
                        "obj": "running",
                        }
                        data["data"].append(run_info)
                save_json(data, data_path)

        for i in range(len(data["data"])):
                if data['data'][i]['id'] == 'running':
                        sample_dict = data['data'][i]['x']
                        fid = data['data'][i]['z']
                        # perform function evaluation
                        res = f(sample_dict,fid)
                        run_info = {
                        "id": res["id"],
                        "x": sample_dict,
                        "z": fid,
                        "cost": res["cost"],
                        "obj": res["obj"],
                        }
                        data["data"][i] = run_info
                        # save to file
                        save_json(data, data_path)
        return 



def gen_equation(path,inputs,outputs,keys):
	
	data_name = path.split('/')[-1]
	file_name = path.split(data_name)[0] + '/equation.csv'

	model = PySRRegressor(
		niterations=10000,
		binary_operators=["+", "*", "/", "-","^"],
		unary_operators=["exp","log","neg","square","cube","sin"],
		model_selection='best',
		maxsize = 40, 
		turbo=True,
		loss="L1DistLoss()",
		maxdepth = 20,
		timeout_in_seconds = 360,
		verbosity=0,
		parsimony = 1e-5,
		equation_file= file_name,
		progress=False
	)

	model.fit(inputs,outputs.reshape(-1,1),variable_names = keys)

	print(model.latex())
	def f(x):
		if x.__class__ != float or x.__class__ != int:
			return model.sympy()
		else:
			return model.sympy().evalf(x)
		
	return 


def add_dimensionless_numbers(x):
	dt = 0.0025
	st = x['coil_rad']/(2*np.pi*x['a'])
	re0 = (2*np.pi*x['f']*x['a']*990*dt)/(9.9*10**(-4))
	de = re0 / (np.sqrt(dt/(2*x['coil_rad'])))
	x['st'] = st
	x['de'] = de
	x['rey'] = x['re']
	return x 




def infer_hf_data(path,new_path):

    data = read_json(path)['data'][:80]

    inputs = np.array([list(d['x'].values()) for d in data])
    keys = list(data[0]['x'].keys())

    outputs = np.array([d['obj'] for d in data]).reshape(-1,1)

    m_inputs,std_inputs = mean_std(inputs)
    m_outputs,std_outputs = mean_std(outputs)

    outputs = normalise(outputs,m_outputs,std_outputs)
    inputs = normalise(inputs,m_inputs,std_inputs)

    gp = build_gp_dict(*train_gp(inputs,outputs,ms=1))

    n_sample = 5000
    bounds = np.array([[np.min(inputs[:,i]),np.max(inputs[:,i])] for i in range(inputs.shape[1])])

    x_sample = lhs(bounds, n_sample,log=False)
    new_data = []
    for i in range(len(x_sample)):
        x_sample[i,6] = np.max(x_sample[:,6])
        x_sample[i,7] = np.max(x_sample[:,7])
        # create dict with keys
        new_x = unnormalise(x_sample[i],m_inputs,std_inputs)
        new_x = dict(zip(keys,new_x))
        new_data.append({'x':new_x})

    new_data = {'data':new_data}

    mean = []
    cov = []
    for x in tqdm(x_sample):
        mean_v, cov_v = inference(gp, jnp.array([x]))
        mean.append(mean_v)
        cov.append(cov_v)


    for i in range(len(x_sample)):
        new_data['data'][i]['mean'] = float(unnormalise(float(mean[i]),m_outputs,std_outputs))
        new_data['data'][i]['std'] = float(unnormalise(float(np.sqrt(cov[i])),m_outputs,std_outputs))
        new_data['data'][i]['obj'] = float(np.random.normal(new_data['data'][i]['mean'],new_data['data'][i]['std']))

    save_json(new_data,'symbolic_mf_data_generation/exp_design/interp_data_dist.json')

    new_data = {'data':new_data}
    for i in range(len(x_sample)):
        new_data['data'][i]['mean'] = float(unnormalise(float(mean[i]),m_outputs,std_outputs))
        new_data['data'][i]['std'] = float(unnormalise(float(np.sqrt(cov[i])),m_outputs,std_outputs))
        new_data['data'][i]['obj'] =  float(unnormalise(float(mean[i]),m_outputs,std_outputs))

    save_json(new_data,new_path)
    return 

def plot_symbolic_expression(csv_path,output_path):
    path = csv_path
    name = 'mf_mean'
    df = pd.read_csv(path)
    # plot score against complexity 

    com = df['Complexity'].values
    loss = df['Loss'].values
    eqs = df['Equation'].values

    n_com = (com - min(com))/(max(com)-min(com))
    n_loss = (loss - min(loss))/(max(loss)-min(loss))
    dists = np.sqrt(n_com**2 + n_loss**2)
    p_opt = np.argmin(dists)


    fig,ax=plt.subplots(1,1,figsize=(6,4))
    ax.plot(df['Complexity'],df['Loss'],lw=2,c='k')
    ax.scatter(com[p_opt],loss[p_opt],s=100,c='tab:red',label='Pareto Optimal',zorder=0)
    ax.text(com[p_opt]+0.1,loss[p_opt]+0.1,r'$((\sin(St)-a/0.0317)^2+1.74)^3$',fontsize=12)
    ax.set_ylabel('Loss')
    ax.set_xlabel('Complexity')
    ax.grid(alpha=0.5)
    ax.legend(frameon=False)
    fig.savefig(output_path,dpi=300,bbox_inches='tight')

