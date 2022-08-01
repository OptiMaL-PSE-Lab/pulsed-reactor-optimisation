from utils import eval_cfd_validation
import json
import os 
import numpy as np  
import pickle

N,time,value = eval_cfd_validation(0.002,5,50,0.012,0.01,0.0025,0.0753)

a = [N,time,value]
with open('simulation-integration/output_validation/experiment_one.pickle', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

N,time,value = eval_cfd_validation(0.002,4,50,0.012,0.01,0.0025,0.0753)

a = [N,time,value]
with open('simulation-integration/output_validation/experiment_two.pickle', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('simulation-integration/output_validation/experiment_one.pickle', 'rb') as handle:
    n,time,value = pickle.load( handle)

res = pd.DataFrame({'time':time,'concentration':value})

res.to_csv('simulation-integration/output_validation/experiment_one.csv')

with open('simulation-integration/output_validation/experiment_two.pickle', 'rb') as handle:
    n,time,value = pickle.load( handle)

res = pd.DataFrame({'time':time,'concentration':value})

res.to_csv('simulation-integration/output_validation/experiment_two.csv')


