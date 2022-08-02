from utils import eval_cfd_validation
import pandas as pd 

for f in range[1,2,3,4]:
    N,time,value = eval_cfd_validation(0.002,5,50,0.012,0.01,0.0025,0.0753,f)
    res = pd.DataFrame({'time':time,'concentration':value})
    res.to_csv('simulation-integration/output_validation/e_1_fid_1'+str(f)+'.csv')

    N,time,value = eval_cfd_validation(0.004,5,50,0.012,0.01,0.0025,0.0753,f)
    res = pd.DataFrame({'time':time,'concentration':value})
    res.to_csv('simulation-integration/output_validation/e_2_fid_'+str(f)+'.csv')