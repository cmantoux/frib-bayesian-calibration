import numpy as np
from math import sqrt

from flame import Machine

from flame_utils import generate_latfile
from flame_utils import BeamState
from flame_utils import ModelFlame

import logging
logging.getLogger('flame.machine').disabled = True

import ray

latfile = 'fitSolscan_20180430.lat'
fm = ModelFlame(latfile)
sols = fm.get_element(type='solenoid')
r, s = fm.run(monitor=-1)
refbg = s.bg[0]
refbg
[s.moment0_rms[0],s.moment0_rms[2],s.moment1[0,2,0]/s.moment0_rms[0]/s.moment0_rms[2]]

def get_m1_mat(emitx, betax, alphax, emity, betay, alphay, cxy, cxyp, cyxp, cxpyp):
    emitx=emitx/1e6
    emity=emity/1e6
    m1_mat = np.eye(4)
    gammax = (1 + alphax * alphax) / betax
    m1_mat[0, 0] = emitx * betax  * 1e6
    bs_x = sqrt(m1_mat[0, 0])
    m1_mat[0, 1] = -(emitx * alphax) * 1e3
    m1_mat[1, 0] = -(emitx * alphax) * 1e3
    m1_mat[1, 1] = emitx * gammax
    bs_xp = sqrt(m1_mat[1, 1])
    
    gammay = (1 + alphay * alphay) / betay
    m1_mat[2, 2] = emity * betay *1e6
    bs_y = sqrt(m1_mat[2, 2])
    m1_mat[2, 3] = -(emity * alphay) * 1e3
    m1_mat[3, 2] = -(emity * alphay) * 1e3
    m1_mat[3, 3] = emity * gammay
    bs_yp = sqrt(m1_mat[3, 3])
    
    m1_mat[2, 0] = cxy * bs_x * bs_y; m1_mat[0, 2] = m1_mat[2, 0]
    m1_mat[2, 1] = cyxp *bs_xp * bs_y; m1_mat[1, 2] = m1_mat[2, 1]
    m1_mat[3, 0] = cxyp *bs_x * bs_yp; m1_mat[0, 3] = m1_mat[3, 0]
    m1_mat[3, 1] = cxpyp *bs_xp * bs_yp; m1_mat[1, 3]= m1_mat[3, 1]
    
    return m1_mat

sols = fm.get_element(type='solenoid', name='FE_LEBT:SOLR_D0951')
_,s = fm.run(monitor=-1)
def model(initials, bfield):
    bary_size = np.zeros((7,7,1))
    bary_size[:4,:4,0] = get_m1_mat(*initials)
                                  
    #fm = ModelFlame(latfile)
    fm.bmstate.moment1 = bary_size

    # apply new configuration for the first three quads
    
    
    sols[0]['properties']['B'] = bfield
    fm.configure(sols)
 
    # simulate with updated model
    _, s = fm.run(monitor=-1)
    return [s.moment0_rms[0],s.moment0_rms[2],s.moment1[0,2,0]/s.moment0_rms[0]/s.moment0_rms[2]]

@ray.remote
def ray_model(theta, b):
    return model(theta, b)

def ray_log_likelihood(theta, delta, data):
    temp=[ray_model.remote(theta, line[0]) for line in data]
    predicted_data=np.vstack(ray.get(temp))    
    diff=predicted_data-data[:,-3:]    
    return -np.sum(diff*diff/2.0/np.array(delta)/np.array(delta))-len(diff)*np.sum(np.log(delta))