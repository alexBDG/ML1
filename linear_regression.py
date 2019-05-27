import time
import numpy as np



###############################################################################
# close-form solution
###############################################################################


def direct_linalg(A,M,b):
    start = time.time()
    x = np.dot( np.dot( np.linalg.inv( A ) , M.transpose() ) , b )
    duration = time.time() - start
    return(x,duration)


###############################################################################
# gradient descent with a constant learning rate
###############################################################################


def const_grad(A,x,b,eps):
    V_err = []
    start = time.time()
    i = 0
    error = 1
    while (error > eps) and (i < 1e6):
        alpha = 0.0001
        x_1 = x
        x = x - 2*alpha*( np.dot(A,x) - b )
        error = np.linalg.norm( x - x_1 )
        i+=1
        V_err += [error]
    if(error>eps):
        print("Didn't converge after ",i," iterations",error)
    duration = time.time() - start
    return(x,duration,V_err)
        
    
###############################################################################
# gradient descent
###############################################################################
    
    
def step_decr_grad(A,x,b,eps):
    V_err = []
    start = time.time()
    i = 0
    error = 1
    while (error > eps) and (i < 1e6):
        alpha = 0.0001 / (1+i)
        x_1 = x
        x = x - 2*alpha*( np.dot(A,x) - b )
        error = np.linalg.norm( x - x_1 )
        i+=1
        V_err += [error]
    if(error>eps):
        print("Didn't converge after ",i," iterations",error)
    duration = time.time() - start
    return(x,duration,V_err)
    
    
###############################################################################
# gradient descent with adaptativ learning rate
###############################################################################

    
def step_decr_grad_adapt(A,x,b,eps,beta):
    V_err = []
    start = time.time()
    i = 0
    error = 1
    while (error > eps) and (i < 1e6):
        alpha = beta / (1+i)
        x_1 = x
        x = x - 2*alpha*( np.dot(A,x) - b )
        error = np.linalg.norm( x - x_1 )
        i+=1
        V_err += [error]
    if(error>eps):
        print("Didn't converge after ",i," iterations",error)
    duration = time.time() - start
    return(x,duration,V_err)
    

###############################################################################
# conjuguate gradient
###############################################################################
    
    
def conj_grad(A,x,b,eps):
    V_err = []
    start = time.time()
    k=0
    r = b-np.dot(A,x)
    d = r
    r_norm = np.linalg.norm(r)
    while((k<1e6) and (r_norm>eps)):
        z = np.dot(A,d)
        alpha = np.dot(r,r)/(np.dot(z,d))
        x = x + alpha*d
        rm = r
        r = r - alpha*z
        r_norm = np.linalg.norm(r)
        beta = np.dot(r,r)/np.dot(rm,rm)
        d = r + beta*d
        k+=1
        V_err += [r_norm]
    if(r_norm>eps):
        print("Didn't converge after ",k," iterations",r_norm)
    duration = time.time() - start
    return(x,duration,V_err)
  

###############################################################################
# optimal gradient
###############################################################################  
  
  
def grad(A,x,b,eps):
    V_err = []
    start = time.time()
    k=0
    r = b-np.dot(A,x)
    r_norm = np.linalg.norm(r)
    while((k<1e6) and (r_norm>eps)):
        z = np.dot(A,r)
        pas = np.dot(r,r)/(np.dot(z,r))
        x = x + pas*r
        r = b-np.dot(A,x)
        r_norm = np.linalg.norm(r) 
        k+=1
        V_err += [r_norm]
    if(r_norm>eps):
        print("Didn't converge after ",k," iterations",r_norm)
    duration = time.time() - start
    return(x,duration,V_err)
