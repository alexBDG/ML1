import data_processing as dp
import linear_regression as lr

import numpy as np
import matplotlib.pyplot as plt
import json # we need to use the JSON package to load the data, since the data is stored in JSON format

with open("proj1_data.json") as fp:
    data = json.load(fp)


###############################################################################
# Execution
###############################################################################
nbw = 60


def SaveWords(data):
    dp.savewords(data[:10000],160)
    print("file created")
    
#SaveWords(data)


def Executeall(nbw):    
    hf_words = dp.find_frequent_words(data[:10000],nbw)
    print("Most frequent ",nbw," words : ", hf_words)
    (X,Y,XTX,XTY) = dp.build_system(data,0,10000,nbw,hf_words)
    
    epsilon = 10**(-10)
    (Wf,tf) = lr.direct_linalg(XTX,X,Y)
    (WCG,tCG,errCG) = lr.conj_grad(XTX,np.zeros(len(X[0])),XTY,epsilon)
    (WG,tG,errG) = lr.grad(XTX,np.zeros(len(X[0])),XTY,epsilon)
    (W1,t1,err1) = lr.const_grad(XTX,np.zeros(len(X[0])),XTY,epsilon)
    (W2,t2,err2) = lr.step_decr_grad(XTX,np.zeros(len(X[0])),XTY,epsilon)
    
    return (X,Wf,WCG,WG,W1,W2,tf,tCG,tG,t1,t2,errCG,errG,err1,err2)
        

###############################################################################
# To plot the MSE function of the complexity, number of words
###############################################################################
    
    
def PlotMultiMSE():
    epsilon = 10**(-10)
    av=[]
    for nbw in range(161):
        hf_words = dp.find_frequent_words(data[:10000],nbw)
        print("Most frequent ",nbw," words : ", hf_words)
        (X,Y,XTX,XTY) = dp.build_system(data,0,10000,nbw,hf_words)
        (WCG,tCG,errCG) = lr.conj_grad(XTX,np.zeros(len(X[0])),XTY,epsilon)
        (X,Y,XTX,XTY) = dp.build_system(data,10000,11000,nbw,hf_words) #if validation
#        (X,Y,XTX,XTY) = dp.build_system(data,0,10000,nbw,hf_words) #if training
#        (X,Y,XTX,XTY) = dp.build_system(data,11000,12000,nbw,hf_words) #if testing
        
        average = 0
        
        YCG = np.dot(X,WCG)
        n=len(YCG)
        for i in range(0,n):
            err = (Y[i]-YCG[i])**2
            average += err
        av += [average/n]
    
    fig = plt.figure()
    plt.xlabel('Number of frequent words used')
    plt.ylabel('Mean Square Error (validation data)') #if validation
#    plt.xlabel('Mean Square Error (training data)') #if training
#    plt.xlabel('Mean Square Error (testing data)') #if testing
    sol = [ k for k in range(len(av)) ]
    plt.plot(sol,av,label="MSE",color='red')
    plt.legend()
    plt.show()

#PlotMultiMSE()


###############################################################################
# To plot the MSE 
###############################################################################
    
    
def PlotMSE():
    epsilon = 10**(-10)
    
    hf_words = dp.find_frequent_words(data[:10000],nbw)
    (X,Y,XTX,XTY) = dp.build_system2(data,0,10000,nbw,hf_words)
#    (WCG,tCG) = lr.direct_linalg(XTX,X,Y) #closed-form
    (WCG,tCG,errCG) = lr.conj_grad(XTX,np.zeros(len(X[0])),XTY,epsilon)
#    (X,Y,XTX,XTY) = dp.build_system2(data,10000,11000,nbw,hf_words) #if validation
#    (X,Y,XTX,XTY) = dp.build_system2(data,0,10000,nbw,hf_words) #if training
    (X,Y,XTX,XTY) = dp.build_system2(data,11000,12000,nbw,hf_words) #if testing
    
    V_err = []
    average = 0
    
    YCG = np.dot(X,WCG)
    n=len(YCG)
    for i in range(0,n):
        err = (Y[i]-YCG[i])**2
        V_err += [err]
        average += err
    average = average/n
    
    fig = plt.figure()
#    plt.xlabel('# of question from the validation data') #if validation
#    plt.xlabel('# of question from the training data') #if training
    plt.xlabel('# of question from the testing data') #if testing
    plt.ylabel('Square error')
    sol = [ k for k in range(n) ]
    av = [ average for k in range(n) ]
    plt.scatter(sol,V_err,s=1,label='Conj Grad') 
    plt.plot(sol,av,label="MSE : {0}".format(round(average,4)),color='red')
#    plt.ylim(0,100)
    plt.legend()
    plt.show()
    
#PlotMSE()
    
    
###############################################################################
# To plot the error of differents method, function of iteration
###############################################################################

    
def PlotErr():
    (X,Wf,WCG,WG,W1,W2,tf,tCG,tG,t1,t2,errCG,errG,err1,err2) = Executeall(nbw)
    fig, ax = plt.subplots(figsize=None)
    ax.set_xlabel('iteration')
    ax.set_ylabel('error')
#    plt.plot([k for k in range(len(errf))],errf,label='linalg')
    ax.semilogy([k for k in range(len(errCG))],errCG,label='Conj Grad')    
    ax.semilogy([k for k in range(len(errG))],errG,label='Optimal Grad') 
#    ax.semilogy([k for k in range(len(err1))],err1,label='Const Grad')
    ax.semilogy([k for k in range(len(err2))],err2,label='Descrease Grad')
    plt.grid(True,which="both")
    plt.legend()
    plt.show()
    
#PlotErr()
    
    
###############################################################################
# To plot the figure of the runtime of different methods
###############################################################################
    
    
def PlotTime():
    (X,Wf,WCG,WG,W1,W2,tf,tCG,tG,t1,t2,errCG,errG,err1,err2) = Executeall(nbw)
    fig, ax = plt.subplots(figsize=None)
    ax.bar([1], [tCG], width=0.8, log='True', label='Conj Grad : {0}s'.format(round(tCG,3)) )
    ax.bar([2], [tf], width=0.8, log='True', label='linalg : {0}s'.format(round(tf,3)) )
    ax.bar([3], [tG], width=0.8, log='True', label='Optimal Grad : {0}s'.format(round(tG,3)) )
#    ax.bar([4], [t1], width=0.8, log='True', label='Const Grad : {0}s'.format(round(t1,3)) )
    ax.bar([4], [t2], width=0.8, log='True', label='Decrease Grad : {0}s'.format(round(t2,3)) )
    plt.grid(True,which="both",axis="y")
    plt.xticks([1,2,3,4], ('Conj Grad', 'linalg', 'Optimal Grad', 'Decrease Grad'))
    plt.ylim(min(tf,tCG,tG,t2)/2,max(tf,tCG,tG,t2)*2)
    plt.legend()
    plt.show()

#PlotTime()
    
    
###############################################################################
# To plot the figure of the error of the gradient descent with different factor
# of learning rate
###############################################################################
    
    
def PlotBadGradient():
    epsilon = 10**(-10)
    hf_words = dp.find_frequent_words(data[:10000],nbw)
    (X,Y,XTX,XTY) = dp.build_system(data,0,10000,nbw,hf_words)
    
    fig, ax = plt.subplots(figsize=None)
    ax.set_xlabel('iteration')
    ax.set_ylabel('error')
    
    for k in range(1,10):
        beta = 10**(-k)
        (WDG,tDG,errDG) = lr.step_decr_grad_adapt(XTX,np.zeros(len(X[0])),XTY,epsilon,beta)
        ax.semilogy([k for k in range(len(errDG))],errDG,label='factor 1e-{0}'.format(k))
    
    plt.grid(True,which="both")
    plt.legend()
    plt.show()
    
#PlotBadGradient()

    
    
    
    