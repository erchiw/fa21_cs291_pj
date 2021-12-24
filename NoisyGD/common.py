import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve 
from autodp.calibrator_zoo import eps_delta_calibrator
from autodp.autodp_core import Mechanism
from autodp.mechanism_zoo import GaussianMechanism
from autodp.transformer_zoo import ComposeGaussian

rng = np.random.RandomState(1729)


# COMMAND ----------

# MAGIC %md data preprocessing

# COMMAND ----------

def add_ones(data):
    # add ones column to raw data X (i.e. intercept)
    # X should be n x d form and the return is in n x (d+1) form
    ones = np.repeat(1, data.shape[0])
    add_ones_data = np.insert(data, data.shape[1], ones, axis=1)
    return add_ones_data

def arg_max_vec(y):
    # convert to arg max by row
    return np.argmax(np.array(y), axis=1)

def get_ili_type(df, keep_filter, label_col = ('labels', 'ILI_type')):
    ## get ILI training label and ILI type for y
    y = df.loc[keep_filter, label_col]
    return y

def impute_back(df):
    return df.fillna(method= 'backfill', axis = 0)

def impute_na(df, value):
    ## impute na by value
    return df.fillna(value)

def one_hot_encoding(y, category=4):
    # y is in {0,1,2,3}, do one_hot_encoding
    # return a matrix in 4 x n form
    enc = np.zeros([category, y.shape[0]])
    for j in range(y.shape[0]):
        enc[y[j],j]=1
    return enc

def poisson_sampling(gamma, n):
    flag = rng.binomial(1, gamma, n)
    return flag
    

# COMMAND ----------

# MAGIC %md general function

# COMMAND ----------

def add_gauss_noise(sigma, dim1, dim2):
    gauss_noise = rng.normal(loc=0, scale=sigma, size=(dim1, dim2))
    return gauss_noise

def softmax_activation(z):
    exponentials = np.exp(z)
    exponentials_row_sums = np.sum(exponentials, axis=1)
    return exponentials / exponentials_row_sums[:, None]

def cross_entropy_loss(y_one_hot, activations):
    loss = -np.sum(y_one_hot * np.log(activations)) 
    #loss = -np.sum(np.sum(y_one_hot * np.log(activations), axis=1))    
    return loss


# COMMAND ----------

# MAGIC %md autodp related

# COMMAND ----------

class NoisyGD_mech(Mechanism):
    def __init__(self,sigma,coeff,name='NoisyGD'):
        Mechanism.__init__(self)
        self.name = name
        self.params={'sigma':sigma,'coeff':coeff}
        
        # ----------- Implement noisy-GD here with "GaussianMechanism" and "ComposeGaussian" ----------------
        gm = GaussianMechanism(sigma, name='Release_gradient')
        compose = ComposeGaussian()
        mech = compose([gm],[coeff])
        # ------------- return a Mechanism object named 'mech' --------------------s
        self.set_all_representation(mech)       

def find_appropriate_niter(sigma, eps, delta):
    # Use autodp calibrator for selecting 'niter'
    NoisyGD_fix_sigma = lambda x:  NoisyGD_mech(sigma,x)
    calibrate = eps_delta_calibrator()
    mech = calibrate(NoisyGD_fix_sigma, eps, delta, [0,500000])
    niter = int(np.floor(mech.params['coeff']))    
    return niter


# COMMAND ----------

# MAGIC %md Noisy Gradient Descent

# COMMAND ----------

def cal_grad(w, X, y, A):
    # calculate the full gradient
    w_gradients = -np.matmul(X.T, y - A)   
    return w_gradients

def clip_grad(w, threshod):
    #clip the full gradient 
    clip = np.minimum(1., threshod/np.linalg.norm(w, 'fro'))
    return clip   

def grad_step(w, x, y):
    # cal gradient per sample
    # w n_feature x n_class
    # x,y,a row vector
    exp_hat_y = np.exp(np.dot(x, w))
    grad = np.outer(x, exp_hat_y/np.sum(exp_hat_y) - y)
    return grad

def logis_pred(X, w):
    ## output a proba vector
    pred = softmax_activation(np.matmul(X,w))
    return pred

def theoretical_lr_choice(beta_L, f0_minus_fniter_bound, dim, sigma, niter):
    # beta_L is the gradient lipschitz constant for the whole objective function
    # sigma is the std of the gradient noise in each coordinate  (notice that this is the noise multiplier * GS)    
    return np.minimum(1/beta_L,np.sqrt(2*f0_minus_fniter_bound / (dim * sigma**2 *beta_L*niter)))

def run_noisyGD(niter, learning_rate, clip_threshold, sigma, X, y, w, GS, clip_or_not):
    n_feature = w.shape[0]
    n_class = w.shape[1]
    n_sample = X.shape[0]
    record_loss = []
    #record_clip = []
    for i in range(niter):
        if clip_or_not == False:
            Z = np.matmul(X, w)
            A = softmax_activation(Z)
            w_gradients = cal_grad(w, X, y, A)
        else: # clip
            w_gradients = 0
            clip = 0
            for j in range(n_sample):
                grad = grad_step(w, X[j], y[j])
                clip = clip_grad(grad, clip_threshold)
                w_gradients += grad*clip
            #record_clip.append(np.mean(clip))
        w -= learning_rate * (w_gradients + add_gauss_noise(GS*sigma, n_feature, n_class))
        # calculated loss
        loss = cross_entropy_loss(y, softmax_activation(np.matmul(X, w)))
        record_loss.append(loss)    
    return w, record_loss #, record_clip

def run_multiple_noisy_gd(X_train, y_train, X_test, y_test, eps_list, rep, delta, sigma, beta_L, f0_minus_fniter_bound, GS, clip_threshold, clip_or_not):
    ## run noisyGD w.r.t. epsilon list
    tr_loss = []
    te_loss = []
    tr_auc = []
    te_auc = []
    n_feature = X_train.shape[1]
    n_class = y_train.shape[1]
        
    for ep_s in eps_list:
        n_iterations = find_appropriate_niter(sigma, ep_s, delta)
        print(ep_s)
        for i in range(rep):   
            w_ini = np.zeros((n_feature, n_class)).astype('float32')

            if n_iterations == 0:
                tr_loss.append(100)
                te_loss.append(100)
                tr_auc.append(0)
                te_auc.append(0)
    
            if n_iterations > 0:    
                learning_rate = theoretical_lr_choice(beta_L, f0_minus_fniter_bound, n_feature, 
                                                      sigma*GS, n_iterations)
                w_train, _ = run_noisyGD(niter=n_iterations, learning_rate=learning_rate, 
                                         clip_threshold = clip_threshold, sigma=sigma, X=X_train, 
                                         y=y_train, w=w_ini, GS=GS, clip_or_not=clip_or_not)
                
                y_pred_prob_train = logis_pred(X_train, w_train)
                y_pred_prob_test = logis_pred(X_test, w_train)
                train_loss = cross_entropy_loss(y_train, softmax_activation(np.matmul(X_train, w_train)))
                test_loss = cross_entropy_loss(y_test, softmax_activation(np.matmul(X_test, w_train)))
                tr_loss.append(train_loss)
                te_loss.append(test_loss)
                tr_auc.append(roc_auc_score(y_train, y_pred_prob_train, multi_class='ovr'))
                te_auc.append(roc_auc_score(y_test, y_pred_prob_test, multi_class='ovr'))
    return tr_loss, te_loss, tr_auc, te_auc
