import cupy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve 

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
    return loss


# COMMAND ----------

# MAGIC %md Noisy Gradient Descent

# COMMAND ----------
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
    #n_sample = X.shape[0]
    #record_loss = []
    #flag_clip = 0
    #class_weight = np.repeat(np.array([1./6, 1, 1, 1]), 
    
    for i in range(niter):
        y_pred = softmax_activation(np.matmul(X, w))
        if clip_or_not == False:
            w_gradients = -np.matmul(X.T, y-y_pred)
        else:
            #flag_clip = 1
            w_gradients = -np.einsum('ij,ik->ijk', X, y-y_pred)
            clip = np.minimum(1, clip_threshold/np.linalg.norm(w_gradients, axis=(1,2), ord='fro'))
            w_gradients = np.einsum('i,ijk->jk', clip, w_gradients)
    
        w -= learning_rate * (w_gradients + add_gauss_noise(GS*sigma, n_feature, n_class))
    return w

def run_multiple_noisy_gd(X_train, y_train, X_test, y_test, eps_list, rep, delta, sigma, beta_L, f0_minus_fniter_bound, GS, clip_threshold, clip_or_not, lr, iters):
    ## run noisyGD w.r.t. epsilon list
    tr_loss = []
    te_loss = []
    tr_auc = []
    te_auc = []
    tr_acc = []
    te_acc = []
    pred_test = []
    pred_train = []
    n_feature = X_train.shape[1]
    n_class = y_train.shape[1]
    #flag_clip = []
    
    tr_tru = np.argmax(y_train, axis=1)
    te_tru = np.argmax(y_test, axis=1)
    j=0
    for ep_s in eps_list:
        n_iterations = iters[j]
        learning_rate = lr[j]
        j+=1
        print(ep_s)
        for i in range(rep):   
            w_ini = np.zeros((n_feature, n_class))

            if n_iterations == 0:
                tr_loss.append(100)
                te_loss.append(100)
                tr_auc.append(0)
                te_auc.append(0)
                tr_acc.append(0)
                te_acc.append(0)
                pred_train.append(1)
                pred_test.append(1)
    
            if n_iterations > 0:    
                w_train = run_noisyGD(niter=n_iterations, learning_rate=learning_rate, 
                                         clip_threshold = clip_threshold, sigma=sigma, X=X_train, 
                                         y=y_train, w=w_ini, GS=GS, clip_or_not=clip_or_not)
                
                y_pred_prob_train = logis_pred(X_train, w_train)
                y_pred_prob_test = logis_pred(X_test, w_train)
                #loss
                train_loss = cross_entropy_loss(y_train, y_pred_prob_train)
                test_loss = cross_entropy_loss(y_test, y_pred_prob_test)
                tr_loss.append(train_loss)
                te_loss.append(test_loss)
                #auc
                tr_auc.append(roc_auc_score(y_train.get(), y_pred_prob_train.get(), multi_class='ovr'))
                te_auc.append(roc_auc_score(y_test.get(), y_pred_prob_test.get(), multi_class='ovr'))
                # acc
                tr_pre = np.argmax(y_pred_prob_train, axis=1)
                pred_train.append(float(sum(tr_pre==0)) / tr_tru.shape[0])
                tr_acc.append(float(sum(tr_pre==tr_tru)) / tr_tru.shape[0])
               
                te_pre = np.argmax(y_pred_prob_test, axis=1)
                pred_test.append(float(sum(te_pre==0)) / te_tru.shape[0])
                te_acc.append(float(sum(te_pre==te_tru)) / te_tru.shape[0])
                
    return tr_loss, te_loss, tr_auc, te_auc, tr_acc, te_acc, pred_train, pred_test
       
