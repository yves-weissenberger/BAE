import tensorflow as tf
import numpy as np
import copy as cp
from . import dat_utils
tfd = tf.contrib.distributions


def make_encoder(data,data_shape, nlatDim,keep_frac=.8):
    activation = tf.nn.relu
    #x = tf.layers.flatten(data)
    x = tf.reshape(data,[-1,data_shape,data_shape,1])
    x = tf.nn.dropout(x,keep_frac)

    x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=2, padding='same', activation=activation)
    x = tf.nn.dropout(x,keep_frac)
    x = tf.layers.conv2d(x, filters=128, kernel_size=5, strides=2, padding='same', activation=activation)
    x = tf.nn.dropout(x,keep_frac)
    
    #x = tf.layers.conv2d(x, filters=64, kernel_size=10, strides=4, padding='same', activation=activation)
    #x = tf.nn.dropout(x,keep_frac)


    #x = tf.layers.dense(x, 200, tf.nn.relu)
    #x = tf.nn.dropout(x,0.8)
    #x = tf.layers.dense(x, 100, tf.nn.relu)
    #x = tf.nn.dropout(x,0.8)

    x = tf.layers.flatten(data)

    #x =  tf.concat([x,inDM],axis=1)
    x = tf.layers.dense(x, 100, tf.nn.relu)
    #x = tf.nn.dropout(x,keep_frac)
    #x =  tf.concat([x,inDM],axis=1)

    loc = tf.layers.dense(x, nlatDim)
    scale = tf.layers.dense(x, nlatDim, tf.nn.softplus)
    return tfd.MultivariateNormalDiag(loc, scale + 1)

def make_decoder(code, data_shape):
    keep_frac = .8
    activation = tf.nn.relu
    x = code#tf.concat([code,inDM],axis=1)
    x = tf.layers.dense(x, 100, activation=activation)
    #x = tf.nn.dropout(x,keep_frac)
    #x = tf.reshape(data,[-1]+[10,10]+[1])


    #x = tf.layers.conv2d_transpose(x, filters=4, kernel_size=2, strides=1, padding='same', activation=activation)

    #x = tf.layers.dense(x,200, tf.nn.relu)
    #x = tf.nn.dropout(x,keep_frac)
    #x = tf.layers.conv2d_transpose(x, filters=2, kernel_size=2, strides=2, padding='same', activation=activation)

    #x = tf.nn.dropout(x,keep_frac)

    x = tf.layers.dense(x, 500, activation=activation)
    #x = tf.nn.dropout(x,keep_frac)
    #x = tf.nn.dropout(x,keep_frac)
    x = tf.layers.flatten(x)

    logit = tf.layers.dense(x, np.product(data_shape),activation=None)
    #scale2 = tf.layers.dense(x, np.product(data_shape),)

    logit = tf.reshape(logit, [-1] + data_shape)
    #scale2 = tf.reshape(scale2, [-1] + data_shape)
    #return tfd.MultivariateNormalDiag(logit,np.ones([float(data_shape[0]),float(data_shape[0])]).astype('float32'))
    # the second argument specifies how many dimensions to project back into
    return tfd.Independent(tfd.Normal(logit,np.ones(logit.shape[1]).astype('float32')),2)

def linear_predictor(code,in_DM,NDIMS):

    linLayer = tf.layers.dense(in_DM,NDIMS ,activation=None)
    return tf.reduce_mean(tf.abs(linLayer - code))

def make_prior(nlatDim):
    loc = tf.zeros(nlatDim)
    scale = tf.ones(nlatDim)
    return tfd.MultivariateNormalDiag(loc, scale)

def get_tvt(batch_sz,nBatch,tvt=[.8,.2,.0],random_seed=100):
    np.random.seed(random_seed)
    allIDXS = np.random.permutation(np.arange(nBatch*batch_sz))
    train_l = int(np.floor(int(tvt[0]*len(allIDXS))/batch_sz)*batch_sz)
    train_idxs = allIDXS[:train_l].reshape(-1,batch_sz)
    validation_idxs = allIDXS[train_l:int(train_l + (tvt[1])*len(allIDXS))]
    test_idxs = allIDXS[int(train_l + (tvt[1])*len(allIDXS)):]
    return allIDXS, train_idxs, validation_idxs, test_idxs



def run_prediction(b,lats,DM,evTs,ev_ixs,nBk,eg_ev,LAT_WINDOW):
    proj = []
    for clT in evTs:
        
        xs_ = lats[clT-LAT_WINDOW[0]:clT+LAT_WINDOW[1],:]
        DM_stim = cp.deepcopy(DM[:,clT-LAT_WINDOW[0]:clT+LAT_WINDOW[1]])
        
        DM_stim[ev_ixs[0]:ev_ixs[0]+nBk] = dat_utils._discrete_convolve(eg_ev,nBk)
        DM_stim[ev_ixs[1]:ev_ixs[1]+nBk] = dat_utils._discrete_convolve(np.zeros_like(eg_ev),nBk)


        DM_spont = cp.deepcopy(DM[:,clT-LAT_WINDOW[0]:clT+LAT_WINDOW[1]])
        DM_spont[ev_ixs[0]:ev_ixs[0]+nBk] = dat_utils._discrete_convolve(np.zeros_like(eg_ev),nBk)
        DM_spont[ev_ixs[1]:ev_ixs[1]+nBk] = dat_utils._discrete_convolve(eg_ev,nBk)

        stim_resp = b.dot(DM_stim)
        spont_resp = b.dot(DM_spont)
        tmp_ = np.sum(np.abs(stim_resp.T-xs_)) - np.sum(np.abs(spont_resp.T-xs_))
        proj.append(tmp_)
    return np.array(proj)