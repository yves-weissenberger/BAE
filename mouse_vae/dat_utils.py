import numpy as np
import itertools
from scipy.linalg import toeplitz
from scipy.ndimage.filters import gaussian_filter
import copy as cp
import time
from sklearn import linear_model as lm
from sklearn.metrics import roc_auc_score


def _fit_behavioural_enc(lats,DM):
    
    """ Fit linear regression model to latent trajectories
        returning model parameters
        
        Arguments:
        ====================
        
        lats:           np.array
                        latent trajectory (nDims x nTimePoints) 
        
        DM:             np.array
                        Design Matrix for the regression Model        
        
    """
    
   
    seed = int(time.time())
    alfas = [0.001,0.01,0.1,1,5,10,50,100,500,1000]
    allBeta = []
    for i in range(lats.shape[0]):
        ar = lm.RidgeCV(alphas=alfas,fit_intercept=False,cv=5)#lm.Ridge(alpha=alfas[np.argmax(tmp)])
        ar.fit(DM.T,lats[i])
        beta = ar.coef_
        allBeta.append(beta)

    return np.array(allBeta)


def _discrete_convolve(arr,n_back):
    """ Perorm discrete convolution """
    arr2 = np.concatenate([np.zeros(n_back),arr,np.zeros(n_back)])
    bern = toeplitz(np.arange(n_back),arr2)[:,n_back:-n_back]
    return bern


def get_attention(DM,desc,n_back):
    """ Add attentional variable """
    click_ix = int(np.where(desc=='clicks0')[0])
    lick_ix = int(np.where(desc=='lickL0')[0])
    licks = np.squeeze(DM[lick_ix])
    dec = np.zeros(licks.shape)
    rts = []
    anti_dec = np.zeros(licks.shape)

    for clT in np.where(np.squeeze(DM[click_ix]))[0]:
        if np.any(licks[clT+3:clT+8]) and not np.any(licks[clT-2:clT+2]):
            #dec[clT+np.min(np.where(licks[clT+2:clT+10])[0])-5] = 1
            dec[clT-n_back] = 1
            #print(clT,clT+np.min(np.where(licks[clT+2:clT+10])[0]),clT+np.min(np.where(licks[clT+2:clT+10])[0])-5)
            rts.append(np.min(np.where(licks[clT:clT+10])[0]))
        elif not np.any(licks[clT+3:clT+8]) and not np.any(licks[clT-2:clT+2]):
            anti_dec[clT-n_back] = 1

            
    dec = _discrete_convolve(dec,n_back)
    anti_dec = _discrete_convolve(anti_dec,n_back)
    print(np.mean(rts),rts[:10])
    return dec,anti_dec

def get_full_DM(DM,desc,nBk,allVols):

    lick_ixs = np.where([1 if 'lickL' in i else 0 for i in desc])[0]
    rew_ixs = np.where([1 if 'rews' in i else 0 for i in desc])[0]
    bout_ixs = np.where([1 if 'bout' in i else 0 for i in desc])[0]
    click_ixs = np.where([1 if 'click' in i else 0 for i in desc])[0]


    ########################################################
    att,anti_att = get_attention(DM,desc,n_back=nBk)

    rest_bouts, click_bouts, clicks_select = get_bouts(DM,desc,allVols)


    tmp_a = np.zeros(DM.shape[1])
    tmp_b = np.zeros(DM.shape[1])

    tmp_a[np.array(click_bouts)-nBk] = 1
    tmp_b[np.array(rest_bouts)-nBk] = 1

    lick_ix = lick_ixs[0]
    xb = DM[lick_ix]
    motivation = np.vstack([gaussian_filter(xb,75),
                            gaussian_filter(xb,150),
                            gaussian_filter(xb,300),
                            gaussian_filter(xb,600),
                            gaussian_filter(xb,1200)]
                            )


    desc_full = ['offset']
    desc_full.extend(['lickL'+str(i) for i in range(len(lick_ixs))])
    desc_full.extend(['rew'+str(i) for i in range(len(rew_ixs))])
    desc_full.extend(['clicks'+str(i) for i in range(len(click_ixs))])
    desc_full.extend(['bout'+str(i) for i in range(len(bout_ixs))])
    desc_full.extend(['dec'+str(i) for i in range(nBk)])
    desc_full.extend(['dec'+str(i) for i in range(nBk)])
    desc_full.extend(['mot'+str(i) for i in range(motivation.shape[0])])
    desc_full.extend(['att'+str(i) for i in range(att.shape[0])])
    desc_full.extend(['att'+str(i) for i in range(anti_att.shape[0])])
    desc_full.extend(['time'+str(i) for i in range(10)])
    desc_full = np.array(desc_full)
    DM_FULL = get_DM_with_lowF(np.vstack([np.ones(DM.shape[1]),
                                DM[lick_ixs],
                                DM[rew_ixs],
                                DM[click_ixs],
                                np.hstack([DM[bout_ixs][:,:],np.zeros([len(bout_ixs),0])]),
                                _discrete_convolve(tmp_a,nBk),
                                _discrete_convolve(tmp_b,nBk),
                                motivation,
                                att,
                                anti_att
                               ]))
    return DM_FULL,desc_full

def get_DM_with_lowF(DM):
    """ Add low frequency oscillations to the design matrix """
    t = np.linspace(0,DM.shape[1],num=DM.shape[1])
    tot_s = DM.shape[1]/14.
    tot_samps = DM.shape[1]/14.

    hz01 = np.cos(t/700.*2*np.pi) #once every 100s
    slow_rhythms = np.vstack([np.cos(t/(1500.*i)*2*np.pi) for i in range(10,20)])
    return np.vstack([DM,slow_rhythms])

def get_bouts(DM,desc,allVols,ili_bout=4,cut_vols=8,resp_window=8,cut_vols_LOWER=np.inf):
    """ Get indices of bout onsets
    
        Arguments:
        =======================
        
        DM:             np.array
                        Design Matrix
        
        desc:           np.array
                        descriptor for the design matrix
     
        ili_bout:       int
                        cutoff for two licks being considered same vs
                        different bout
        
        cut_vols:       int || float
                        cutoff volume for including in stimulus bout
                        set
        
        resp_window:    int
                        cutoff window for counting as a hit trial
                    
                    
    
    """
    
    lickTs = np.where(np.squeeze(DM[desc=='lickL0']))[0]
    delta_lickTs = lickTs[1:] - lickTs[:-1]

    bout_init_licks = np.where(delta_lickTs>=ili_bout)[0]
    bout_init_ts = lickTs[1:][bout_init_licks]

    clickTs = np.where(np.squeeze(DM[desc=='clicks0']))[0]
    #Here separate the bouts into two types

    #rm_ixs = np.where(np.logical_or.reduce([allVols==99,allVols==12,allVols==11]))[0]
    rm_ixs = np.where(np.logical_and(allVols>cut_vols,allVols<cut_vols_LOWER))[0]

    clickTs2 = np.delete(clickTs,rm_ixs)
    #print(len(clickTs2))

    click_bouts = []
    rest_bouts = []
    clicks_select=[]
    jjk = 0
    for i in bout_init_ts:

        if not np.any(np.logical_and((i-lickTs)<7,(i-lickTs)>1)):

            #if there are any clicks between 2 and 12 frames following stimulus onset
            if np.any(np.logical_and((i-clickTs2)>=2,(i-clickTs2)<=resp_window)):
                #select the relevant stimulus
                #jjk += 1
                clickX = clickTs2[np.min(np.where(np.logical_and((i-clickTs2)>=2,(i-clickTs2)<=resp_window))[0])]
                #print (clickX)
                #if there are not licks occurring 10 frames preceding the stimulus
                if not np.any(np.logical_and((lickTs-clickX)>-5,(lickTs-clickX)<2)):
                    click_bouts.append(i)
                    clicks_select.append(np.where(clickTs==clickX)[0])

                
                clickTs2 = np.delete(clickTs2,np.where(clickTs2==clickX)[0])
            if not np.any(np.logical_and((i-clickTs)>-2,(i-clickTs)<20)):
                #if not np.any(np.logical_and((i-lickTs)>-5,(i-lickTs)<-1)):
                rest_bouts.append(i)
    return rest_bouts, click_bouts, clicks_select



def get_couterbalanced_sets(click_bouts,rest_bouts):
    
    """ Function cutting out start of latent states aligned to the onset
        of stimulus driven and spontaneous lick bouts
        
        Arguments:
        ===========================
        
        click_bouts:        list
                            list containing indices of lick bout starts
                        
        rest_bouts:         list
                            list containing indices of spontaneous lick bout starts
   
                        
    """

    nClk, nRest = len(click_bouts), len(rest_bouts)
    print(len(click_bouts),len(rest_bouts))
    click_ts_sel = []
    rest_ts_sel = []

    if nClk>nRest:
        tmp =cp.deepcopy(rest_bouts)
        rest_bouts = cp.deepcopy(click_bouts)
        click_bouts = cp.deepcopy(tmp)


    rest_bouts_sel = []
    tmp = cp.deepcopy(rest_bouts)

    for i in click_bouts:
        match_rest = np.argmin(np.abs(tmp-i))
        rest_bouts_sel.append(tmp[match_rest])
        rest_ts_sel.append(tmp[match_rest])

        tmp = np.delete(tmp,match_rest)
        click_ts_sel.append(i)
        #rest_ts_sel.append(tmp[match_rest])

        
    if nClk>nRest:

        tmp2 = cp.deepcopy(rest_ts_sel)
        rest_ts_sel = cp.deepcopy(click_ts_sel)
        click_ts_sel = cp.deepcopy(tmp2)



    return np.array(click_ts_sel),np.array(rest_ts_sel)