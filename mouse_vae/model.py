import tensorflow as tf
import numpy as np
import sys
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score

import time
from sklearn import linear_model as lm
import copy as cp
from . import dat_utils
from . import model_utils


class BAE:
    """ Augmented Latent Variable Model to fit to data """
    
    def __init__(self,network_params=None,data_split=[.7,.2,.1]):

        self.network_params = network_params
        self._hasData = False
        self._prevLL = -np.inf
        self._hasInit = False
        self.data_split = data_split

        # Helper function to divide data into training, test and validation sets
        from .model_utils import get_tvt
        self._get_tvt = get_tvt
        if network_params is None:
            self._set_default_params()





    def add_data(self,video,DM,descriptor):
        """ Add video data and a design matrix. Currently only supports videos that are
            square in shape.

            Arguments:
            =======================

            video:          np.array
                            video of the subject of size (nFrames x nPixels x nPixels)
            
            DM:             np.array
                            Design Matrix for Augmenting the latent variable model. Size = (nFtrs,nFrames)

            descriptor:     np.array
                            descriptor for the design matrix

        """

        if (video.shape[1]!=video.shape[2]):
            raise Exception("Currently only supports square image frames Dimensions of video must be (nFrames x nPixels x nPixels)")
        else:
            self.video = video
            self._sz = video.shape[1]
            self.nFrames = video.shape[0]

        if not np.logical_or(np.ndim(DM)!=2,DM.shape[1]==video.shape[0]):
            raise Exception("DM should be 2d where second dimension has same size as number of video frames")
        else:
            self.DM = DM


        if len(descriptor)!=DM.shape[0]:
            raise Exception("Descriptor should be same shape as Design Matrix")
        else:
            self.DM_descriptor = descriptor
        self._hasData = True



    def run_tf_setup(self,make_encoder=None,make_decoder=None,linear_predictor=None,make_prior=None):

        """ Initialise network parameters and tensorflow graph 
            If no encoder, decoder or linear predictor are specified, will use default
            parameters.
        """

        tfd = tf.contrib.distributions

        if make_decoder is None:
            from .model_utils import make_decoder
        if linear_predictor is None:
            from .model_utils import linear_predictor
        if make_prior is None:
            from .model_utils import make_prior
        if make_encoder is None:
            from .model_utils import make_encoder




        if not self._hasData:
            print("You haven't added data yet. Run the add_data function to add video data and a Design Matrix first =)")
        else:


            self._init_placeholders()

            #Create variable sharing wrappers around en-and-decoders
            make_encoder = tf.make_template('encoder', make_encoder)
            make_decoder = tf.make_template('decoder', make_decoder)
            make_predictor = tf.make_template('decoder', linear_predictor)

            #Prior on the latent space
            self.prior = make_prior(nlatDim=self.network_params['nLatentDim'])

            #Posterior distribution, given the video, over latent variables with dropout
            self._posterior = make_encoder(self.data,
                                            data_shape=self._sz,
                                            nlatDim=self.network_params['nLatentDim'],
                                            keep_frac=.8)

            #Posterior distribution, given the video, over latent variables without dropout
            self.posterior2 = make_encoder(self.data,
                                           data_shape=self._sz,
                                           nlatDim=self.network_params['nLatentDim'],
                                           keep_frac=1)

            #grab sample from posterior to calculate cost via reparameterisation trick
            self.post_samp =  self._posterior.sample()
            self.lin_cost = make_predictor(self.post_samp,self.inDM,NDIMS=self.network_params['nLatentDim'])
            self.likelihood = make_decoder(self.post_samp, [self._sz,self._sz]).log_prob(self.data)
            self.divergence = tfd.kl_divergence(self._posterior, self.prior)
            self.elbo = tf.reduce_mean(self.likelihood - self.divergence - self.network_params['encode_weight']*self.lin_cost)
            self.predictor = make_decoder(self._latents,[self._sz,self._sz]).mean()

            # end of cost function code block


            self.optimize = tf.train.AdamOptimizer(self.network_params['learning_rate']).minimize(-self.elbo)
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            self._hasInit = True

            nBatch = int(np.floor(self.nFrames/self.network_params['batch_sz']))

            self.allIDXS, self.train_idxs, \
            self.validation_idxs, self.test_idxs = self._get_tvt(self.network_params['batch_sz'],
                                                                 nBatch,
                                                                 tvt=self.data_split,
                                                                 random_seed=100)
            self.nTrain_batch = self.train_idxs.shape[0]








    def fit(self,verbose=True):
        """ Fit the latent variable model. To chance fitting parameters update
            lvm.network_params before fitting
        """ 
        if not self._hasInit:
            raise Exception("Run tensorflow setup first")

        else:
            if verbose:
                print("Fitting is happening")

            for epoch in range(self.network_params['n_epochs']):


                train_idxs = np.random.permutation(self.train_idxs.flatten()).reshape(-1,self.network_params['batch_sz'])
                nValPs = int(np.floor(len(self.validation_idxs)/100.))
                tmp_a = []

                for i,kkk in enumerate(self.train_idxs):
                    if verbose:
                        sys.stdout.write("\rRunning minibatch: %s/%s ||   " %(i+1,self.nTrain_batch))
                        sys.stdout.flush()

                    self.sess.run(self.optimize, {self.data: self.video[kkk,:],
                                                  self.inDM:self.DM[:,kkk].T
                                                    })


                for kk in range(nValPs+1):
                    ix_set_b = self.validation_idxs[kk*100:(kk+1)*100]
                    if len(ix_set_b)>0:
                        test_elbo = self.sess.run( self.elbo, {self.data: self.video[ix_set_b,:],
                                                               self.inDM:self.DM[:,ix_set_b].T})
                        tmp_a.append(test_elbo)
                if verbose:
                    print(' Epoch', epoch+1, 'Validation Loss:', -np.mean(tmp_a))
                test_elbo = np.mean(tmp_a)




    def get_latent_states(self):
        """ Use fit en-and decoder networks to extract latent states."""
        totN = self.video.shape[0]

        nRnds = int(np.floor(totN/100.))
        if np.remainder(totN,nRnds)!=0:
            plus = 1

        embedAll = []
        for i in range(nRnds+plus):
            sys.stdout.write("\r Extracting from batch: %s / %s" %(i,nRnds))
            sys.stdout.flush()
            tmp_b = self.sess.run(self.posterior2.loc, {self.data: self.video[i*100:(i+1)*100,:]
                            })
            embedAll.append(tmp_b)

            self.lats = np.vstack(embedAll)

        print('\n Succesfully extracted Latent States: access via lvm.lats')


    def decode(self,decode_ev='dec',window=[1,4]):

        """ Run decoder on events

            Arguments:
            ==============================

            decode_ev:      str
                            name of variable to be decoded, see DM_descriptor


            window:         list || np.array
                            list of two integers specifying the window before
                            and after the event of interest to use for decoding

            Returns:
            ==============================

        """

        nBk = int(np.sum([decode_ev in i for i in self.DM_descriptor])/2.)


        dec_ix_st = np.where(self.DM_descriptor==decode_ev+'0')[0][0]
        
        #index of stimulus driven and spontaneous bout regressors
        click_ix,spont_ix = np.where(self.DM_descriptor==decode_ev+'0')[0]
        

        #index, in design matrix of stimulus driven bouts
        click_sel = np.where(self.DM[click_ix])[0]
        #index, in design matrix of spontaneous bouts
        spont_sel = np.where(self.DM[spont_ix])[0]
        
        all_sel = np.concatenate([click_sel,spont_sel])
        if not hasattr(self,'enc_params'):
            print('Fitting encoding model')
            self.enc_params = self.fit_encoding_model()


        eg_ev = self.DM[click_ix][click_sel[0]-window[0]: click_sel[0]+window[1]]

        #Project latent states of stimulus driven bouts to the decision axis
        proj1 = model_utils.run_prediction(self.enc_params,self.lats,self.DM,click_sel,
                               [click_ix,spont_ix],nBk,eg_ev,window)

        
        #Project latent states of spontaneous bouts to the decision axis
        proj2 = model_utils.run_prediction(self.enc_params,self.lats,self.DM,spont_sel,
                               [click_ix,spont_ix],nBk,eg_ev,window)

        return (click_sel,spont_sel), (proj1,proj2)


    def fit_encoding_model(self):

        """ Fit Behavioural encoding model to extracted latent states. 

        """
        if hasattr(self,'lats'):
            self.enc_params = dat_utils._fit_behavioural_enc(self.lats.T,self.DM)
            print("Encoding Model Fit")
        else:
            print("Latent states have not been extracted yet. First run lvm.fit() followed by lvm.get_latent_states()")


    def estimate_decoding_perf(self,decode_ev='dec',window=[1,5],kfp=[3,1],verbose=True,counterbalance=True):
        """ Estimate Decoding Performance. Rather than refitting the full network each iteration, 
            we exploit the fact that latent states are extracted independetly
            of the behavioural encoding model and refit encoding model parameters in
            each of K-folds.
            
            Arguments:
            ==============================

            decode_ev:      str
                            name of variable to be decoded, see DM_descriptor

            window:         list || np.array
                            list of two integers specifying the window before
                            and after the event of interest to use for decoding

            kfp:            list || np.array
                            list of 2 parameters defining structure of K-Fold cross validation.
                            First parameter specifies the number of splits in the data. Second
                            parameter specifies the number of repeats

            verbose:        bool
                            if True more stuff is printed

            counterbalance: bool
                            whether to select temporally counterbalanced groups 
                            of bouts
        """
        

        rskf = RepeatedStratifiedKFold(n_splits=kfp[0], n_repeats=kfp[1],random_state=99)
        nBk = int(np.sum([decode_ev in i for i in self.DM_descriptor])/2.)
        dec_ix_st = np.where(self.DM_descriptor==decode_ev+'0')[0][0]
        
        #index of stimulus driven and spontaneous bout regressors
        click_ix,spont_ix = np.where(self.DM_descriptor==decode_ev+'0')[0]
        

        #index, in design matrix of stimulus driven bouts
        click_sel = np.where(self.DM[click_ix])[0]
        #index, in design matrix of spontaneous bouts
        spont_sel = np.where(self.DM[spont_ix])[0]

        if counterbalance:
            click_sel,spont_sel = dat_utils.get_couterbalanced_sets(click_sel,spont_sel)

        
        all_sel = np.concatenate([click_sel,spont_sel])
        
        #initialise counter
        kk = 1

        #Create a dict to store results in
        self.decode_perf = {'roc':[],
                            'frac_corr': []}

        #Divide data into train and test set
        for train_index, test_index in rskf.split(all_sel,y=np.concatenate([np.ones_like(click_sel),np.zeros_like(spont_sel)])):
            sys.stdout.write("\rrunning fold %s" %kk)
            sys.stdout.flush()
            
            #Get indices to test bouts to be removed from fitting dataset
            test_idxs = all_sel[test_index]
            
            DM_tmp = cp.deepcopy(self.DM)
            lats_tmp = cp.deepcopy(self.lats).T

            for j in reversed(test_idxs):
                #print(j)
                DM_tmp = np.delete(DM_tmp,slice(j-2*np.sum(window),j+2*np.sum(window)),axis=1)
                lats_tmp = np.delete(lats_tmp,slice(j-2*np.sum(window),j+2*np.sum(window)),axis=1)

            #Get clicks and spontaneous bouts
            test_index_click = test_index[np.where(test_index<len(click_sel))[0]]
            test_index_spont = test_index[np.where(test_index>=len(click_sel))[0]]-len(click_sel)

            
            #Fit behavioural encoding model to training dataset
            b = dat_utils._fit_behavioural_enc(lats_tmp,DM_tmp)
            
            #This is what the event should look like
            eg_ev = self.DM[dec_ix_st][test_idxs[test_index_click[0]]-window[0]: \
                                       test_idxs[test_index_click[0]]+window[1]]



            #Project latent states of stimulus driven bouts to the decision axis
            proj1 = model_utils.run_prediction(b,self.lats,self.DM,click_sel[test_index_click],
                                   [click_ix,spont_ix],nBk,eg_ev,window)

            
            #Project latent states of spontaneous bouts to the decision axis
            proj2 = model_utils.run_prediction(b,self.lats,self.DM,spont_sel[test_index_spont],
                                   [click_ix,spont_ix],nBk,eg_ev,window)



            #fraction correctly classified
            fCi = np.mean([np.mean(np.array(proj1)<0),np.mean(np.array(proj2)>0)])

            roci = roc_auc_score(np.concatenate([np.zeros_like(proj1),
                                                 np.ones_like(proj2)]),
                                np.concatenate([proj1,proj2]))

            self.decode_perf['roc'].append(roci)
            self.decode_perf['frac_corr'].append(fCi)
            
            if verbose:
                print('\nroc-score: %s    || fraction correct: %s' %(roci,fCi))
                
            kk += 1

        print('Performance Estimation Complete. \nResults stored in lvm.decode_perf. \nEstimated decoding accuracy: %s%%' \
                 %(np.round(100*np.mean(self.decode_perf['frac_corr']))))



    def estimate_decoding_perf_full_cv(self,decode_ev='dec',train_test=[.8,.2],window=[1,5],counterbalance=True):
        """ Fully cross-validated decoding estimation. For the paranoid amongst us.
            Fit network and behavioural encoding model leaving out a subset of 
            stimulus driven and spontaneous bouts that are then used to 
            estimate decoding performance

            Arguments:
            ===========================

            decode_ev:      str
                            name of variable to be decoded, see DM_descriptor

            train_test:     list of float
                            list of floats that should sum to one specifying the
                            relative sizes of train and test sets

            window:         list || np.array
                            list of two integers specifying the window before
                            and after the event of interest to use for decoding

            counterbalance: bool
                            whether to select temporally counterbalanced groups 
                            of bouts


        """ 
        if not self._hasInit:
            raise Exception("Run tensorflow setup first")
        else:
            print("Reinitialising all parameters")
            self.sess.run(tf.global_variables_initializer())
            print('Training network')



            nBk = int(np.sum([decode_ev in i for i in self.DM_descriptor])/2.)
            dec_ix_st = np.where(self.DM_descriptor==decode_ev+'0')[0][0]
            
            #index of stimulus driven and spontaneous bout regressors
            click_ix,spont_ix = np.where(self.DM_descriptor==decode_ev+'0')[0]

            #index, in design matrix of stimulus driven bouts
            click_sel = np.where(self.DM[click_ix])[0]
            #index, in design matrix of spontaneous bouts
            spont_sel = np.where(self.DM[spont_ix])[0]

            if counterbalance:
                click_sel,spont_sel = dat_utils.get_couterbalanced_sets(click_sel,spont_sel)

            print(len(click_sel),len(spont_sel))
            ixs_1 = np.random.permutation(np.arange(len(click_sel)))
            ixs_2 = np.random.permutation(np.arange(len(spont_sel)))


            nTest1 = int(len(ixs_1)*train_test[1])
            click_test = click_sel[ixs_1[:nTest1]]


            nTest2 = int(len(ixs_2)*train_test[1])
            spont_test = spont_sel[ixs_2[:nTest2]]

            CV_dict = {'click_test': click_test,
                       'spont_test': spont_test
                       }

            rm_click = np.concatenate([np.arange(i-10,i+30) for i in click_test])
            rm_spont = np.concatenate([np.arange(i-10,i+30) for i in spont_test])
            allRM = np.concatenate([rm_click,rm_spont])





            for epoch in range(self.network_params['n_epochs']):


                train_idxs = np.random.permutation(self.train_idxs.flatten()).reshape(-1,self.network_params['batch_sz'])
                nValPs = int(np.floor(len(self.validation_idxs)/100.))
                tmp_a = []

                for i,kkk in enumerate(self.train_idxs):
                    sys.stdout.write("\rRunning minibatch: %s/%s ||   " %(i+1,self.nTrain_batch))
                    sys.stdout.flush()
                    kkk = np.array([i for i in kkk if i not in allRM])

                    self.sess.run(self.optimize, {self.data: self.video[kkk,:],
                                                  self.inDM:self.DM[:,kkk].T
                                                    })


                for kk in range(nValPs+1):
                    ix_set_b = self.validation_idxs[kk*100:(kk+1)*100]
                    if len(ix_set_b)>0:
                        test_elbo = self.sess.run( self.elbo, {self.data: self.video[ix_set_b,:],
                                                               self.inDM:self.DM[:,ix_set_b].T})
                        tmp_a.append(test_elbo)
                print(' Epoch', epoch+1, 'Validation Loss:', -np.mean(tmp_a))
                test_elbo = np.mean(tmp_a)


            train_ixs = np.arange(self.DM.shape[1])
            self.get_latent_states()
            train_ixs = [i for i in train_ixs if i not in allRM]
            self.enc_params = dat_utils._fit_behavioural_enc(self.lats[train_ixs].T,self.DM[:,train_ixs])
            eg_ev = self.DM[click_ix][click_sel[0]-window[0]: click_sel[0]+window[1]]

            #Project latent states of stimulus driven bouts to the decision axis
            proj1 = model_utils.run_prediction(self.enc_params,self.lats,self.DM,CV_dict['click_test'],
                                   [click_ix,spont_ix],nBk,eg_ev,window)

            
            #Project latent states of spontaneous bouts to the decision axis
            proj2 = model_utils.run_prediction(self.enc_params,self.lats,self.DM,CV_dict['spont_test'],
                                   [click_ix,spont_ix],nBk,eg_ev,window)


            fCi = np.mean([np.mean(np.array(proj1)<0),np.mean(np.array(proj2)>0)])
            print(fCi)



    def _init_placeholders(self):
        self.data = tf.placeholder(tf.float32, [None, self._sz,self._sz])
        self.inDM = tf.placeholder(tf.float32, [None, self.DM.shape[0]])
        self._latents = tf.placeholder(tf.float32, [None, self.network_params['nLatentDim']])



    def _set_default_params(self):
        self.network_params = {}
        self.network_params['nLatentDim'] = 10
        self.network_params['train_frac_keep'] = 0.8
        self.network_params['encode_weight'] = float(1e4)
        self.network_params['learning_rate'] = 0.005
        self.network_params['n_epochs'] = 10
        self.network_params['batch_sz'] = 200