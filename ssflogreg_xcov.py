"""
Predictive pattern decomposition:
Semi-supervised region/network decomposition by low-rank logistic regression
"""

print __doc__

import os
import os.path as op
import numpy as np
import glob
from scipy.linalg import norm
import nibabel as nib
from sklearn.grid_search import RandomizedSearchCV
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from nilearn.input_data import NiftiMasker
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import theano
import theano.tensor as T
from matplotlib import pylab as plt
print('Running THEANO on %s' % theano.config.device)
from nilearn.image import concat_imgs
import joblib
import time
from unpack_mat import loadmatnow
import re

RES_NAME = 'ssflogreg_xcov'
WRITE_DIR = op.join(os.getcwd(), RES_NAME)
if not op.exists(WRITE_DIR):
    os.mkdir(WRITE_DIR)
    
##############################################################################
# load+preprocess data
##############################################################################

print('Loading data...')

# load the information + niftis from SPM analyses
mat_paths = glob.glob('/Volumes/TRESOR/houpand/1st_level_unnormalized/*/SPM.mat')


cond_labels = []
sub_labels = []
nii_paths = []

# 5 trials per block; 
cond_names = ['nehmen', 'fangen', 'rollen',
              'Ring', 'Kugel', 'Zylinder',
              'Ring_3D', 'Kugel_3D', 'Zylinder_3D',
              'Ring_Hand', 'Kugel_Hand', 'Zylinder_Hand']

for ipath, mpath in enumerate(mat_paths):
    m = loadmatnow(mpath)
    path = op.dirname(mpath)
    print('Mat-File: %i/%i' % (ipath + 1, len(mat_paths)))

    try:
        nscans = len(m['SPM']['Vbeta'])
    except:
        raise IOError('This subject has no Vbeta structure! - Skipping...')

    # 1=control, 2=depression, 3=schizophrenia
    sub_label = re.search(r'T[0-9]{1,4}', path).group(0)

    for s in xrange(nscans):
        item = m['SPM']['Vbeta'][s]
        descr = str(item.descrip)

        istr = descr.find('Sn(')

        if istr != -1:
            try:
                strcond = re.search('.*_', descr[(istr+6):]).group(0)[:-1]
            except:
                continue
                
            nii_path = op.join(path, item.fname)
            print strcond

            if not op.exists(nii_path):
                raise IOError('One of the img is not in the right spot')
                
            try:
                i_cond = cond_names.index(strcond)
                cond_labels.append(i_cond)
                nii_paths.append(nii_path)
                sub_labels.append(sub_label)
            except ValueError:
                print('Skipped (%s)!' % descr)

        else:
            print('Skipped (%s)!' % descr)
        
cond_labels = np.array(cond_labels)
sub_labels = np.array(sub_labels)
nii_paths = np.array(nii_paths)

assert len(cond_labels) == len(nii_paths) == len(sub_labels)
assert len(np.unique(cond_labels)) == 12
assert len(np.unique(sub_labels)) == 20
print('Found ' + str(len(nii_paths)) + ' images!')



import nibabel as nib
from nilearn.input_data import NiftiMasker
from nilearn.image import resample_img
nii_temp = nib.load(nii_paths[0])

mask_file = nib.load('grey10_icbm_3mm_bin_2ero.nii.gz')
r_mask = resample_img(
    img=mask_file,
    target_affine=nii_temp.get_affine(),
    target_shape=nii_temp.shape,
    interpolation='nearest')
r_mask.to_filename('debug_mask.nii')
assert len(np.unique(r_mask.get_data())) == 2


# reduce the task images to the gray matter
masker = NiftiMasker(mask_img=r_mask, smoothing_fwhm=False, standardize=False)
masker.fit()

n_files = len(nii_paths)
n_vox = r_mask.get_data().sum()

if op.exists('dump_FS.npy'):
    FS = np.load('dump_FS.npy')
    cond_labels = np.load('dump_cond.npy')
    sub_labels = np.load('dump_subs.npy')
else:
    FS = np.zeros((n_files, n_vox))
    for i_nii in range(n_files):
        print('Loading nifti into memory: %i/%i' % (i_nii + 1, n_files))
        data = np.nan_to_num(nib.load(nii_paths[i_nii]).get_data())
        nii = nib.Nifti1Image(data, r_mask.get_affine())
        cur_1d_data = masker.transform(nii)
        FS[i_nii, :] = cur_1d_data

    # save feature space to disk
    np.save('dump_FS', FS)
    np.save('dump_cond', cond_labels)
    np.save('dump_subs', sub_labels)


from sklearn.preprocessing import StandardScaler

FS = StandardScaler().fit_transform(FS)
labels = cond_labels

# type conversion
FS = np.float32(FS)
labels = np.int32(labels)

# contrasts are IN ORDER -> shuffle!
new_inds = np.arange(0, FS.shape[0])
np.random.shuffle(new_inds)

FS = FS[new_inds]
labels = labels[new_inds]
sub_labels = sub_labels[new_inds]


# HACK! define the tasks to predict
inds_verbs = np.logical_or(np.logical_or(labels == 0, labels == 1), labels == 2)
inds_nomen = np.logical_or(np.logical_or(labels == 3, labels == 4), labels == 5)
inds_objects = np.logical_or(np.logical_or(labels == 6, labels == 7), labels == 8)
inds_objects_hand = np.logical_or(np.logical_or(labels == 9, labels == 10), labels == 11)


labels[inds_verbs] = 0
labels[inds_nomen] = 1
labels[inds_objects] = 2
labels[inds_objects_hand] = 3


##############################################################################
# define computation graph
##############################################################################

# Brian Cheung
def xcov(actset_1, actset_2):
    N = actset_1.shape[0].astype(theano.config.floatX)
    actset_1 = actset_1 - actset_1.mean(axis=0, keepdims=True)
    actset_2 = actset_2 - actset_2.mean(axis=0, keepdims=True)
    cc = T.dot(actset_1.T, actset_2)/N
    cost = .5 * T.sqr(cc).mean()
    return cost


class SSEncoder(BaseEstimator):
    def __init__(self, n_hidden, gain1, learning_rate, max_epochs=100,
                 l1=0.1, l2=0.1, gamma=0.5, lambda_param=.5):
        """
        Parameters
        ----------
        lambda : float
            Mediates between AE and LR. lambda==1 equates with LR only.
        """
        self.n_hidden = n_hidden
        self.gain1 = gain1
        self.max_epochs = max_epochs
        self.learning_rate = np.float32(learning_rate)
        self.penalty_l1 = np.float32(l1)
        self.penalty_l2 = np.float32(l2)
        self.gamma = np.float32(gamma)
        self.lambda_param = np.float32(lambda_param)
        
        self.max_norm = 4  # for gradient clipping (Ilya Sutskever)

    # def rectify(X):
    #     return T.maximum(0., X)

    from theano.tensor.shared_randomstreams import RandomStreams

    def RMSprop(self, cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            acc = theano.shared(p.get_value() * 0.)
            acc_new = rho * acc + (1 - rho) * g ** 2
            gradient_scaling = T.sqrt(acc_new + epsilon)
            g = g / gradient_scaling
            updates.append((acc, acc_new))
            updates.append((p, p - lr * g))
        return updates

    def RMSprop2(self, cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            acc = theano.shared(p.get_value() * 0.)
            acc_new = rho * acc + (1 - rho) * g ** 2
            gradient_scaling = T.sqrt(acc_new + epsilon)
            g = g / gradient_scaling
            dx = -(lr * g)
            if self.max_norm:  # parameter (!) clipping from Gabriel Synnaeve (avoid gradient explotion)
                W = p + dx  # W is the new parameter
                col_norms = W.norm(2, axis=0)
                desired_norms = T.clip(col_norms, 0, self.max_norm)
                # project weight cols on a multidimensional L2 ball
                updates.append((p, W * (desired_norms / (1e-6 + col_norms))))
            else:
                updates.append((p, p + dx))
            updates.append((acc, acc_new))
        return updates
        
    def get_param_pool(self):
        cur_params = (
            self.V1s, self.bV0, self.bV1,
            self.W0s, self.W1s, self.bW0s, self.bW1s
        )
        return cur_params

    def fit(self, X_task, y, sub_ids):
        DEBUG_FLAG = True

        # self.max_epochs = 333
        self.batch_size = 100
        n_input = X_task.shape[1]  # sklearn-like structure
        n_output = n_input
        rng = np.random.RandomState(42)
        self.input_taskdata = T.matrix(dtype='float32', name='input_taskdata')
        self.params_from_last_iters = []

        index = T.iscalar(name='index')
        
        # prepare data for theano computation
        if not DEBUG_FLAG:
            X_train_s = theano.shared(
                value=np.float32(X_task), name='X_train_s')
            y_train_s = theano.shared(
                value=np.int32(y), name='y_train_s')
        else:
            # from sklearn.cross_validation import StratifiedShuffleSplit
            # folder = StratifiedShuffleSplit(y, n_iter=1, test_size=0.20)
            # new_trains, inds_val = iter(folder).next()

            from sklearn.cross_validation import LeavePLabelOut
            folder = LeavePLabelOut(sub_ids, p=1)
            new_trains, inds_val = iter(folder).next()
            
            # valid_subs = np.array([1107, 1109, 1110, 1114,  # healthy
            #                        2105, 2106, 2113, 2125,  # depression
            #                        3280, 3279, 3276, 3275]) # sz
            # inds_val = np.in1d(sub_ids, valid_subs)
            # new_trains = np.logical_not(inds_val)
            
            print('Data points in train set: %i' % np.sum(new_trains))
            print('Data points in validation set: %i' % np.sum(inds_val))
            print('Data features: %i' % n_input)

            X_train, X_val = X_task[new_trains], X_task[inds_val]
            y_train, y_val = y[new_trains], y[inds_val]
            

            X_train_s = theano.shared(value=np.float32(X_train),
                                      name='X_train_s', borrow=False)
            y_train_s = theano.shared(value=np.int32(y_train),
                                      name='y_train_s', borrow=False)
            # X_val_s = theano.shared(value=np.float32(X_val),
            #                         name='X_train_s', borrow=False)
            # y_val_s = theano.shared(value=np.int32(y_val),
            #                         name='y_cal_s', borrow=False)
            self.dbg_epochs_ = list()
            self.dbg_acc_train_ = list()
            self.dbg_acc_val_ = list()
            self.dbg_ae_cost_ = list()
            self.dbg_lr_cost_ = list()
            self.dbg_ae_nonimprovesteps = list()
            self.dbg_acc_other_ds_ = list()
            self.dbg_combined_cost_ = list()
            self.dbg_prfs_ = list()
            self.dbg_prfs_other_ds_ = list()
        
        train_samples = len(X_train)

        # V -> supervised / logistic regression
        # W -> unsupervised / auto-encoder

        # computational graph: auto-encoder
        W0_vals = rng.randn(n_input, self.n_hidden).astype(np.float32) * self.gain1
        self.W0s = theano.shared(W0_vals)

        # self.W1s = self.W0s.T  # tied
        W1_vals = rng.randn(self.n_hidden, n_input).astype(np.float32) * self.gain1
        self.W1s = theano.shared(W1_vals)

        bW0_vals = np.zeros(self.n_hidden).astype(np.float32)
        self.bW0s = theano.shared(value=bW0_vals, name='bW0')
        bW1_vals = np.zeros(n_output).astype(np.float32)
        self.bW1s = theano.shared(value=bW1_vals, name='bW1')

        encoding = (self.input_taskdata.dot(self.W0s) + self.bW0s).dot(self.W1s) + self.bW1s

        self.ae_loss = T.sum((self.input_taskdata - encoding) ** 2, axis=1)

        self.ae_cost = (
            T.mean(self.ae_loss) / n_input
        )

        # params1 = [self.W0s, self.bW0s, self.bW1s]
        # gparams1 = [T.grad(cost=self.ae_cost, wrt=param1) for param1 in params1]
        # 
        # lr = self.learning_rate
        # updates = self.RMSprop(cost=self.ae_cost, params=params1,
        #                        lr=self.learning_rate)

        # f_train_ae = theano.function(
        #     [index],
        #     [self.ae_cost],
        #     givens=givens_ae,
        #     updates=updates)

        # computation graph: logistic regression
        clf_n_output = len(np.unique(y))
        print('SSFLogreg: Fitting %i classes' % clf_n_output)
        my_y = T.ivector(name='y')

        bV0_vals = np.zeros(self.n_hidden).astype(np.float32)
        self.bV0 = theano.shared(value=bV0_vals, name='bV0')
        bV1_vals = np.zeros(clf_n_output).astype(np.float32)
        self.bV1 = theano.shared(value=bV1_vals, name='bV1')
        
        # V0_vals = rng.randn(n_input, self.n_hidden).astype(np.float32) * self.gain1
        # self.V0s = theano.shared(V0_vals)
        V1_vals = rng.randn(self.n_hidden, clf_n_output).astype(np.float32) * self.gain1
        self.V1s = theano.shared(V1_vals)

        self.p_y_given_x = T.nnet.softmax(
            # T.dot(T.dot(self.input_taskdata, self.V0s) + self.bV0, self.V1s) + self.bV1
            T.dot(T.dot(self.input_taskdata, self.W0s) + self.bV0, self.V1s) + self.bV1
        )
        self.lr_cost = -T.mean(T.log(self.p_y_given_x)[T.arange(my_y.shape[0]), my_y])
        self.lr_cost = (
            self.lr_cost +
            T.mean(abs(self.W0s)) * self.penalty_l1 +
            # T.mean(abs(self.V0s)) * self.penalty_l1 +
            T.mean(abs(self.bV0)) * self.penalty_l1 +
            T.mean(abs(self.V1s)) * self.penalty_l1 +
            T.mean(abs(self.bV1)) * self.penalty_l1 +

            T.mean((self.W0s ** np.float32(2))) * self.penalty_l2 +
            # T.mean((self.V0s ** 2)) * self.penalty_l2 +
            T.mean((self.bV0 ** np.float32(2))) * self.penalty_l2 +
            T.mean((self.V1s ** np.float32(2))) * self.penalty_l2 +
            T.mean((self.bV1 ** np.float32(2))) * self.penalty_l2
        )
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # params2 = [self.V0s, self.bV0, self.V1s, self.bV1]
        # params2 = [self.W0s, self.bV0, self.V1s, self.bV1]
        # updates2 = self.RMSprop(cost=self.lr_cost, params=params2,
        #                         lr=self.learning_rate)

        # f_train_lr = theano.function(
        #     [index],
        #     [self.lr_cost],
        #     givens=givens_lr,
        #     updates=updates2)
        
        self.covar_cost = T.dot(self.W0s.T, self.W0s) - T.eye(self.W0s.shape[1])
        self.covar_cost = T.sum(T.sum(self.covar_cost ** 2, axis=1), axis=0)  # Frobenius

        # combined loss for AE and LR
        combined_params = [self.W0s, self.bW0s, self.bW1s, self.W1s,
                        #    self.V0s, self.V1s, self.bV0, self.bV1]
                           self.V1s, self.bV0, self.bV1]
        self.combined_cost = (
            (np.float32(1) - self.lambda_param) * self.ae_cost +
            self.lambda_param * self.lr_cost +
            self.gamma * self.covar_cost
        )
        combined_updates = self.RMSprop2(
            cost=self.combined_cost,
            params=combined_params,
            lr=self.learning_rate)
        givens_combined = {
            self.input_taskdata: X_train_s[index * self.batch_size:(index + 1) * self.batch_size],
            my_y: y_train_s[index * self.batch_size:(index + 1) * self.batch_size]
        }
        f_train_combined = theano.function(
            [index],
            # [self.combined_cost, self.ae_cost, self.lr_cost, self.lr_cost],
            [self.combined_cost, self.ae_cost, self.lr_cost, self.covar_cost],
            givens=givens_combined,
            updates=combined_updates, allow_input_downcast=False)

        # optimization loop
        start_time = time.time()
        ae_last_cost = np.inf
        lr_last_cost = np.inf
        no_improve_steps = 0
        acc_train, acc_val = 0., 0.
        for i_epoch in range(self.max_epochs):
            if i_epoch == 1:
                epoch_dur = time.time() - start_time
                total_mins = (epoch_dur * self.max_epochs) / 60
                hs, mins = divmod(total_mins, 60)
                print("Max estimated duration: %i hours and %i minutes" % (hs, mins))

            # AE
            n_batches = train_samples // self.batch_size
            for i in range(n_batches):
                # lr_cur_cost = f_train_lr(i)[0]
                # ae_cur_cost = lr_cur_cost
                combined_cost, ae_cur_cost, lr_cur_cost, covar_cost = f_train_combined(i)

            # evaluate epoch cost
            if ae_last_cost - ae_cur_cost < 0.1:
                no_improve_steps += 1
            else:
                ae_last_cost = ae_cur_cost
                no_improve_steps = 0

            # logistic
            lr_last_cost = lr_cur_cost
            acc_train = self.score(X_train, y_train)
            acc_val, prfs_val = self.score(X_val, y_val, return_prfs=True)

            print('E:%i, ae_cost:%.4f, lr_cost:%.4f, covar_cost:%.4f, train_score:%.2f, vald_score:%.2f, ae_badsteps:%i' % (
                i_epoch + 1, ae_cur_cost, lr_cur_cost, covar_cost, acc_train, acc_val, no_improve_steps))

            # if (i_epoch % 10 == 0):
            self.dbg_ae_cost_.append(ae_cur_cost)
            self.dbg_lr_cost_.append(lr_cur_cost)
            self.dbg_combined_cost_.append(combined_cost)

            self.dbg_epochs_.append(i_epoch + 1)
            self.dbg_ae_nonimprovesteps.append(no_improve_steps)
            self.dbg_acc_train_.append(acc_train)
            self.dbg_acc_val_.append(acc_val)
            self.dbg_prfs_.append(prfs_val)
                
            # save paramters from last 100 iterations
            # if i_epoch > (self.max_epochs - 100):
            #     print('Param pool!')
            param_pool = self.get_param_pool()
            self.params_from_last_iters.append(param_pool)

        total_mins = (time.time() - start_time) / 60
        hs, mins = divmod(total_mins, 60)
        print("Final duration: %i hours and %i minutes" % (hs, mins))

        return self

    def predict(self, X):
        X_test_s = theano.shared(value=np.float32(X), name='X_test_s', borrow=True)

        givens_te = {
            self.input_taskdata: X_test_s
        }

        f_test = theano.function(
            [],
            [self.y_pred],
            givens=givens_te)
        predictions = f_test()
        del X_test_s
        del givens_te
        return predictions[0]

    def score(self, X, y, return_prfs=False):
        pred_y = self.predict(X)
        acc = np.mean(pred_y == y)
        prfs = precision_recall_fscore_support(pred_y, y)
        if return_prfs:
            return acc, prfs
        else:
            return acc


##############################################################################
# plot figures
##############################################################################

def dump_comps(masker, compressor, components, threshold=2, fwhm=None,
               perc=None):
    from scipy.stats import zscore
    from nilearn.plotting import plot_stat_map
    from nilearn.image import smooth_img
    from scipy.stats import scoreatpercentile

    if isinstance(compressor, basestring):
        comp_name = compressor
    else:
        comp_name = compressor.__str__().split('(')[0]

    for i_c, comp in enumerate(components):
        path_mask = op.join(WRITE_DIR, '%s_%i-%i' % (comp_name,
                                                     n_comp, i_c + 1))
        nii_raw = masker.inverse_transform(comp)
        nii_raw.to_filename(path_mask + '.nii.gz')
        
        comp_z = zscore(comp)
        
        if perc is not None:
            cur_thresh = scoreatpercentile(np.abs(comp_z), per=perc)
            path_mask += '_perc%i' % perc
            print('Applying percentile %.2f (threshold: %.2f)' % (perc, cur_thresh))
        else:
            cur_thresh = threshold
            path_mask += '_thr%.2f' % cur_thresh
            print('Applying threshold: %.2f' % cur_thresh)

        nii_z = masker.inverse_transform(comp_z)
        gz_path = path_mask + '_zmap.nii.gz'
        nii_z.to_filename(gz_path)
        plot_stat_map(gz_path, bg_img='colin.nii', threshold=cur_thresh,
                      cut_coords=(0, -2, 0), draw_cross=False,
                      output_file=path_mask + 'zmap.png')
                      
        # optional: do smoothing
        if fwhm is not None:
            nii_z_fwhm = smooth_img(nii_z, fwhm=fwhm)
            gz_mm_path = path_mask + '_zmap_%imm.nii.gz' % fwhm
            nii_z_fwhm.to_filename(gz_mm_path)
            plot_stat_map(nii_z_fwhm, bg_img='colin.nii', threshold=cur_thresh,
                          cut_coords=(0, -2, 0), draw_cross=False,
                          output_file=path_mask +
                          ('zmap_%imm.png' % fwhm))

n_comps = [10]
# n_comps = [40, 30, 20, 10, 5]
for n_comp in n_comps:
    # for lambda_param in [0]:
    for lambda_param in [1.0]:
        l1 = 0.1  # sparsity
        l2 = 0.1  # shrinkage
        cur_gamma = 2  # orthogonality
        my_title = r'Low-rank LR + AE (combined loss, shared decomp): n_comp=%i L1=%.1f L2=%.1f gamma=%.1f lambda=%.2f res=3mm ero2' % (
            n_comp, l1, l2, cur_gamma, lambda_param
        )
        print(my_title)
        estimator = SSEncoder(
            n_hidden=n_comp,
            gain1=0.004,  # empirically determined by CV
            learning_rate = np.float32(0.0001),  # empirically determined by CV,
            max_epochs=500, l1=l1, l2=l2,
            gamma=cur_gamma, lambda_param=lambda_param)
        
        estimator.fit(FS, labels, sub_labels)

        fname = my_title.replace(' ', '_').replace('+', '').replace(':', '').replace('__', '_').replace('%', '')
        cur_path = op.join(WRITE_DIR, fname)
        joblib.dump(estimator, cur_path, compress=9)
        # estimator = joblib.load(cur_path)
        # plt.savefig(cur_path + '_SUMMARY.png', dpi=200)
        
        # dump data also as numpy array
        np.save(cur_path + 'dbg_epochs_', np.array(estimator.dbg_epochs_))
        np.save(cur_path + 'dbg_acc_train_', np.array(estimator.dbg_acc_train_))
        np.save(cur_path + 'dbg_acc_val_', np.array(estimator.dbg_acc_val_))
        np.save(cur_path + 'dbg_ae_cost_', np.array(estimator.dbg_ae_cost_))
        np.save(cur_path + 'dbg_lr_cost_', np.array(estimator.dbg_lr_cost_))
        np.save(cur_path + 'dbg_ae_nonimprovesteps', np.array(estimator.dbg_ae_nonimprovesteps))
        np.save(cur_path + 'dbg_acc_other_ds_', np.array(estimator.dbg_acc_other_ds_))
        np.save(cur_path + 'dbg_combined_cost_', np.array(estimator.dbg_combined_cost_))
        np.save(cur_path + 'dbg_prfs_', np.array(estimator.dbg_prfs_))
        np.save(cur_path + 'dbg_prfs_other_ds_', np.array(estimator.dbg_prfs_other_ds_))

        W0_mat = estimator.W0s.get_value().T
        np.save(cur_path + 'W0comps', W0_mat)
        
        V1_mat = estimator.V1s.get_value().T
        np.save(cur_path + 'V1comps', V1_mat)
        # dump_comps(nifti_masker, fname, comps, threshold=0.5)

STOP_CALCULATION

# print network components (1st layer)
from nilearn.image import smooth_img
nifti_masker = masker
n_comp = 10
lmbd = 1.0
gam = 2.0
TH = 1.0
pkgs = glob.glob(RES_NAME + '/*ero2W0comps.npy')
for p in pkgs:
    lambda_param = np.float(re.search('lambda=(.{4})', p).group(1))
    n_hidden = int(re.search('comp=(.{1,3})_', p).group(1))
    gamma_param = np.float(re.search('gamma=(.{3,4})_', p).group(1))
    if n_comp != n_hidden or lambda_param != lmbd or gamma_param != gam:
        continue
        
    new_fname = 'comps_n=%i_lambda=%.2f_gamma=%.2f_th%.1f' % (n_hidden, lambda_param, gamma_param, TH)
    comps = np.load(p)
    dump_comps(nifti_masker, new_fname, comps, threshold=TH, fwhm=4)


class_names = ['verbs', 'nouns', 'objects', 'objects_hand']
n_classes = len(class_names)

def plot_coefficients(coef, out_fpath):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(9, 5))
    
    masked_data = np.ma.masked_where(coef != 0., coef)
    plt.imshow(masked_data, interpolation='nearest', cmap=plt.cm.gray_r)
    masked_data = np.ma.masked_where(coef == 0., coef)
    plt.imshow(masked_data, interpolation='nearest', cmap=plt.cm.RdBu_r)

    n_classes, n_comps = coef.shape
    plt.xticks(
        range(n_comps),
        (np.arange(n_comps) + 1))
    plt.yticks(range(n_classes), class_names)

    plt.ylabel('class')
    plt.xlabel('latent component')
    plt.grid(False)
    plt.colorbar()
    # plt.title(cur_title, {'fontsize': 16})
    plt.savefig(out_fpath)
    plt.show()

from nilearn.image import smooth_img
nifti_masker = masker
n_comp = 10
lmbd = 1.0
gam = 2.0
TH = 1.0
pkgs = glob.glob(RES_NAME + '/*ero2V1comps.npy')
for p in pkgs:
    lambda_param = np.float(re.search('lambda=(.{4})', p).group(1))
    n_hidden = int(re.search('comp=(.{1,3})_', p).group(1))
    gamma_param = np.float(re.search('gamma=(.{3,4})_', p).group(1))
    if n_comp != n_hidden or lambda_param != lmbd or gamma_param != gam:
        continue
        
    new_fname = 'comps_n=%i_lambda=%.2f_gamma=%.2f_th%.1f' % (n_hidden, lambda_param, gamma_param, TH)
    comps = np.load(p)
    dump_comps(nifti_masker, new_fname, comps, threshold=TH, fwhm=4)


# aesthetics
# comps[3], comps[2] = comps[2], comps[3]
# class_names[3], class_names[2] = class_names[2], class_names[3]

plot_coefficients(comps, 'ssflogreg_xcov/_taskweights.png')
plot_coefficients(comps, 'ssflogreg_xcov/_taskweights.pdf')





