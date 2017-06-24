"""
Robustness analysis: split half analysis
"""

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
from matplotlib import pylab as plt
from nilearn.input_data import NiftiMasker
from nilearn.image import concat_imgs
import joblib
import time
from unpack_mat import loadmatnow
import re

RES_NAME = 'mns_mask'
WRITE_DIR = op.join(os.getcwd(), RES_NAME)
if not op.exists(WRITE_DIR):
    os.mkdir(WRITE_DIR)
    
##############################################################################
# load+preprocess data
##############################################################################

print('Loading data...')

# load the information + niftis from SPM analyses
mat_paths = glob.glob('/Volumes/TRESOR/houpand/1st_level_unnormalized/*/SPM.mat')


DMP_FNAME = 'FS_labels_sublabels_dump__'
if op.exists(DMP_FNAME):
    FS, labels, sub_labels = joblib.load(DMP_FNAME)
    masker = NiftiMasker(mask_img=('debug_mask.nii'))
    masker.fit()
else:
    # load the data from scratch
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
    n_vox = r_mask.get_data().sum()


    n_files = len(nii_paths)

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
    
    stop

    joblib.dump([FS, labels, sub_labels], DMP_FNAME)



# HACK! define the tasks to predict
inds_verbs = np.logical_or(np.logical_or(labels == 0, labels == 1), labels == 2)
inds_nomen = np.logical_or(np.logical_or(labels == 3, labels == 4), labels == 5)
inds_objects = np.logical_or(np.logical_or(labels == 6, labels == 7), labels == 8)
inds_objects_hand = np.logical_or(np.logical_or(labels == 9, labels == 10), labels == 11)


labels[inds_verbs] = 0
labels[inds_nomen] = 1

labels[inds_objects_hand] = 0  # should show similar patterns to verbs
labels[inds_objects] = 1  # should show similar patterns to nouns

train_inds = np.logical_or(inds_verbs, inds_nomen)
test_inds = np.logical_or(inds_objects_hand, inds_objects)

##############################################################################
# compute
##############################################################################

FS_brain = masker.inverse_transform(FS)

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_selection import RFE
from nilearn.input_data import NiftiMasker

feature_mask = 'masks/rObservation_AND_Imitation.img'
# feature_mask = 'masks/rObservation.nii'
# feature_mask = 'masks/rImitation.nii'

# RFE
feature_mask_nii = nib.Nifti1Image(
    np.array(np.nan_to_num(nib.load(feature_mask).get_data()) > 0, dtype=np.int32),
    nib.load(feature_mask).get_affine()
)
meta_mask = NiftiMasker(mask_img=feature_mask_nii, smoothing_fwhm=False,
                          standardize=False)
meta_mask.fit()
n_feat = np.sum(feature_mask_nii.get_data())
print('Voxel in feature mask: %i' % n_feat)

FS_mask = meta_mask.transform(FS_brain)

clf = LogisticRegression(
    penalty='l2',
    multi_class='ovr',
    verbose=0, C=5.0
)

from scipy.stats import zscore
from scipy.stats import binom_test
import copy

n_iter = 100
it_accs = []
it_pvalue = []
org_train_inds = np.where(train_inds == 1)[0]
org_test_inds = np.where(test_inds == 1)[0]

for it in range(n_iter):
    train_inds = copy.copy(org_train_inds)
    test_inds = copy.copy(org_test_inds)
    np.random.shuffle(train_inds)
    np.random.shuffle(test_inds)
    train_inds = train_inds[:1200]
    test_inds = test_inds[:1200]
    # for k in [80]:
    best_pvalue = 1
    best_acc = 0
    for k in np.arange(0, 1000, 25)[1:]:
        print('-' * 80)
        print('k=%i' % k)
        
        selector = RFE(clf, n_features_to_select=k, step=0.5, verbose=00)
        selector = selector.fit(FS_mask[train_inds],
                                labels[train_inds])


        clf.fit(FS_mask[train_inds][:, selector.support_],
                labels[train_inds])

        acc = clf.score(FS_mask[test_inds][:, selector.support_],
                        labels[test_inds])
        print('Total acc: %.3f' % acc)

        # dump
        meta_space = np.zeros(selector.support_.shape, dtype=np.float32)
        meta_space[selector.support_] = clf.coef_
        brain_coef_nii = meta_mask.inverse_transform(meta_space)
        brain_coef_nii.to_filename(
            'train_verbs_nomen_predict_hand_objects_0.54accuracy.nii.gz')

        meta_space[selector.support_] = clf.coef_
        meta_space = zscore(meta_space)
        brain_coef_nii = meta_mask.inverse_transform(meta_space)
        brain_coef_nii.to_filename(
            'train_verbs_nomen_predict_hand_objects_0.54accuracy_zscore.nii.gz')
            
        # assess significance
        pred_y = clf.predict(FS_mask[test_inds][:, selector.support_])
        test_y = labels[test_inds]
        
        n_trials = len(test_y)  # 2400
        n_hits = np.sum(test_y == pred_y)  # 1292
        p_value = binom_test(x=n_hits, n=n_trials, p=0.5)
        
        best_pvalue = min(p_value, best_pvalue)
        best_acc = max(acc, best_acc)
            
    print(p_value)
    it_pvalue.append(best_pvalue)
    print(acc)
    it_accs.append(best_acc)
DONE

# k-wise prediction
for k in np.arange(0, 2500, 50)[1:]:
    print('-' * 80)
    print('k=%i' % k)
    
    my_shape = nib.load(feature_mask).get_data().shape
    my_data_1d = np.nan_to_num(nib.load(feature_mask).get_data().ravel())
    my_data_new = np.zeros_like(my_data_1d)
    hot_inds = np.argsort(my_data_1d)[::-1][:k]
    my_data_new[hot_inds] = my_data_1d[hot_inds]

    feature_mask_nii = nib.Nifti1Image(
        np.array(my_data_new.reshape(my_shape) > 0, dtype=np.int32),
        nib.load(feature_mask).get_affine()
    )
    feature_mask_nii.to_filename('dbg_feat_mask.nii.gz')

    feat_masker = NiftiMasker(mask_img=feature_mask_nii, smoothing_fwhm=False,
                              standardize=False)
    feat_masker.fit()
    n_feat = np.sum(feature_mask_nii.get_data())
    print('Voxel in feature mask: %i' % n_feat)

    FS_mask = feat_masker.transform(FS_brain)

    accs = []
    for _ in np.arange(25):
        inds_bs_sample = np.random.randint(0, len(labels[train_inds]), len(labels[train_inds]))

        clf = LogisticRegression(
            penalty='l2',
            multi_class='ovr',
            verbose=0, C=5.0
        )

        clf.fit(FS_mask[train_inds][inds_bs_sample], labels[train_inds][inds_bs_sample])
        acc = clf.score(FS_mask[test_inds], labels[test_inds])
        # print acc
        accs.append(acc)

    print('Total acc: %.3f' % np.mean(accs))






