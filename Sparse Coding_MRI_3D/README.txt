**********************************************
* Matlab Codes For MRI Data Super-resolution *
**********************************************

Adapted From Demo Codes For Image Super-resolution via Sparse Representation

Reference

J. Yang et al. Image super-resolution as sparse representation of raw image patches. CVPR 2008.

J. Yang et al. Image super-resolution via sparse representation. IEEE Transactions on Image Processing, Vol 19, Issue 11, pp2861-2873, 2010

MRI_Train_Dictionary.m: Training the dictionaries
MRI_Super_Resolution.m: Perform super-resolution

Files named by old_*** are no longer be used.

==============================================

External function dependency chain:
MRI_Train_Dictionary.m
  |--load_untouch_nii()  % NIfTI_20140122 matlab toolbox
  |--sample_patches()
  |	   |--affine()       % NIfTI_20140122 matlab toolbox
  |--patch_pruning()
  |--train_coupled_dict
	   |--reg_sparse_coding()
			|--L1QP_FeatureSign_Set()
			|    |--L1QP_FeatureSign_yang()
			|--getObjective_RegSc()
			|--l2ls_learn_basis_dual()

MRI_Super_Resolution.m
  |--load_untouch_nii()  % NIfTI_20140122 matlab toolbox
  |--affine()            % NIfTI_20140122 matlab toolbox
  |--ScSR()
  