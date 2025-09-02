import os
import glob
import shutil
import subprocess
import time
import gc
import psutil
import numpy as np
import mne
from mne.parallel import parallel_func
import nibabel as nib
import pandas as pd
from mne.preprocessing import maxwell_filter
print('MNE version:', mne.__version__)
from mne.viz import plot_bem
import matplotlib.pyplot as plt
from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs, read_ica
from scipy.io import savemat
from mne.decoding import SlidingEstimator, cross_val_multiscore
from mne.coreg import Coregistration
from mne.io import read_info
from mne.minimum_norm import (make_inverse_operator, apply_inverse, apply_inverse_epochs, apply_inverse_raw,write_inverse_operator)
from functools import partial
from scipy import stats
from mne.stats import (spatio_temporal_cluster_1samp_test,summarize_clusters_stc, ttest_1samp_no_p)
