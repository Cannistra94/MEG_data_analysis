#create noise covariance matrix from empty room recording
dir="MEG_processed/empty_room/"

fname = os.path.join(dir, 'sss_filtering_raw.fif')

raw = mne.io.read_raw_fif(fname, verbose=False)

meg_picks = mne.pick_types(raw.info, meg=True, eeg=False)

fname_cov = os.path.join(dir, 'noise_cov_matrix_empty_room-cov.fif' )

noise_cov=mne.compute_raw_covariance (raw, picks=meg_picks)
noise_cov.save(fname_cov, overwrite=True)
