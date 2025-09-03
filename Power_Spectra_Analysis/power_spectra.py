#This script calculates frequency-bands power in source space for each brain region according to FreeSurfer cortical parcellation (aparc)

#defining paths
subjects_list = pd.read_csv("allsubj.csv", header=0)
subjects_dir = "processed_t1_freesurfer_output/"
output_files = "Resting/morphing/"
fname_fsaverage_src = os.path.join(subjects_dir, "fsaverage", "bem", "fsaverage-5-src.fif")
save_dir = "Resting/power_spectra_files"
os.makedirs(save_dir, exist_ok=True)

# Load labels and source space
labels = mne.read_labels_from_annot("fsaverage", parc="aparc", subjects_dir=subjects_dir)
# Remove unknown labels before extraction
labels = [label for label in labels if 'unknown' not in label.name.lower()]
print(f"Labels used for extraction: {[label.name for label in labels]}")

src = mne.read_source_spaces(fname_fsaverage_src)

alpha_power_all = []  # will store dicts for each subject

for i in range (subjects_list.shape[0]):
    print_memory_usage()

    subj_id = subjects_list.loc[i, 'Id']
    stc_fname = os.path.join(output_files, subj_id, "mne_dSPM_inverse_morph")
    
    # Load morphed STC
    stc = mne.read_source_estimate(stc_fname, subject="fsaverage")
    print(f"shape of source time course: {stc.shape}")

    # Extract label time series from each ROI
    ts = mne.extract_label_time_course(stc, labels, src, mode="mean_flip")
    print(f"label time series shape: {ts.shape}")

    # Z-score normalize each ROI time series (axis=1 for each ROI)
    ts_z = zscore(ts, axis=1)
    
    # Define epoching parameters
    sfreq = stc.sfreq
    epoch_length_sec = 5
    samples_per_epoch = int(epoch_length_sec * sfreq)
    n_labels, n_times = ts_z.shape
    n_epochs = n_times // samples_per_epoch

    # Truncate time series to have integer number of epochs---it doesnt change a lot since timeseries is already 330s*250Hz
    ts_z = ts_z[:, :n_epochs * samples_per_epoch]

    # Reshape into epochs: (n_epochs, n_labels, n_times_per_epoch)
    epochs_data = ts_z.reshape(n_labels, n_epochs, samples_per_epoch)
    epochs_data = np.transpose(epochs_data, (1, 0, 2))  # (n_epochs, n_labels, n_times_per_epoch)
    print(f"Epochs data shape: {epochs_data.shape}")

    # Compute PSD per ROI and epoch using Welch's method
    fmin, fmax = 4, 7  # theta band
    alpha_power = np.zeros((n_labels, n_epochs))
    alpha_power_avg = np.zeros(n_labels)
    for r in range(n_labels):
        # Extract all epochs for this ROI: shape (n_epochs, n_times_per_epoch)
        roi_epochs = epochs_data[:, r, :]
    
        # Compute PSD for all epochs at once
        psd, freqs = psd_array_welch(
            roi_epochs,
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            n_fft=samples_per_epoch,      # full epoch length
            n_per_seg=samples_per_epoch,  # one segment per epoch
            n_overlap=0,                  # non-overlapping
            window='hamming',
            average='mean',               # average across segment(s)
            remove_dc=True,
            verbose=False
            )
    
    # psd shape: (n_epochs, n_freqs) since average='mean' averages within each segment
    # Take mean across frequencies and epochs to get a single value per ROI
        alpha_power_avg[r] = np.mean(psd)

    # Store results as dict: subject ID + one alpha power per ROI label
    subj_data = {
        "subject": subj_id,
        **{labels[i].name: alpha_power_avg[i] for i in range(len(labels))}
    }

    # Save individual subject alpha power CSV
    save_path = os.path.join(save_dir, f"theta_power_{subj_id}.csv")
    pd.DataFrame([subj_data]).to_csv(save_path, index=False)
    print(f"Saved theta power for {subj_id} at {save_path}")

    # Clean up
    del stc, ts, ts_z, epochs_data, alpha_power, subj_data, psd, freqs

    gc.collect()
    print_memory_usage()

print("Alpha power extraction and individual saving completed.")
