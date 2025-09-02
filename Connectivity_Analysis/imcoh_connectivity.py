#read the morphed stcs files for each epoch and then use them for connectivity estimation

fmin, fmax = 4, 7  # frequency band
con_method = "imcoh"  # imaginary coherence method for connectivity estimation
smooth = 10

subjects_list = pd.read_csv("allsubj.csv", header=0)
subjects_dir = 'processed_t1_freesurfer_output/'
morph_dir = "Resting/9_morphing/"
output_conn_dir = "Resting/11_connectivity_imaginary_coherence/"

fname_fsaverage_src = os.path.join(subjects_dir, "fsaverage", "bem", "fsaverage-5-src.fif")
os.makedirs(output_conn_dir, exist_ok=True)

# Load labels in fsaverage
labels = mne.read_labels_from_annot("fsaverage", parc="aparc", subjects_dir=subjects_dir)
labels = [label for label in labels if 'unknown' not in label.name.lower()]

for i in range(subjects_list.shape[0]):
    print_memory_usage()
    subject_id = subjects_list.loc[i, 'Id']
    print(f"\nProcessing subject: {subject_id}")

    morph_out_dir = os.path.join(morph_dir, subject_id)
    
    stc_files = sorted([f for f in os.listdir(morph_out_dir) if f.startswith("mne_dSPM_inverse_morph_epoch") and f.endswith(".stc") ])

    stc_list = []
    for fname in stc_files:
        stc_path = os.path.join(morph_out_dir, fname)
        stc_list.append(mne.read_source_estimate(stc_path))
    
    # --- Extract label time series ---
    label_ts = mne.extract_label_time_course(
        stc_list, labels,
        src=mne.read_source_spaces(fname_fsaverage_src),
        mode='mean_flip',
        return_generator=False
    )
    label_ts = np.array(label_ts)
    print(f"Shape of time course (all labels): {label_ts.shape}")

    # z-score normalization
    label_ts = (label_ts - label_ts.mean(axis=2, keepdims=True)) / label_ts.std(axis=2, keepdims=True)
    del stc_list

    # --- Compute connectivity ---
    con = spectral_connectivity_epochs(
        data=label_ts,
        method=con_method,
        sfreq=sfreq_target,
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        mode='fourier',
        n_jobs=12,
        verbose=True
    )

    con_mat = con.get_data(output='dense')
    if con_mat.ndim == 3:
        con_mat = con_mat[:, :, 0]

    # --- Save connectivity as CSV ---
    df = pd.DataFrame(
        con_mat,
        columns=[label.name for label in labels],
        index=[label.name for label in labels]
    )
    fname_conn_out = os.path.join(output_conn_dir, f"{subject_id}_imcoh_theta_resting.csv")
    df.to_csv(fname_conn_out)
    print(f"Saved connectivity for {subject_id} -> {fname_conn_out}")

    # --- Cleanup ---
    del label_ts, con, df, con_mat
    gc.collect()
