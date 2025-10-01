#read the morphed stcs files for each epoch and then use them for connectivity estimation

# Parameters
sfreq_target = 250

con_method = "imcoh"  # imaginary coherence method for connectivity estimation

# Define frequency bands
freq_bands = {
    "theta": (4, 7),
    "alpha": (8, 13),
    "lowbeta": (15, 20),
    "highbeta": (20, 30),
    "lowgamma": (30, 55),
    "highgamma": (65, 95)
}


# Directories
subjects_list = pd.read_csv("allsubj.csv", header=0)
subjects_dir = "processed_t1_freesurfer_output/"
inverse_dir = "8_inverse_solution"
output_conn_dir = "11_connectivity_imaginary_coherence/"
fname_fsaverage_src = os.path.join(subjects_dir, "fsaverage", "bem", "fsaverage-5-src.fif")
os.makedirs(output_conn_dir, exist_ok=True)

for i in range(subjects_list.shape[0]):
    print_memory_usage()
    subject_id = subjects_list.loc[i, "Id"]
    print(f"\nProcessing subject: {subject_id}")
    
    # --- Read inverse operator for the subject ---
    fname_inv = os.path.join(inverse_dir, subject_id, "meg-inv.fif")
    inv = read_inverse_operator(fname_inv)
    print(f"Loaded inverse operator for {subject_id}")
   
    # Load labels using aparc FreeSurfer parcellation
    labels = mne.read_labels_from_annot(subject_id, parc="aparc", subjects_dir=subjects_dir)
    labels = [label for label in labels if "unknown" not in label.name.lower()]
    
   # print(labels)
   
    native_out_dir = os.path.join(inverse_dir, subject_id, "native_stcs")
    stc_files = sorted([f for f in os.listdir(native_out_dir) if f.startswith("native_dSPM_inverse_epoch") and f.endswith(".stc")])

    print(f"Found {len(stc_files)} STCs for subject {subject_id}")
    stc_list = []
    for fname in stc_files:
        stc_path = os.path.join(native_out_dir, fname)
        stc_list.append(mne.read_source_estimate(stc_path))

    # --- Extract label time series ---
    label_ts = mne.extract_label_time_course(
        stc_list, labels,
        src=inv["src"] ,
     #   src=mne.read_source_spaces(fname_fsaverage_src), commented out as we use subject's inverse 
        mode="mean_flip",
        return_generator=False,
    )
    label_ts = np.array(label_ts)
    print(f"Shape of time course (all labels): {label_ts.shape}")  # (n_epochs, n_labels, n_times)

    # Compute connectivity for each frequency band
    for band_name, (fmin, fmax) in freq_bands.items():
        print(f"Computing {con_method} connectivity for {band_name} ({fmin}-{fmax} Hz)")
        con = spectral_connectivity_epochs(
            data=label_ts,
            method=con_method,
            sfreq=sfreq_target,
            fmin=fmin,
            fmax=fmax,
            faverage=True,
            mode="fourier",
            n_jobs=12,
            verbose=True
        )

        con_mat = con.get_data(output="dense")
        if con_mat.ndim == 3:
            con_mat = con_mat[:, :, 0]

        # Save connectivity
        df = pd.DataFrame(con_mat,
                          columns=[label.name for label in labels],
                          index=[label.name for label in labels])
        fname_conn_out = os.path.join(output_conn_dir, f"{subject_id}_{con_method}_{band_name}_resting.csv")
        df.to_csv(fname_conn_out)
        print(f"Saved connectivity for {subject_id} ({band_name}) -> {fname_conn_out}")

        del con, con_mat, df
        gc.collect()

    # Cleanup after each subject
    del label_ts, stc_list
    gc.collect()
