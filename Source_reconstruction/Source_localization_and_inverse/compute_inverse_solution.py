subjects_list=pd.read_csv("allsubj.csv", header=0)
subjects_dir = 'processed_t1_freesurfer_output/'
raw_dir = 'Resting/2_ica_decomposition/'
inverse_dir = "Resting/8_inverse_solution"
morph_dir = "Resting/9_morphing/"

fname_fsaverage_src = os.path.join(subjects_dir, "fsaverage", "bem", "fsaverage-5-src.fif")
os.makedirs(output_conn_dir, exist_ok=True)
os.makedirs(morph_dir, exist_ok=True)

# Parameters
sfreq_target = 250
epoch_length_sec = 5.0
snr = 3.0
lambda2 = 1.0 / snr ** 2
method = "dSPM"
fmin, fmax = 4, 7  # frequency band
con_method = "imcoh"  # imaginary coherence
smooth = 10

# Load labels in fsaverage
labels = mne.read_labels_from_annot("fsaverage", parc="aparc", subjects_dir=subjects_dir)
labels = [label for label in labels if 'unknown' not in label.name.lower()]

for i in range(subjects_list.shape[0]):
    print_memory_usage()

    subject_id = subjects_list.loc[i, 'Id']
    print(f"\nProcessing subject: {subject_id}")

    # File paths
    fname_raw = os.path.join(raw_dir, subject_id, 'raw_sss_ica_cleaned.fif')
    fname_inv = os.path.join(inverse_dir, subject_id, 'meg-inv.fif')
    fname_conn_out = os.path.join(output_conn_dir, f"{subject_id}_imcoh.npy")

    # --- Read inverse operator ---
    inv = read_inverse_operator(fname_inv)
    
    
    # --- Read raw data ---
    raw = mne.io.read_raw_fif(fname_raw, preload=True)
    raw.crop(tmin=0, tmax=330)
    raw.resample(sfreq=sfreq_target)  # resampling ICA cleaned data to 250Hz
    
    
    # --- Fixed-length epochs for stcs extraction---
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=epoch_length_sec,
        preload=True,
        reject_by_annotation=True,
        proj=True,
        overlap=0.0
        )

    del raw  # free memory
    
    # --- Apply inverse to epochs ---
    stcs = apply_inverse_epochs(epochs, inv, lambda2, method,pick_ori="normal", return_generator=True)
    
    del epochs

    #save subject's STCs before morphing (will be used for connectivity estimation)
    stc_list = []
    native_out_dir = os.path.join (inverse_dir, subject_id, 'native_stcs')
    for idx, stc in enumerate(stcs):
        print(f'start morphing epoch {idx+1}')
        # --- Save native STC before morphing ---
        native_out_dir = os.path.join(inverse_dir, subject_id, "native_stcs")
        os.makedirs(native_out_dir, exist_ok=True)
        stc.save(
            os.path.join(native_out_dir, f'native_dSPM_inverse_epoch{idx:03d}_{stc.subject}'),
            overwrite=True
        )

    # --- Morph STCs to fsaverage --- save also morphed STCs in case needed
    src_fsaverage = mne.read_source_spaces(fname_fsaverage_src)
       
    morph_out_dir = os.path.join(morph_dir, subject_id)
    os.makedirs(morph_out_dir, exist_ok=True)
    stc_list = []
    morph_mat = None
    for idx, stc in enumerate(stcs):
        print(f'start morphing epoch {idx+1}')
        if morph_mat is None:
            morph_mat = mne.compute_source_morph(
                stc,
                subject_from=subject_id,
                subject_to="fsaverage",
                src_to=src_fsaverage,
                subjects_dir=subjects_dir,
                smooth=smooth
            )
        stc_fsaverage = morph_mat.apply(stc)
        stc_list.append(stc_fsaverage)
    
        # Save each morphed STC with epoch index
        stc_fsaverage.save(
        os.path.join(morph_out_dir, f'mne_dSPM_inverse_morph_epoch{idx:03d}_{stc.subject}'),
        overwrite=True)
        del stc, stc_fsaverage

