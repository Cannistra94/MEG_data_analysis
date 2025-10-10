subjects_list=pd.read_csv("allsubj.csv", header=0)
subjects_dir = 'processed_t1_freesurfer_output/'
raw_dir = 'Resting/2_ica_decomposition/'
inverse_dir = "8_inverse_solution"
os.makedirs(output_conn_dir, exist_ok=True)
os.makedirs(morph_dir, exist_ok=True)

# Parameters
sfreq_target = 250
epoch_length_sec = 5.0
snr = 3.0
lambda2 = 1.0 / snr ** 2
method = "dSPM"
fmin, fmax = 4, 7  # alpha/theta band
con_method = "imcoh"  # imaginary coherence
smooth = 10

for i in range(subjects_list.shape[0]):
    
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
    
    raw.resample(sfreq=sfreq_target)  
    
    # --- Fixed-length epochs ---
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
      
    stc_list = []
   
    for idx, stc in enumerate(stcs):
        
        # --- Save native STC before morphing ---
        native_out_dir = os.path.join(inverse_dir, subject_id, "native_stcs")
        os.makedirs(native_out_dir, exist_ok=True)
        stc.save(
            os.path.join(native_out_dir, f'native_dSPM_inverse_epoch{idx:03d}_{stc.subject}'),
            overwrite=True
        )


