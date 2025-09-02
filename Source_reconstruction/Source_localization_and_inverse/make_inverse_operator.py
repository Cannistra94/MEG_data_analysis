subjects_list=pd.read_csv("allsubj.csv", header=0)
subjects_dir='MEG_processed/Resting/2_ica_decomposition/'
cov_dir= 'MEG_processed/empty_room'
fwd_dir="Resting/forward_solution/"
freesurfer_subjects_dir = "processed_t1_freesurfer_output/"
output_dir="Resting/inverse_solution"
mindist = 5

for i in range (subjects_list.shape[0]):
    print_memory_usage()
     
    subject_id= subjects_list.loc[i,'Id']
    print("processing subject:", subject_id)
    subject_dir = os.path.join(subjects_dir, subject_id)
    
    fname_ave = os.path.join(subject_dir, 'raw_sss_ica_cleaned.fif') #read the raw instead
    raw =  mne.io.read_raw_fif(fname_ave)
    #resampling ICA cleaned data
    raw.resample(sfreq=250)
    fname_cov = os.path.join(cov_dir, 'noise_cov_matrix_empty_room-cov.fif' )
    
    fname_fwd = os.path.join(fwd_dir, subject_id, 'meg-fwd.fif')
    
    fname_inv_folder = os.path.join(output_dir, subject_id)
    os.makedirs(fname_inv_folder, exist_ok=True)
    
    fname_inv = os.path.join(fname_inv_folder, 'meg-inv.fif')

    cov = mne.read_cov(fname_cov)
    forward = mne.read_forward_solution(fname_fwd)
    
     
    info = raw.info

    inverse_operator = make_inverse_operator(info, forward, cov, loose=0.2, depth=0.8)
    del forward
    
    write_inverse_operator(fname_inv, inverse_operator, overwrite=True)
