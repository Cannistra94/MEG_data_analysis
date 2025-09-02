subjects_list=pd.read_csv("allsubj.csv", header=0)
subjects_dir='MEG_processed/Resting/2_ica_decomposition/'
freesurfer_subjects_dir = "processed_t1_freesurfer_output/"
output_dir="MEG_processed/Resting/forward_solution/"
coreg_files= 'Resting/coregistration_files/'
mindist = 5

for i in range (subjects_list.shape[0]):
    subject_id= subjects_list.loc[i,'Id']
    print("processing subject:", subject_id)
    subject_dir = os.path.join(subjects_dir, subject_id)
    
    fname_raw = os.path.join(subject_dir, 'raw_sss_ica_cleaned.fif')
    
    fname_fwd_folder = os.path.join(output_dir, subject_id)
    os.makedirs(fname_fwd_folder, exist_ok=True)
    fname_fwd = os.path.join(fname_fwd_folder, 'meg-fwd.fif')
    
    fname_trans = os.path.join(coreg_files, subject_id, 'coreg_trans.fif')
    #1 - layer model, only MEG
    fname_bem = os.path.join(freesurfer_subjects_dir, subject_id, 'bem', f'{subject_id}-5120-bem-sol.fif')
    
    fname_src = os.path.join(freesurfer_subjects_dir, subject_id, 'bem', f'{subject_id}-ico5-src.fif')
    
  #  src = mne.setup_source_space(subject, spacing="oct4", add_dist="patch", subjects_dir=subjects_dir)  this step was done when creating bem surfaces (see script at then end of this file)
    
 #   info = mne.io.read_info(fname_ave)

    fwd = mne.make_forward_solution(fname_raw, fname_trans, fname_src, fname_bem, meg=True, eeg=False, mindist=mindist)
    
    mne.write_forward_solution(fname_fwd, fwd, overwrite=True)

