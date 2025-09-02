subjects_list=pd.read_csv("allsubj.csv", header=0)
meg_dir='MEG_processed/Resting/1_maxwell/'
freesurfer_subjects_dir = "processed_t1_freesurfer_output/"
output_dir="MEG_processed/PainPressure/coregistration_files/"

for i in range (subjects_list.shape[0]):
    subject_id= subjects_list.loc[i,'Id']
    print("processing subject:", subject_id)
    subject_dir = os.path.join(meg_dir, subject_id)
    output_subject_dir = os.path.join(output_dir, subject_id)
    os.makedirs(output_subject_dir, exist_ok=True)
    log_file = os.path.join(output_subject_dir, "coregistration_log.txt")
    coreg_plots_dir = os.path.join(output_subject_dir, "coregistration_plots")
    os.makedirs(coreg_plots_dir, exist_ok=True)
    
    raw_fname_in = os.path.join(subject_dir, "sss_raw.fif")
    info=read_info(raw_fname_in)
    fiducials = mne.coreg.get_mni_fiducials(subject=subject_id, subjects_dir=freesurfer_subjects_dir)
    coreg = Coregistration(info, subject_id, freesurfer_subjects_dir, fiducials=fiducials)
        
    coreg.fit_fiducials(verbose=True)
        
    coreg.fit_icp(n_iterations=6, nasion_weight=2.0, verbose=True)
       
    coreg.omit_head_shape_points(distance=5.0 / 1000)  # distance is in meters
    
    coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=True)
        
    dists = coreg.compute_dig_mri_distances() * 1e3  # in mm
    trans_out = os.path.join(output_subject_dir, "coreg_trans.fif")
    mne.write_trans(trans_out, coreg.trans, overwrite=True)
    with open(log_file, "w") as f:
        
        f.write(f"Distance between HSP and MRI (mean/min/max):\n")
        f.write(f"{np.mean(dists):.2f} mm / {np.min(dists):.2f} mm / {np.max(dists):.2f} mm\n")
