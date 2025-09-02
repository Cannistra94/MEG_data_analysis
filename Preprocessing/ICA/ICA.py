#ICA decomposition and signal correction
subjects_list= pd.read_csv("allsubj.csv", header=0)
subjects_dir = "MEG_processed/Resting/1_maxwell"
output_dir="MEG_processed/Resting/2_ica_decomposition"

for i in range (subjects_list.shape[0]):
    
    subject_id= subjects_list.loc[i,'Id']
    print (f" processing subject {subject_id}")
    subject_dir= os.path.join(subjects_dir, subject_id)
    subject_dir = os.path.join(subjects_dir, subject_id)
    output_subject_dir = os.path.join(output_dir, subject_id)
    os.makedirs(output_subject_dir, exist_ok=True)
    ica_out = os.path.join (output_subject_dir, "raw_sss_ica.fif")
    plots_dir = os.path.join(output_subject_dir, "ICA_decomposition_plots")
    os.makedirs(plots_dir, exist_ok=True)
    # Reading the raw file
    raw_fname_in = os.path.join(subject_dir, "sss_filtering_raw.fif")
    raw = mne.io.read_raw_fif(raw_fname_in, verbose=False)
    
    n_components = 0.999
    random_state=42
    ica = ICA(method='fastica', max_iter='auto',random_state=random_state,n_components=n_components)
    #picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,stim=False, exclude='bads')
    ica.fit(raw,  reject=dict(grad=4000e-13, mag=4e-12))
    raw.load_data()
    explained_var_ratio = ica.get_explained_variance_ratio(raw)
    for channel_type, ratio in explained_var_ratio.items():
        print(f"Fraction of {channel_type} variance explained by all components: {ratio}")
        ica.save(ica_out, overwrite=True)
    overlay_plot_path_1 = os.path.join(plots_dir, "ica_overlay_heartbeats.png")
    overlay_plot_path_2 = os.path.join(plots_dir, "ica_overlay_blinks.png")
    overlay_plot_path_3 = os.path.join(plots_dir, "ica_QC.png")
    raw.load_data()
    ica.plot_overlay(raw, exclude=[0], picks="mag", show=False) #blinks
    plt.savefig(overlay_plot_path_2)
    plt.close()
    
    ica.plot_overlay(raw, exclude=[1], picks="mag", show=False) #heartbeats
    plt.savefig(overlay_plot_path_1)
    plt.close()
    
    #ica.plot_properties(raw, picks=[0, 1]) #QC
    #plt.savefig(overlay_plot_path_3)
    #plt.close()
    
    ica.exclude = []
    raw.load_data()
    # find which ICs match the EOG pattern
    eog_indices, eog_scores = ica.find_bads_eog(raw)
    ica.exclude.extend (eog_indices)
    
    # find which ICs match the ECG pattern
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw, method="correlation", threshold="auto")
    ica.exclude.extend (ecg_indices)
    
    # Barplot of ICA component "EOG match" scores
    fig_scores = ica.plot_scores(eog_scores, show=False)
    fig_scores.savefig(os.path.join(plots_dir, "ica_eog_scores.png"), dpi=300)
    plt.close(fig_scores)
    
        
    # Plot scores (ECG match scores) and save
    fig_scores = ica.plot_scores(ecg_scores, show=False)
    fig_scores.savefig(os.path.join(plots_dir, "ica_scores_ecg.png"), dpi=300)
    plt.close(fig_scores)
    
        
    # Plot sources with ECG matches and save
    if ecg_indices:
        fig_sources = ica.plot_sources(raw, picks=ecg_indices, show=False)
        fig_sources.savefig(os.path.join(plots_dir, "ica_sources_ecg_comp.png"), dpi=300)
        plt.close(fig_sources)
    else:
        print(f"No ECG-related components found for subject {subject_id}. Skipping ECG source plot.")
           
    # Plot IC timecourses applied to raw data (sources)
    fig_sources_raw = ica.plot_sources(raw, show_scrollbars=False, show=False)
    fig_sources_raw.savefig(os.path.join(plots_dir, "ica_sources_raw.png"), dpi=300)
    plt.close(fig_sources_raw)
    
    # we found ecg and eog components to exclude after ICA fitting, now we can apply on the data
    
    reconst_raw = raw.copy()
    ica.apply(reconst_raw)
    
    reconst_raw.save(os.path.join(output_subject_dir, "raw_sss_ica_cleaned.fif"), overwrite=True)
