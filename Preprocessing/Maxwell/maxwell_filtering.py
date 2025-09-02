fine_cal_file = ("sss_cal_3159_factory_02122021.dat")
crosstalk_file = ("ct_sparse_triux2.fif")

subjects_list= pd.read_csv("allsubj.csv", header=0)
subjects_dir = "MEG_allsubj/Resting"
output_dir = "MEG_processed/Resting/1_maxwell"

for i in range (subjects_list.shape[0]):
    
    subject_id= subjects_list.loc[i,'Id']
    print (f" processing subject {subject_id}")
    subject_dir= os.path.join(subjects_dir, subject_id)
    raw_fname_in = os.path.join(subject_dir, "Resting2_raw.fif")
    raw = mne.io.read_raw_fif(raw_fname_in)
    output_subject_dir = os.path.join(output_dir, subject_id)
    os.makedirs(output_subject_dir, exist_ok=True)
    raw_sss_out = os.path.join (output_subject_dir, "sss_raw.fif")
    plots_dir = os.path.join(output_subject_dir, "maxwell_filter_plots")
    os.makedirs(plots_dir, exist_ok=True)
    destination = raw.info['dev_head_t']
    origin = destination['trans'][:3, 3]
    n_active = mne.chpi.get_active_chpi(raw)
    if n_active.mean() > 0:
        print('active HPI coil detected, running tSSS')
        chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw, t_step_min=0.01, t_window='auto', ext_order=1, tmin=0, tmax=None, verbose=True)
        info = raw.info
        chpi_locs = mne.chpi.compute_chpi_locs(info, chpi_amplitudes)
        head_pos = mne.chpi.compute_head_pos(info, chpi_locs)
        fig_head_pos_traces = mne.viz.plot_head_positions(
        head_pos, mode="traces", destination=raw.info["dev_head_t"], info=raw.info, show=False)
        head_pos_traces_path = os.path.join(plots_dir, "head_pos_traces.png")
        fig_head_pos_traces.savefig(head_pos_traces_path, dpi=300)
        plt.close(fig_head_pos_traces)

        # Plot head position field
        fig_head_pos_field = mne.viz.plot_head_positions(head_pos, mode="field", destination=raw.info["dev_head_t"], info=raw.info, show=False)
        head_pos_field_path = os.path.join(plots_dir, "head_pos_field.png")
        fig_head_pos_field.savefig(head_pos_field_path, dpi=300)
        plt.close(fig_head_pos_field)
        
        # Run maxwell filter with head_pos
        raw_sss = mne.preprocessing.maxwell_filter(raw, calibration=fine_cal_file, cross_talk=crosstalk_file, origin=origin, destination=destination, head_pos=head_pos)
    else:
        print('no active HPI coil detected, running SSS')
        # If no active cHPI channels
    
    
    raw_sss.save(raw_sss_out, overwrite=True)
