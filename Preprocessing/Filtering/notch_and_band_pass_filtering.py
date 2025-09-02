subjects_list= pd.read_csv("allsubj.csv", header=0)
subjects_dir = "MEG_processed/Resting/1_maxwell"

for i in range (subjects_list.shape[0]):
    
    subject_id= subjects_list.loc[i,'Id']
    print (f" processing subject {subject_id}")
    subject_dir= os.path.join(subjects_dir, subject_id)
    raw_fname_in = os.path.join(subject_dir, "sss_raw.fif")
    raw = mne.io.read_raw_fif(raw_fname_in, preload=True)
       
   
    # === 1. Notch filter to remove powerline noise at 60 Hz ===
    freqs = (60)
    raw_notch = raw.copy().notch_filter(freqs=freqs)
    for title, data in zip(["Un", "Notch "], [raw, raw_notch]):
        fig = data.compute_psd(fmax=300).plot(
            average=False, amplitude=False, picks="data", exclude="bads"
        )
        fig.suptitle(f"{title}filtered", size="xx-large", weight="bold")
        filepath = os.path.join(subject_dir, f"{title.strip().lower()}filtered_psd.png")
        fig.savefig(filepath, dpi=300)
        plt.close(fig)
        
    # === 2. Band-pass filter (e.g., 1–100 Hz) ===
    l_freq = 1.0
    h_freq = 100.0
  #  raw.load_data()
    picks_meg = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, stim=False, exclude=())
    raw_notch.filter(l_freq, h_freq, l_trans_bandwidth='auto', h_trans_bandwidth='auto',filter_length='auto', phase='zero', fir_window='hamming', fir_design='firwin', picks=picks_meg)
    
    # === 3. Save the preprocessed raw data ===
    output_fname = os.path.join(subject_dir, "sss_filtering_raw.fif")
    raw_notch.save(output_fname, overwrite=True)
