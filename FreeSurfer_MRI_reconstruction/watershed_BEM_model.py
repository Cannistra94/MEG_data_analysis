list=pd.read_csv("allsubj.csv", header=0)
subjects_dir = "processed_t1_freesurfer_output"
for i in range (list.shape[0]):
    subject = list.loc[i,'Id']
  
    # Create Watershed BEM model
    mne.bem.make_watershed_bem(subject=subject, subjects_dir=subjects_dir, overwrite=True)

    # Check if the BEM surfaces are created
    bem_dir = os.path.join(subjects_dir, subject, "bem")
    print("BEM directory content:", os.listdir(bem_dir))
    fig= plot_bem(subject=subject, subjects_dir=subjects_dir, show=False)
  
    # Define the path to save the BEM plot
    plot_path = os.path.join(subjects_dir, subject, "bem_plot.png")
  
    # Save the plot
    fig.savefig(plot_path)
    print(f"BEM plot saved to: {plot_path}")
