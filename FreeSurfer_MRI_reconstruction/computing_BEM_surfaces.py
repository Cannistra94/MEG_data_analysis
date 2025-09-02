#Computing BEMs surfaces
# Set FreeSurfer environment variables
freesurfer_home = "/usr/local/freesurfer/stable7.4.1"
os.environ['FREESURFER_HOME'] = freesurfer_home
subprocess.call(f"source {freesurfer_home}/SetUpFreeSurfer.sh", shell=True)
# Add FreeSurfer's bin directory to the PATH
os.environ["PATH"] += ":/usr/local/freesurfer/stable7.4.1/bin"

subjects_list=pd.read_csv("allsubj.csv", header=0)
subjects_dir = "processed_t1_freesurfer_output"
spacing="ico5"
for i in range (1):
    subject_id= list.loc[i,'Id']
    subject_dir = os.path.join(subjects_dir, subject_id)
    bem_dir = os.path.join(subject_dir, "bem")
    # Step 1: Create BEM surfaces
    for n_layers in (1, 3):
        # Single-layer and three-layer models
        extra = '-'.join(['5120'] * n_layers)
        fname_bem_surfaces = os.path.join(bem_dir, f"{subject_id}-{extra}-bem.fif")
        if not os.path.isfile(fname_bem_surfaces):
            print(f"Setting up {n_layers}-layer BEM for {subject_id}")
            conductivity = (0.3, 0.006, 0.3)[:n_layers]  # Conductivity values for tissues
            try:
                bem_surfaces = mne.make_bem_model(
                subject=subject_id, ico=4, conductivity=conductivity,
                subjects_dir=subjects_dir
                )
                mne.write_bem_surfaces(fname_bem_surfaces, bem_surfaces)
            except RuntimeError as exp:
                print(f"FAILED to create {n_layers}-layer BEM for {subject_id}: {exp.args[0]}")
                continue

        # Step 2: Create BEM solutions
        fname_bem_solution = os.path.join(bem_dir, f"{subject_id}-{extra}-bem-sol.fif")
        if not os.path.isfile(fname_bem_solution):
            print(f"Computing {n_layers}-layer BEM solution for {subject_id}")
            bem_model = mne.read_bem_surfaces(fname_bem_surfaces)
            bem_solution = mne.make_bem_solution(bem_model)
            mne.write_bem_solution(fname_bem_solution, bem_solution)

    print(f"BEM setup complete for {subject_id}")

    # Step 3: Create the surface source space
    fname_src = os.path.join(bem_dir, f"{subject_id}-{spacing}-src.fif")
    if not os.path.isfile(fname_src):
        print("Setting up source space")
        src = mne.setup_source_space(subject_id, spacing, subjects_dir=subjects_dir)
        mne.write_source_spaces(fname_src, src)

    print("Processing completed for", subject_id)

#Create FSAVERAGE within subjects_folder
fsaverage_src_dir = op.join(freesurfer_home, 'subjects', 'fsaverage')
fsaverage_dst_dir = op.join(subjects_dir, 'fsaverage')

print('Copying fsaverage into subjects directory')  

# Only remove symlink or directory if it exists
if op.islink(fsaverage_dst_dir) or op.isdir(fsaverage_dst_dir):
    os.unlink(fsaverage_dst_dir)  # Remove symlink if it exists

shutil.copytree(fsaverage_src_dir, fsaverage_dst_dir)

# Create BEM directory for fsaverage if it does not exist
fsaverage_bem = op.join(fsaverage_dst_dir, 'bem')
if not op.isdir(fsaverage_bem):
    os.mkdir(fsaverage_bem)

# Set up and save source space for fsaverage if it does not exist
fsaverage_src = op.join(fsaverage_bem, 'fsaverage-5-src.fif')
if not op.isfile(fsaverage_src):
    print('Setting up source space for fsaverage')
    src = mne.setup_source_space('fsaverage', 'ico5', subjects_dir=subjects_dir)
    for s in src:
        assert np.array_equal(s['vertno'], np.arange(10242))
    mne.write_source_spaces(fsaverage_src, src)

# now fsaverage
fsaverage_src_dir = op.join(os.environ['FREESURFER_HOME'], 'subjects', 'fsaverage')
fsaverage_dst_dir = op.join(subjects_dir, 'fsaverage')

print('Copying fsaverage into subjects directory')  # to allow writting in folder
os.unlink(fsaverage_dst_dir)  # remove symlink
shutil.copytree(fsaverage_src_dir, fsaverage_dst_dir)

fsaverage_bem = op.join(fsaverage_dst_dir, 'bem')
if not op.isdir(fsaverage_bem):
    os.mkdir(fsaverage_bem)

fsaverage_src = op.join(fsaverage_bem, 'fsaverage-5-src.fif')
if not op.isfile(fsaverage_src):
    print('Setting up source space for fsaverage')
    src = mne.setup_source_space('fsaverage', 'ico5',
                                 subjects_dir=subjects_dir)
    for s in src:
        assert np.array_equal(s['vertno'], np.arange(10242))
    mne.write_source_spaces(fsaverage_src, src)
