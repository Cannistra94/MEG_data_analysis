# Set FreeSurfer environment
source /usr/local/freesurfer/stable7.4.1/SetUpFreeSurfer.sh  

subject_id=$1

# Define the T1-weighted MRI input file path
t1_fname="${subject_id}/anat.nii.gz"

# Define the output log file
log_file="${subject_id}/my-recon-all.txt"

# Define the output directory for FreeSurfer results
subjects_output_dir="processed_t1_freesurfer_output"

recon-all -all -s "$subject_id" -sd "$subjects_output_dir" -i "$t1_fname" &> "$log_file"
