#!/bin/bash
#SBATCH -J RF_model  	       
#SBATCH -o output_%A_%a.out    
#SBATCH -e error_%A_%a.err      

## Use this .sh shell script to train random forest models in parallel on the JASMIN Lotus cluster
# See https://help.jasmin.ac.uk/category/4889-slurm for more details about batch computing on JASMIN
# Code examples with command-line arguments for each level are included below for different input bands and reference datasets

# MUST activate conda environment first...
conda activate mres-fire

python_scripts=(

# Level 1: 13 spectral bands, sampling 200,000 pixels per class, cloud masked data
"L1_train.py --satellite_files SCL_Winter_10bands_indices.tif SCL_Spring_10bands_3indices.tif SCL_Summer_10bands_3indices.tif SCL_Autumn_10bands_3indices.tif --groundtruth_file FINAL_L1_SPECTRAL_Coimbra.tif --L2_groundtruth_file FINAL_L2_SPECTRAL_Coimbra.tif --L3_groundtruth_file FINAL_L3_SPECTRAL_Coimbra.tif --label_type L1 --model_output RUN2_spectral_L1_10bands_SCL.pkl --model_predictions RUN2_spectral_L1_10bands_SCL.csv --pixels_per_class 200000"

# Level 2: 10 spectral blands, data with low cloud cover
"L2L3_train.py --satellite_files NoCloud_Winter_10bands.tif NoCloud_Spring_10bands.tif NoCloud_Summer_10bands.tif --groundtruth_file FINAL_L2_CORINE_Coimbra.tif --label_type L2 --model_output RUN1_L2_10bands_NoCloud.pkl --model_predictions RUN1_L2_10bands_NoCloud.csv"

# Level 3: 10 spectral bands, cloud masked data, level 3 classification with CORINE nomenclature 
"L2L3_train.py --satellite_files SCL_Winter_10bands.tif SCL_Spring_10bands.tif SCL_Summer_10bands.tif SCL_Autumn_10bands.tif --groundtruth_file FINALISED_L3_CORINE_Coimbra.tif --label_type L3 --model_output RUN1_L3_10bands_SCL_reclassified.pkl --model_predictions RUN1_L3_10bands_SCL_reclassified.csv"

# Hierarchical class 1:
"H_train.py --input_class 1 --model_output L2H_CORINE_CLASS1.pkl --input_predictions RUN3_CORINE_L1_10bands_SCL.csv --output_predictions L2H_CORINE_CLASS1.csv"

)

# Iterate and submit each script as separate job
# Set up like this to allow multiple python commands in one batch submission, 
# but submit them as separate jobs so they get the resources they need
# rather than submitting them alltogether as one job...
for script in "${python_scripts[@]}"
do
    # Generate job ID
    job_id=$(date +%s%N)

    # Create job script
    job_script="job_$job_id.sh"
    echo '#!/bin/bash' > "$job_script"
    echo "#SBATCH -J RF_job_$job_id" >> "$job_script"
    echo "#SBATCH -o output_%j_$job_id.out" >> "$job_script"
    echo "#SBATCH -e error_%j_$job_id.err" >> "$job_script"
    echo "#SBATCH --partition=high-mem" >> "$job_script" # <-- see https://help.jasmin.ac.uk/article/4881-lotus-queues for other queue options
    #echo "#SBATCH --ntasks=16" >> "$job_script" <-- can experiment with this parameter
    #echo "#SBATCH --ntasks=24" >> "$job_script" <-- can experiment with this parameter
    echo "#SBATCH --time=24:00:00" >> "$job_script"
    echo "#SBATCH --mem=256000" >> "$job_script" # <-- high-mem partition requires a memory specification!!
    echo "conda activate mres-fire" >> "$job_script"
    echo "python $script" >> "$job_script"

    # Submit job
    sbatch "$job_script"

    rm "$job_script"
done