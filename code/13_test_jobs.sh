#!/bin/bash
#SBATCH -J RF_model  	       
#SBATCH -o output_%A_%a.out    
#SBATCH -e error_%A_%a.err      

## Use this .sh shell script to TEST random forest models in parallel on the JASMIN Lotus cluster
# Use AFTER running train.sh models
# See https://help.jasmin.ac.uk/category/4889-slurm for more details about batch computing on JASMIN
# Code examples with command-line arguments for each level are included below for different input bands and reference datasets

# MUST activate conda environment first...
conda activate mres-fire

python_scripts=(

# Level 1: 10 spectral bands, cloud masked data, CORINE nomerclature
"L1_RF_test_model.py --test_satellite_files Braga_SCL_Winter_10bands.tif Braga_SCL_Spring_10bands.tif Braga_SCL_Summer_10bands.tif Braga_SCL_Autumn_10bands.tif --test_groundtruth_file FINAL_L1_CORINE_Braga.tif --test_L2_groundtruth_file Braga_FINAL_CORINE_L2.tif --test_L3_groundtruth_file Braga_FINAL_CORINE_L3.tif --input_model RUN3_CORINE_L1_10bands_SCL.pkl --model_predictions L1_10bands_CORINE_Braga.csv --output_tif_file L1_10bands_CORINE_Braga.tif"

# Level 2: 10 spectral blands, cloud masked data, spectral class nomenclature
"L2L3_test.py --test_satellite_files Braga_SCL_Winter_10bands.tif Braga_SCL_Spring_10bands.tif Braga_SCL_Summer_10bands.tif Braga_SCL_Autumn_10bands.tif --test_groundtruth_file FINAL_FINAL_SPECTRAL_L2_Braga.tif --label_type L2 --input_model RUN2_L2_10bands_SCL_CORINE.pkl --model_predictions Braga_L2_CORINE_predictions.csv --output_tif_file Braga_L2_CORINE_predictions.tif"

# Level 3: 13 spectral bands, cloud masked data, CORINE nomenclature 
"L2L3_test.py --test_satellite_files Braga_SCL_Winter_10bands_3indices.tif Braga_SCL_Spring_10bands_3indices.tif Braga_SCL_Summer_10bands_3indices.tif Braga_SCL_Autumn_10bands_3indices.tif --test_groundtruth_file FINAL_L3_CORINE_Braga.tif --label_type L3 --input_model RUN1_L3_10bands_SCL_reclassified.pkl --model_predictions Braga_L3_CORINE_predictions.csv --output_tif_file Braga_L3_CORINE_predictions.tif"
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