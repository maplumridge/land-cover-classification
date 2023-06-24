#!/bin/bash
#SBATCH -J RF_model  	       
#SBATCH -o output_%A_%a.out    
#SBATCH -e error_%A_%a.err      

## Only need to run this script and add all my commands to this

conda activate mres-fire

python_scripts=(
# DONE (latest v is modified)--> "practice_RF_model.py --satellite_files Winter_4bands.tif Spring_4bands.tif Summer_4bands.tif Autumn_4bands.tif --groundtruth_file L1_CORINE_32629_10m_Cropped.tif --pixels_per_class 129007 --label_type L1 --model_output L1_4bands_model.pkl --model_predictions L1_4bands_predictions.csv"
# Modified pixels per class below... try again
"practice_RF_model.py --satellite_files Winter_10bands.tif Spring_10bands.tif Summer_10bands.tif Autumn_10bands.tif --groundtruth_file L1_CORINE_32629_10m_Cropped.tif --groundtruth_file L2_CORINE_32629_10m_Cropped.tif --pixels_per_class 21184 --label_type L1 --model_output L1_10bands_model.pkl --model_predictions L1_10bands_predictions.csv"
# DONE --> "practice_test_RF_model.py --test_satellite_files Braga_Winter_4bands.tif Braga_Spring_4bands.tif Braga_Summer_4bands.tif Braga_Autumn_4bands.tif --test_groundtruth_file Braga_CORINE_Cropped_L1.tif  --label_type L1 --input_model L1_4bands_model.pkl --model_predictions unseen_L1_4bands_predictions.csv"
# TO DO "practice_test_RF_model.py --test_satellite_files Braga_Winter_10bands.tif Braga_Spring_10bands.tif Braga_Summer_10bands.tif Braga_Autumn_10bands.tif --test_groundtruth_file Braga_CORINE_Cropped_L1.tif  --label_type L1 --input_model L1_10bands_model.pkl --model_predictions unseen_L1_10bands_predictions.csv"
# DONE (with dif command in excel) --> "practice_H_train_RF_model.py --satellite_files Winter_4bands.tif Spring_4bands.tif Summer_4bands.tif Autumn_4bands.tif --groundtruth_file L2_CORINE_32629_10m_Cropped.tif --label_type L2 --model_output L2H_4bands_best_model.pkl --input_predictions L1_4bands_predictions.csv --output_predictions L2H_4bands_predictions.csv"
)

# Iterate over array and submit each script as separate job
## Set up like this to allow multiple python commands in one batch submission, 
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
    echo "#SBATCH --partition=short-serial" >> "$job_script"
    #echo "#SBATCH --ntasks=16" >> "$job_script"
    #echo "#SBATCH --ntasks=24" >> "$job_script"
    echo "#SBATCH --time=24:00:00" >> "$job_script"
    #echo "#SBATCH --mem=256000" >> "$job_script"
    echo "conda activate mres-fire" >> "$job_script"
    echo "python $script" >> "$job_script"

    # Submit job
    sbatch "$job_script"

    rm "$job_script"
done