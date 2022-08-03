import os
import shutil
import glob
import sys
import pandas as pd
from tqdm import tqdm
import argparse

sys.path.append(os.path.join(os.environ["HOME"], "code/github/ABP_pred/"))
import src.project_configs as project_configs
import src.utils as utils

min_number_window_files = 10
log_file = "{}.log".format(os.path.splitext(sys.argv[0])[0])

def main():
    parser = argparse.ArgumentParser(description="Removes all .npy window files\
            if patients have less than {} total .npy window files".format(min_number_window_files))
    parser.add_argument('--rm', help="Remove the files from the file system\
            (can't undo!)", action="store_true", default=False)
    parser.add_argument('--dry', help="Dry-run to see how many files would be\
            removed (use --rm flag to actually remove)(default)", action="store_true", default=True)
    args = parser.parse_args()

    # get list of all .npy window files
    train_windows = glob.glob(os.path.join(project_configs.window_save_dir, "*.npy"))
    
    # create data frame with file paths
    train_windows_df = pd.DataFrame(train_windows, columns=["file_path"])

    # get directory of file
    train_windows_df["dir_name"] = train_windows_df["file_path"].apply(lambda x: os.path.dirname(x))
    
    # from this list, get all patient IDs, and count of for each patient 
    train_windows_df["patient_ID"] = train_windows_df["file_path"].apply(lambda x: utils.get_patient_from_file(x))

    # get only those patients who have a few number of total windows
    patients_to_redo = train_windows_df.groupby("patient_ID").filter(lambda x: len(x) < min_number_window_files) 

    # remove all files for these patients
    if args.rm:
        for f in tqdm(patients_to_redo["file_path"]):
            os.remove(f)
        print("Removed {} files from {} patients with less than {} .npy window\
                files".format(patients_to_redo.shape[0], 
                    patients_to_redo["patient_ID"].unique().shape[0], 
                    min_number_window_files))
        with open(log_file, "w") as lf:
            lf.write("Removed {} files from {} patients with less than {} .npy window\
                files".format(patients_to_redo.shape[0], 
                    patients_to_redo["patient_ID"].unique().shape[0], 
                    min_number_window_files))
    else:
        print("Would have removed {} files from {} patients with less than {}\
                .npy window files".format(patients_to_redo.shape[0], 
                    patients_to_redo["patient_ID"].unique().shape[0], 
                    min_number_window_files))


if __name__ == "__main__":
    main()
