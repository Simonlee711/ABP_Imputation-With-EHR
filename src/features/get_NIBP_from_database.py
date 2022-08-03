import sys
import os
import pandas as pd
import glob
from sqlalchemy import create_engine
from tqdm import tqdm

sys.path.append("../../")
import src.project_configs as project_configs

db_string = "postgres://mimicuser:mimic@localhost:5432/mimic"
db = create_engine(db_string)

# this query gets NIBP values for a given patient, along with time value was taken according to chart
query_str = """
select
ce.charttime, 
ce.valuenum, 
ce.itemid
from mimiciii.chartevents ce
where ce.subject_id={} 
and ce.itemid in (455, 220179, 8441, 220180, 456, 220181) -- NIBP Sys, NIBP Sys, NIBP Dias, NIBP Dias, NIBP Mean, NIBP Mean
order by ce.subject_id, ce.charttime;
"""

# need to map between integer coding and a human-readable string
bp_code_string_mapping = {
    455: "NIBP_sys",
    220179: "NIBP_sys",
    8441: "NIBP_dias",
    220180: "NIBP_dias",
    456: "NIBP_mean",
    220181: "NIBP_mean"
}
# make common code for each BP type
bp_code_int_mapping = {
    455: 220179,
    8441: 220180,
    456: 220181
}


def get_nibp_from_db(patient_id, save_dir, write_csv=True, overwrite=True):
    file_name = os.path.join(save_dir, "{}_NIBP.csv.gz".format(patient_id))
    if os.path.exists(file_name) and overwrite is False:
        print("{} exists and overwrite set to False. Skipping".format(file_name))
        return None
    query = query_str.format(patient_id)
    df = pd.read_sql_query(query, con=db)
    # remap BP int codes
    df["itemid"].replace(bp_code_int_mapping, inplace=True)
    # check to see if we got sys, dias, and mean BP values
    if len(df["itemid"].unique()) < 3:
        print("uh oh... didn't find all measurements for patient {}".format(patient_id))
        print(df["itemid"].unique())
        print("Skipping...")
        return None
    # long to wide
    # TODO: fix this issue when we possibly have duplicate measurements at same time
    try:
        df = df.pivot(index="charttime", columns="itemid", values="valuenum")
    except ValueError as e:
        print(e)
        print(df[df.duplicated(subset="charttime", keep=False)])
        return None
    assert (len(df.columns.values) == 3), "{} does not contain 3 columns after pivot (found {})".format(file_name,
                                                                                                        df.shape[1])
    # remap vital IDs from int to string
    df.rename(columns=bp_code_string_mapping, inplace=True)
    # rearrange column names because tradition
    df = df[["NIBP_sys", "NIBP_dias", "NIBP_mean"]]
    # write to file
    if write_csv:
        df.to_csv(file_name, sep=",", header=True, index=True)
    return df


def main():
    if not os.path.exists(project_configs.nibp_data_dir):
        os.makedirs(project_configs.nibp_data_dir)
    files = [s.strip() for s in open("../data/ECG_PLETH_ABP_IDs_wdb3_matched.txt").readlines()]
    patient_ids = [int(os.path.basename(s).split("-")[0][1:]) for s in files]

    for i in tqdm(patient_ids):
        get_nibp_from_db(i, save_dir=project_configs.nibp_data_dir, overwrite=False)


if __name__ == "__main__":
    main()
