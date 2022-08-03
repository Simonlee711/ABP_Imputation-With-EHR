import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")
import src.project_configs as project_configs

from sqlalchemy import create_engine
engine = create_engine('postgresql://mimicuser:mimic@localhost:5432/mimic')

# files with list of .npy windows (just use ls on project_config.train_dir)
train_file_f = "train_files.txt"
val_file_f = "val_files.txt"
test_file_f = "test_files.txt"
# file to save demographic stats info
demographic_stats_file = "demographic_stats.txt"

query_str = """
select * 
from mimiciii.icustay_detail ic
left join mimiciii.heightweight hw
on ic.icustay_id=hw.icustay_id
where ic.subject_id in ({});
"""

query_str = """
select 
ic.subject_id,
case 
when AVG(ic.admission_age) >= 300 then 90.
else AVG(ic.admission_age) 
end as age,
AVG(hw.height_first) as height,
AVG(hw.weight_first) as weight,
AVG(hw.weight_first) / (AVG(hw.height_first)/100.*AVG(hw.height_first)/100.) as bmi,
---ic.ethnicity,
ic.gender as sex
from mimiciii.icustay_detail ic
left join mimiciii.heightweight hw
on ic.icustay_id=hw.icustay_id
where ic.subject_id in ({})
group by ic.subject_id, ic.gender;
"""


def get_patient_list(f):
    """
    Get list of patient IDs from file containing list of preprocessed numpy arrays
    :param f: path to file containing names of all train/test numpy arrays
    :return: list of patient IDs
    """
    with open(f) as pl:
        # read in list of patient files
        file_list = [p.strip() for p in pl.readlines()]
        print("Found {} files in {}".format(len(file_list), f))
        # split file names to get patient ID
        file_list = [p.split("-")[0] for p in file_list]
        # remove leading p from patient IDs
        file_list = [p[1:] if p[0] == 'p' else p for p in file_list]
        # convert patient IDs from str to int and back to str to trim leading zeros
        file_list = [str(int(p)) for p in file_list]
        # get list of unique patient IDs
        file_list = list(set(file_list))
        print("Found {} unique patient IDs in {}".format(len(file_list), f))
        return file_list


def remove_outliers(df):
    """
    Remove outliers from demographic stats data frame
    :param df: pandas data frame to filter
    :return: filtered pandas data frame
    """
    # set height outliers to None
    df.height.mask(df.height < 100, inplace=True)
    # set BMI outliers to None
    df.bmi.mask(df.bmi > 100, inplace=True)
    return df


def plot_demographics():
    fontsize = 16
    demo_files = ["train_demographics.csv", "val_demographics.csv", "test_demographics.csv"]
    colors = ["blue", "orange", "green"]
    legend_labels = ["Train", "Validation", "Test"]
    f = demo_files[0]
    df = pd.read_csv(f, index_col=0)
    # for each demographic feature
    for i in range(df.shape[1]):
        fig, ax = plt.subplots(len(demo_files), 1, figsize=(16, 16), sharex=True, sharey=True)
        # for each file
        for j in range(len(demo_files)):
            f = demo_files[j]
            df = pd.read_csv(f, index_col=0)
            df = remove_outliers(df)
            # try to convert string to int 
            try:
                df["sex"] = df["sex"].apply(lambda x: 0 if x.lower() == "m" else 1)
            # unless already a numeric value, skip
            except AttributeError:
                pass

            ax[j].hist(df.iloc[:, i], alpha=1.0, bins=20, density=True, color=colors[j], label=legend_labels[j])
            import matplotlib.ticker as plticker
            if df.columns.values[i] == "weight":
                tick_spacing = 10
            else:
                tick_spacing = 5
            loc = plticker.MultipleLocator(base=tick_spacing)  # this locator puts ticks at regular intervals
            ax[j].xaxis.set_major_locator(loc)
            ax[j].legend(loc="upper right", fontsize=project_configs.axis_label_font_size+4)
            ax[j].tick_params(labelsize=project_configs.axis_tick_label_size+4)
        plt.xlabel(df.columns.values[i].capitalize(), fontsize=project_configs.axis_label_font_size+4)
        plt.ylabel("Fraction of population", fontsize=project_configs.axis_label_font_size+4)
        plt.tight_layout()
        plt.savefig("../../reports/figures/{}_hist.png".format(df.columns.values[i]))
        plt.show()


def main():
    # dump list of patient files
    train_files = [os.path.basename(x) for x in glob.glob(os.path.join(project_configs.train_dir, "*.npy"))]
    with open(train_file_f, "w") as train_f:
        train_f.writelines("%s\n" % f for f in train_files)
    val_files = [os.path.basename(x) for x in glob.glob(os.path.join(project_configs.val_dir, "*.npy"))]
    with open(val_file_f, "w") as val_f:
        val_f.writelines("%s\n" % f for f in val_files)
    test_files = [os.path.basename(x) for x in glob.glob(os.path.join(project_configs.test_dir, "*.npy"))]
    with open(test_file_f, "w") as test_f:
        test_f.writelines("%s\n" % f for f in test_files)

    # get list of patient files for train/test sets
    train_patients = get_patient_list(train_file_f)
    val_patients = get_patient_list(val_file_f)
    test_patients = get_patient_list(test_file_f)

    train_query = query_str.format(",".join(train_patients))
    val_query = query_str.format(",".join(val_patients))
    test_query = query_str.format(",".join(test_patients))

    train_df = pd.read_sql_query(train_query, con=engine)
    train_df = remove_outliers(train_df)
    print(train_df.describe(include="all"))
    train_df["subject_id"] = train_df["subject_id"].apply(lambda x: "p%06d" % x)
    train_df.to_csv("train_demographics.csv", sep=",", header=True, index=False)

    val_df = pd.read_sql_query(val_query, con=engine)
    val_df = remove_outliers(val_df)
    print(val_df.describe(include="all"))
    val_df["subject_id"] = val_df["subject_id"].apply(lambda x: "p%06d" % x)
    val_df.to_csv("val_demographics.csv", sep=",", header=True, index=False)

    test_df = pd.read_sql_query(test_query, con=engine)
    test_df = remove_outliers(test_df)
    print(test_df.describe(include="all"))
    test_df["subject_id"] = test_df["subject_id"].apply(lambda x: "p%06d" % x)
    test_df.to_csv("test_demographics.csv", sep=",", header=True, index=False)

    with open(demographic_stats_file, 'w') as demo_f:
        demo_f.write("=" * 30 + "\n")
        demo_f.write("TRAIN DATA" + "\n")
        demo_f.write("=" * 30 + "\n")
        demo_f.write(str(train_df.describe(include="all")))
        demo_f.write("\n")

        demo_f.write("=" * 30 + "\n")
        demo_f.write("VAL DATA" + "\n")
        demo_f.write("=" * 30 + "\n")
        demo_f.write(str(val_df.describe(include="all")))
        demo_f.write("\n")

        demo_f.write("=" * 30 + "\n")
        demo_f.write("TEST DATA" + "\n")
        demo_f.write("=" * 30 + "\n")
        demo_f.write(str(test_df.describe(include="all")))
        demo_f.write("\n")


if __name__ == "__main__":
    #main()
    plot_demographics()
