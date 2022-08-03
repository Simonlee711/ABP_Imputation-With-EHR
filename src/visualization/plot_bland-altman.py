import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob

import seaborn as sns
sys.path.append("../../")
import src.utils as utils

figsize=(8, 8)
sys_axis_lim = [50, 200]
dias_axis_lim = [0, 150]
save_dir = "../../reports/figures"


def main():
    parser = argparse.ArgumentParser(description="Script for creating true vs predicted BP correlation plots")
    parser.add_argument('--input', help='Directory containing input files')
    parser.add_argument('--output', help='Directory to save output images', default=save_dir)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print("ERROR: {} does not exist!".format(args.input))
        exit(-1)
    if not os.path.exists(args.output):
        print("{} does not exist. Creating...".format(args.output))
        os.makedirs(args.output)

    sys_bland_altman_files = glob.glob(os.path.join(args.input, "*sys_bland-altman.csv"))
    dias_bland_altman_files = glob.glob(os.path.join(args.input, "*dias_bland-altman.csv"))

    for f in sys_bland_altman_files:
        print("Processing {}".format(f))
        save_file_name = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_csv(f)
        df.head()
        utils.bland_altman_plot_v2(df["sys_true"], df["sys_pred"],
                                   axis_lim=sys_axis_lim, figsize=figsize,
                                   title_string="Bland-Altman - Systolic ABP - Validation: {} +/- {}")
        plt.savefig(os.path.join(save_dir, save_file_name + "_bland-altman_v2.png"))
        plt.close()

    for f in dias_bland_altman_files:
        print("Processing {}".format(f))
        save_file_name = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_csv(f)
        df.head()
        utils.bland_altman_plot_v2(df["dias_true"], df["dias_pred"],
                                   axis_lim=dias_axis_lim, figsize=figsize,
                                   title_string="Bland-Altman - Diastolic ABP - Validation: {} +/- {}")
        plt.savefig(os.path.join(save_dir, save_file_name + "_bland-altman_v2.png"))
        plt.close()


if __name__ == "__main__":
    main()
