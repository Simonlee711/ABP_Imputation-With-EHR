import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob

import seaborn as sns


figsize=(8, 8)
sys_axis_lim = [50, 200]
dias_axis_lim = [0, 150]
save_dir = "../../reports/figures"


def plot_correlation(df, x, y, axis_lim=[50, 200], title_string="Pearson product-moment r: {}"):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(axis_lim, axis_lim)
    p = sns.scatterplot(x=df[x], y=df[y], ax=ax)
    # set axis limits
    p.set_ylim(axis_lim)
    p.set_xlim(axis_lim)
    p.set_yticks(np.arange(axis_lim[0], axis_lim[1] + 1, 25))
    p.set_xticks(np.arange(axis_lim[0], axis_lim[1] + 1, 25))
    ax.tick_params(labelsize=13)
    # set axis labels
    axis_label_font_size = 14
    p.set_ylabel("Predicted Arterial Pressure [mmHg]", fontsize=axis_label_font_size)
    p.set_xlabel("Invasive Arterial Pressure [mmHg]", fontsize=axis_label_font_size)
    # add number of points to plot
    ax.legend(["N={}".format(len(df[x].values))], loc='upper left')
    # add title
    title_font_size = 16
    correlation_coeff = np.corrcoef(df[x], df[y])[0, 1]
    ax.set_title(title_string.format(np.round(correlation_coeff, 2)), fontsize=title_font_size)


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
        plot_correlation(df, x="sys_true", y="sys_pred",
                         axis_lim=sys_axis_lim,
                         title_string="Pearson product-moment - Systolic ABP - Validation r: {0:.2f}")
        plt.savefig(os.path.join(save_dir, save_file_name + "_correlation.png"))
        plt.close()

    for f in dias_bland_altman_files:
        print("Processing {}".format(f))
        save_file_name = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_csv(f)
        df.head()
        plot_correlation(df, x="dias_true", y="dias_pred",
                         axis_lim=dias_axis_lim,
                         title_string="Pearson product-moment - Diastolic ABP - Validation r: {0:.2f}")
        plt.savefig(os.path.join(save_dir, save_file_name + "_correlation.png"))
        plt.close()


if __name__ == "__main__":
    main()
