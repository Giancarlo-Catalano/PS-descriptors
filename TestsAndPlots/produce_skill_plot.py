
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



csv_file_location = r"/ExplanatoryCachedData/BT/Final/skill_table.csv"


df = pd.read_csv(csv_file_location)


def make_plot_for_property(property_name: str):

    # Create a box plot for each skill
    fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(15, 25))
    axes = axes.flatten()

    for i in range(18):
        skill = f'SKILL_{i}'
        ax = axes[i]
        df.boxplot(column=property_name, by=skill, ax=ax)
        ax.set_title(f'{skill}')
        ax.set_xlabel('Skill Value')
        ax.set_ylabel(property_name)

    # Adjust layout
    plt.suptitle(f'{property_name} by Skill')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    #plt.show()

    plt.savefig(fr'C:\Users\gac8\PycharmProjects\PS-PDF\Experimentation\BT\Final\{property_name}.pdf')


def make_violin_plot_for_property(property_name: str):

    # Create a box plot for each skill
    fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(15, 25))
    axes = axes.flatten()

    for i in range(18):
        skill = f'SKILL_{i}'
        ax = axes[i]
        sns.violinplot(x=skill, y=property_name, data=df, ax=ax)
        ax.set_title(f'{skill}')
        ax.set_xlabel('Skill Value')
        ax.set_ylabel(property_name)

    # Adjust layout
    plt.suptitle(f'{property_name} by Skill')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    #plt.show()

    plt.savefig(fr'C:\Users\gac8\PycharmProjects\PS-PDF\Experimentation\BT\Final\{property_name}_violin.pdf')


#make_plot_for_property("saturday_coverage")
#make_plot_for_property("sunday_coverage")

make_violin_plot_for_property("saturday_coverage")
make_violin_plot_for_property("sunday_coverage")
