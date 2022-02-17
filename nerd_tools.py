import random

import pandas as pd
import seaborn as sns
import statsmodels.stats.api as sms

from matplotlib import pyplot as plt

def load_qualtrics(path, drop_incomplete=True):
    """
    Loads data exported from Qualtrics as csv
    """
    df = pd.read_csv(path)
    df.drop([0,1], inplace=True)
    if drop_incomplete == True:
        print(f"Dropping {len(df[df['Finished'] == '0'])} unfinished responses...")
        df.query("Finished == '1'", inplace=True)
    return df

def aspect_ratio_locker(aspect_ratio, multiplier):
    """
    Creates an easy tool to manipulate figure size inside a fixed aspect ratio
    """
    return([i * multiplier for i in aspect_ratio])

def confint_error(x):
    return sms.DescrStatsW(x).tconfint_mean()[1] - x.mean()

def fancy_hboxplot(
    x,
    y,
    data,
    likert_limits=[],
    midpoint_line='',
    add_mean=True,
    title="",
    y_label="",
    x_label="",
    x_jitter_param=0.1,
    y_jitter_param=0.08,
    label_rotation=90,
    size=0.4,
    alpha=0.2,
    aspect_ratio = [16, 9],
    output_path="",
    sort_values = True,
    **kwargs
):
    """
    Creates an horizontal boxplot (cat on X axis) with a jittered overlay for each datapoint.
    Options for enforcing likert scale limits and for adding a diamond representing the mean.
    """

    plt.figure(figsize=aspect_ratio_locker(aspect_ratio, size), dpi = 600)

    # defines the order
    order = data.groupby(x)[y].mean().reset_index()
    order.columns = [x, "agg" + y]

    data = pd.merge(data, order, on=x)

    if sort_values == True:
        data.sort_values("agg" + y, ascending=False, inplace=True)

    # draws the basic plot
    p = sns.boxplot(
        x=x,
        y=y,
        data=data,
        color="skyblue",
        fliersize=0,
        **kwargs
        )

    p.spines["top"].set_visible(False)
    p.spines["right"].set_visible(False)

    p.set_xlabel(x_label)
    p.set_ylabel(y_label)
    p.set_title(title)

    for tick in p.get_xticklabels():
        tick.set_rotation(label_rotation)

    for i, row in data.iterrows():
        x_jitter = random.gauss(0,x_jitter_param)
        y_jitter = random.gauss(0,y_jitter_param)

        y_coord = row[y] + y_jitter

        if likert_limits != []:
            if y_coord < likert_limits[0]:
                y_coord = likert_limits[0]
            if y_coord > likert_limits[1]:
                y_coord = likert_limits[1]

        for index, x_value in enumerate(data[x].unique()):
            if row[x] == x_value:
                p.plot(index + x_jitter, y_coord, 'ro', color='navy', alpha=alpha)
                if add_mean == True:
                    p.plot(index, data[data[x] == x_value][y].mean(), "D", color="darkblue", markersize=9)

    if midpoint_line != "":
        p.axhline(midpoint_line, linestyle='--', color='navy', alpha=.5)

    if output_path != "":
        p.get_figure().savefig(output_path, bbox_inches="tight", dpi=600)

    return p

def fancy_vboxplot(
    x,
    y,
    data,
    likert_limits=[],
    midpoint_line='',
    add_mean=True,
    title="",
    y_label="",
    x_label="",
    x_jitter_param=0.08,
    y_jitter_param=0.1,
    label_rotation=0,
    size=0.4,
    alpha=0.2,
    aspect_ratio=[16, 9],
    output_path="",
    sort_values = True,
    **kwargs
):
    """
    Creates a vertical boxplot (cat on Y axis) with a jittered overlay for each datapoint.
    Options for enforcing likert scale limits and for adding a diamond representing the mean.
    """

    plt.figure(figsize=aspect_ratio_locker(aspect_ratio, size), dpi = 600)

    # defines the order
    order = data.groupby(y)[x].mean().reset_index()
    order.columns = [y, "agg" + x]

    data = pd.merge(data, order, on=y)

    if sort_values == True:
        data.sort_values("agg" + x, ascending=False, inplace=True)

    # draws the basic plot
    p = sns.boxplot(
        x=x,
        y=y,
        data=data,
        color="skyblue",
        fliersize=0,
        **kwargs
        )

    p.spines["top"].set_visible(False)
    p.spines["right"].set_visible(False)

    p.set_xlabel(x_label)
    p.set_ylabel(y_label)
    p.set_title(title)

    for tick in p.get_yticklabels():
        tick.set_rotation(label_rotation)

    for i, row in data.iterrows():
        x_jitter = random.gauss(0,x_jitter_param)
        y_jitter = random.gauss(0,y_jitter_param)

        x_coord = row[x] + x_jitter

        if likert_limits != []:
            if x_coord < likert_limits[0]:
                x_coord = likert_limits[0]
            if x_coord > likert_limits[1]:
                x_coord = likert_limits[1]

        for index, y_value in enumerate(data[y].unique()):
            if row[y] == y_value:
                p.plot(x_coord, index + y_jitter, 'ro', color='navy', alpha=alpha)
                if add_mean == True:                
                    p.plot(data[data[y] == y_value][x].mean(), index, "D", color="darkblue", markersize=9)

    if midpoint_line != "":
        p.axvline(midpoint_line, linestyle='--', color='navy', alpha=.5)

    if output_path != "":
        p.get_figure().savefig(output_path, bbox_inches="tight", dpi=600)

    return p


def vtiefighterplot(
    x,
    y,
    data,
    midpoint_line='',
    title="",
    y_label="",
    x_label="",
    size = 0.4,
    aspect_ratio = [16, 9],
    output_path="",
    margin=.5,
    new_line="",
    sort_values=True,
    add_n=True,
    **kwargs
):
    """
    Creates a vertical tiefighter plot (cat on Y axis) with a 95% CI as error bars.
    """

    plt.figure(figsize=aspect_ratio_locker(aspect_ratio, size), dpi = 600)

    new_data = data.groupby(y)[x].agg(['mean', 'size', confint_error]).reset_index()

    if add_n == True:
        new_data["condition_name"] = new_data.apply(lambda x: x[y] + new_line + "(n = " + str(x["size"]) + ")", axis=1)
    else:
        new_data["condition_name"] = new_data[y]

    if sort_values == True:
        new_data.sort_values("mean", ascending=False, inplace=True)

    p = sns.pointplot(
        x="mean",
        y="condition_name",
        data=new_data,
        ci=None,
        linestyles="",
        scale=0.5,
        **kwargs
        )

    p.set_title(title)
    p.set_ylabel(y_label)
    p.set_xlabel(x_label)

    p.spines["top"].set_visible(False)
    p.spines["right"].set_visible(False)

    p.margins(y=margin)

    if midpoint_line != '':
        p.axvline(midpoint_line, linestyle='--', alpha=.7)

    plt.errorbar(new_data['mean'], new_data['condition_name'], xerr=new_data['confint_error'], fmt='.', capsize=7)

    if output_path != "":
        p.get_figure().savefig(output_path, bbox_inches="tight", dpi=600)

    return p

def htiefighterplot(
    x,
    y,
    data,
    midpoint_line='',
    title="",
    y_label="",
    x_label="",
    size = 0.4,
    aspect_ratio = [16, 9],
    output_path="",
    margin=.5,
    new_line="",
    sort_values=True,
    add_n=True,
    **kwargs
):
    """
    Creates a vertical tiefighter plot (cat on Y axis) with a 95% CI as error bars.
    """

    plt.figure(figsize=aspect_ratio_locker(aspect_ratio, size), dpi = 600)

    new_data = data.groupby(x)[y].agg(['mean', 'size', confint_error]).reset_index()

    if add_n == True:
        new_data["condition_name"] = new_data.apply(lambda x: x[y] + new_line + "(n = " + str(x["size"]) + ")", axis=1)
    else:
        new_data["condition_name"] = new_data[y]

    if sort_values == True:
        new_data.sort_values("mean", ascending=False, inplace=True)

    p = sns.pointplot(
        x="condition_name",
        y="mean",
        data=new_data,
        ci=None,
        linestyles="",
        scale=0.5,
        **kwargs
        )

    p.set_title(title)
    p.set_ylabel(y_label)
    p.set_xlabel(x_label)

    p.spines["top"].set_visible(False)
    p.spines["right"].set_visible(False)

    p.margins(x=margin)

    if midpoint_line != '':
        p.axhline(midpoint_line, linestyle='--', alpha=.7)

    plt.errorbar(new_data['condition_name'], new_data['mean'], yerr=new_data['confint_error'], fmt='.', capsize=7)

    if output_path != "":
        p.get_figure().savefig(output_path, bbox_inches="tight", dpi=600)

    return p

def histogram(series, title = "", x_label = "", y_label = "", kde=False, bins=7, size = 0.4, aspect_ratio = [16, 9], output_path="", **kwargs):
    """
    Plots an aesthetically pleasing histogram
    """

    plt.figure(figsize=aspect_ratio_locker(aspect_ratio, size), dpi = 600)

    p = sns.histplot(series, kde=kde, bins=bins, **kwargs)

    p.set_title(title)
    p.set_xlabel(x_label)
    p.set_ylabel(y_label)

    p.spines["top"].set_visible(False)
    p.spines["right"].set_visible(False)

    if output_path != "":
        p.get_figure().savefig(output_path, bbox_inches="tight", dpi=600)

    return p

def hist_ridge(
    columns,
    rows,
    filtering_variable,
    data,
    xtick_labels=[],
    row_labels=True,
    row_label_rotation=90,
    aspect_ratio=[16, 32],
    size=0.4,
    output_path="",
    **kwargs
):
    """
    Creates a ridge plot of histograms with len(columns) columns and len(rows) rows.
    Every item in columns must be a level of a factor picked up by filtering_variable.
    """

    if len(xtick_labels) > 0 and type(xtick_labels[0]) == list:
        fig, axes = plt.subplots(nrows=len(rows), ncols=len(columns), figsize=nt.aspect_ratio_locker(aspect_ratio, size))
    else:
        fig, axes = plt.subplots(nrows=len(rows), ncols=len(columns), sharex="col", figsize=nt.aspect_ratio_locker(aspect_ratio, size))
    
    for i_c, c in enumerate(columns):
        for i_v, v in enumerate(rows):
            q = sns.histplot(
                data[data[filtering_variable] == c][v],
                ax=axes[i_v,i_c],
                kde=False,
                **kwargs
            )
            
            q.set_title('')
            q.set_xlabel('')
            q.set_ylabel('')
            q.spines["top"].set_visible(False)
            q.spines["right"].set_visible(False)
    
    #gets the labels right for the x axis if supplied
    if len(xtick_labels) > 0:
        #checks whether xtick_labels is a list of lists, or if it's a list of labels
        if type(xtick_labels[0]) != list:
            for i_c, _ in enumerate(columns):
                axes[i_v, i_c].set_xticks([i for i in range(1, len(xtick_labels)+1)])
                axes[i_v, i_c].set_xticklabels(xtick_labels)
        else: #in that case, the function expects 1 list per row.
            for i_c, c in enumerate(columns):
                for i_v, v in enumerate(rows):
                    axes[i_v, i_c].set_xticks([i for i in range(1, len(xtick_labels[i_v])+1)])
                    axes[i_v, i_c].set_xticklabels(xtick_labels[i_v])
                    
    for ax, col in zip(axes[0], columns):
        ax.set_title(col)

    if row_labels == True:
        for ax, row in zip(axes[:,0], rows):
            ax.set_ylabel(row, rotation=row_label_rotation, size='medium')
        
    fig.tight_layout()
        
    if output_path != "":
        fig.savefig(output_path, bbox_inches="tight", dpi=600)
        
    return fig