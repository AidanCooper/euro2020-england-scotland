import math
from datetime import datetime, time
from typing import Union

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import simps


def prep_data(df_raw: pd.DataFrame, match_date: str) -> pd.DataFrame:
    """
    Perform basic data preparation to support analysis of specific match.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw gym data DataFrame
    match_date : str
        Date of match, represented as a string: YYYY-MM-DD

    Returns
    -------
    pd.DataFrame
        Gym data, filtered for match date and previous six weeks of the same week day.
        Each week is numbered - 1-6 or baseline days and 7 for match day.
    """
    df = df_raw[df_raw["datetime"] - pd.to_timedelta(1, unit="d") <= match_date].copy()

    # retain only same days of week as match date
    dow = datetime.strptime(match_date, "%Y-%m-%d").weekday()
    df = df[df["datetime"].dt.dayofweek == dow]

    # number each week
    df["week"] = df["datetime"].dt.isocalendar().week
    df["week"] = df["week"] - df["week"].min() + 1
    if df["week"].max() == 8:
        df["week"] = df["week"] - 1
        df = df[df["week"] > 0]

    return df


def prep_data_stats(
    df: pd.DataFrame, kickoff: int, norm: bool = True, metric: str = "kick-off"
) -> pd.DataFrame:
    """
    Perform data preparation to support statistical analysis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame prepared for a specific match by `utils.prep_data`
    kickoff : int
        Time of match kick-off, represented as an int (e.g., 8pm = 20)
    norm : bool
        Optionally normalise data, such that both English and Scottish gyms have
        same mean occupancy (1.0) during baseline period.
    metric : str
        If "kick-off", calculate occupancy values at `kickoff` time. If "auc",
        calculate Area Under the gym occupancy Curve (AUC) over entire day.

    Returns
    -------
    pd.DataFrame
        Occupancy data at kick-off, in long format (required format for
        processing in `pingouin`).
    """

    def auc(group):
        x = group["time"]
        y = group["occupancy"]
        return simps(y, x)

    dfw = df.copy()
    dfw["time"] = dfw["datetime"].dt.time
    dfw.drop("datetime", axis=1, inplace=True)
    if metric == "kick-off":
        dfw = dfw[dfw["time"] == time(int(kickoff), 0, 0)].drop("time", axis=1)
        dfw = dfw.pivot(
            index=["country", "gym"], columns="week", values="occupancy"
        ).reset_index()
        dfw.columns = [
            "country",
            "gym",
            "w1",
            "w2",
            "w3",
            "w4",
            "w5",
            "w6",
            "w7",
        ]
        dfw["w1-6"] = dfw.iloc[:, 2:-1].mean(axis=1)
        if norm:
            eng_baseline = dfw.query("country == 'England'")["w1-6"].mean()
            sct_baseline = dfw.query("country == 'Scotland'")["w1-6"].mean()
            baseline_map = {"England": eng_baseline, "Scotland": sct_baseline}
            dfw["w1-6"] = dfw["w1-6"] / dfw["country"].map(baseline_map)
            dfw["w7"] = dfw["w7"] / dfw["country"].map(baseline_map)
        dfw = (
            dfw[["country", "gym", "w1-6", "w7"]]
            .melt(id_vars=["country", "gym"], value_vars=["w1-6", "w7"])
            .rename(columns={"variable": "week", "value": "occupancy"})
        )
    elif metric == "auc":
        dfw = dfw.pivot(
            index=["country", "gym", "time"], columns="week", values="occupancy"
        ).reset_index()
        dfw.columns = [
            "country",
            "gym",
            "time",
            "w1",
            "w2",
            "w3",
            "w4",
            "w5",
            "w6",
            "w7",
        ]
        dfw["w1-6"] = dfw.iloc[:, 3:-1].mean(axis=1)
        if norm:
            eng_baseline = dfw.query("country == 'England'")["w1-6"].mean()
            sct_baseline = dfw.query("country == 'Scotland'")["w1-6"].mean()
            baseline_map = {"England": eng_baseline, "Scotland": sct_baseline}
            dfw["w1-6"] = dfw["w1-6"] / dfw["country"].map(baseline_map)
            dfw["w7"] = dfw["w7"] / dfw["country"].map(baseline_map)
        dfw = (
            dfw[["country", "gym", "time", "w1-6", "w7"]]
            .melt(id_vars=["country", "gym", "time"], value_vars=["w1-6", "w7"])
            .rename(columns={"variable": "week", "value": "occupancy"})
        )
        dfw["time"] = dfw["time"].apply(lambda x: x.hour + x.minute / 60)
        dfw = (
            dfw.groupby(["country", "gym", "week"])
            .apply(auc)
            .reset_index()
            .rename(columns={0: "auc"})
        ).sort_values(["week", "gym"])
    else:
        print("`metric` must be 'kick-off' or 'auc'.")
        return

    return dfw


def plot_match(
    df,
    match_date: str,
    country: str,
    kickoff: int,
    c: str = "red",
    ax=None,
    ylim: float = None,
    gym_text: bool = True,
) -> plt.figure:
    """
    Plot match day gym occupancy vs previous six weeks.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame prepared for a specific match by `utils.prep_data`
    match_date : str
        Date of match, represented as a string: YYYY-MM-DD
    country : str
        Country being plotted - "England" or "Scotland".
    kickoff : int
        Time of match kick-off, represented as an int (e.g., 8pm = 20)
    c : str
        Colour to use for plotted occupancy data.
    ax
        Optionally specify a matplotlib axis to plot to; otherwise, one is created.
    ylim : float
        Optionally specify upper limit for y-axis; otherwise, automatically fit y
        axis to plotte data.
    gym_text : bool
        Whether or not to annotate flag with country text.

    Returns
    -------
    plt.figure
        Plotted gym occupancy data.
    """
    if ax:
        fig = None
    else:
        fig, ax = plt.subplots(figsize=(12, 6))

    dow_map = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }
    dow = dow_map[datetime.strptime(match_date, "%Y-%m-%d").weekday()]

    # six weeks of control
    df6 = (
        df.query("week <= 6 & country == @country")
        .groupby("datetime")
        .mean()
        .reset_index()
        .drop("week", axis=1)
    )
    df6["time"] = pd.Series([val.time() for val in df6["datetime"]])
    df6 = df6.groupby("time")["occupancy"].agg(["mean", "std", "sem"]).reset_index()
    df6.time = df6.time.apply(lambda x: x.hour + x.minute / 60)
    df6["lower"] = df6["mean"] - stats.norm.ppf(0.975) * df6["sem"]
    df6["upper"] = df6["mean"] + stats.norm.ppf(0.975) * df6["sem"]

    # game day
    df7 = (
        df.query("week == 7 & country == @country")
        .groupby("datetime")
        .mean()
        .reset_index()
        .drop("week", axis=1)
    )
    df7["time"] = df7["datetime"].apply(lambda x: x.hour + x.minute / 60)

    # plot
    ax.fill_between(
        df6["time"],
        df6["lower"],
        df6["upper"],
        color=c,
        alpha=0.1,
        label="95% confidence interval",
    )
    ax.plot(
        df7["time"],
        df7["occupancy"],
        c=c,
        linestyle="--",
        label=f"Occupancy on {dow} {match_date}",
    )
    ax.plot(
        df6["time"],
        df6["mean"],
        c=c,
        alpha=0.4,
        label=f"Mean occupancy over 6 previous {dow}s",
    )
    if ylim:
        ylim = (-2, 90)
    else:
        ylim = ax.get_ylim()
    ax.fill_betweenx(
        ylim,
        [kickoff, kickoff],
        [kickoff + 1.75, kickoff + 1.75],
        color="#228B22",
        alpha=0.3,
        label=f"Match in progress ({match_date})",
    )
    ax.set_ylim(ylim)

    ax.legend(loc="lower left", frameon=True)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02}:00" for h in range(0, 24, 2)], rotation=15)
    ax.set_xlim(0, 24)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=False))
    ax.grid()
    ax.set_xlabel("Time of Day")

    if gym_text:
        ax.text(
            0.005,
            0.64,
            f"Gyms in {country}",
            fontsize=14,
            fontweight="bold",
            transform=ax.transAxes,
        )
    ax.spines[:].set_visible(True)

    image = plt.imread(f"flags/{country}.png")
    axin = ax.inset_axes([-0.055, 0.7, 0.28, 0.28])
    axin.imshow(image)
    axin.spines[:].set_visible(True)
    axin.tick_params(
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False,
    )

    plt.close()

    if fig:
        return fig
    else:
        return ax
