from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def is_logical_row(row) -> bool:
    """
    Logically evaluate a given row to see if it follows:
    "Low" <= "Open"/"Close" <= "High".

    @params:
      row: pd.DataFrame row

    @reutrns:
      boolean indicating if the row is "logical"
    """
    return (
        (row["Low"] <= row["Open"] <= row["High"])
        and (row["Low"] <= row["Close"] <= row["High"])
        and (row["Volume"] > 0)
        and (row["Open Interest"] > 0)
        and all(
            val >= 0 for val in [row["Low"], row["Open"], row["High"], row["Close"]]
        )
    )


def add_logical_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a flag indicating "logical" entries using the
    "is_logical_row" helper method.

    @params:
      df: pd.DataFrame

    @returns:
      pd.DataFrame with "is_logical" column
    """
    result = df.copy()
    result["is_logical"] = result.apply(is_logical_row, axis=1)
    return result


def summary_stats_and_hist(
    df: pd.DataFrame,
    cols: list = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Open Interest",
    ],
) -> None:
    """
    Prints out summary stats, nullity percentage, and histograms for
    specified columns in an input df.

    @params:
      df: pd.DataFrame
      cols: list of columns to print summary stats and histograms for
    """
    df_columns = set(df.columns)
    for col in cols:
        if col not in df_columns:
            raise ValueError(f"Column: {col} not found in dataframe")

        print(f"\nFor column: \033[1m{col}\033[0m")  # in-text ANSI for bold effect
        print("Summary Stats: ")
        print(df[col].describe())

        print("\n")
        print("Missingness: ")
        print(sum(pd.isna(df[col])) / (len(df[col]) + 0.001))  # avoiding DividedByZero

        print("\n")
        print("Histogram:")
        plt.hist(df[col])
        plt.show()
        print("\n")


def scatter_plot(
    df,
    cols: list = ["Open", "High", "Low", "Close", "Volume", "Open Interest"],
) -> None:
    """
    Generate seaborn scatterplots for specified columns within an input dataframe.

    @params:
      df: pd.DataFrame
      cols: list of columns to generate scatterplots for
    """
    df_copy = df.copy()
    for col in cols:
        sns.scatterplot(data=df_copy, x="Date", y=col)
        plt.xticks(rotation=45)  # rotate x-axis labels
        plt.tight_layout()
        plt.show()


def line_plot(
    df,
    cols: list = ["Open", "High", "Low", "Close", "Volume", "Open Interest"],
):
    """
    Generate seaborn lineplots for specified columns within an input dataframe.

    @params:
      df: pd.DataFrame
      cols: list of columns to generate lineplots for
    """
    for col in cols:
        plt.figure(figsize=(12, 5))
        sns.lineplot(data=tmp, x="Date", y=col)  # , marker='o')
        plt.xticks(rotation=45)
        plt.show()


def between_percentiles(
    df,
    start: float = 0.1,
    end: float = 0.9,
    cols: list = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Open Interest",
    ],
):
    """
    Generate a percentile map for specified columns within an input dataframe
    where key: value pairs indicate the start-end percentiles for each column.

    @params:
      df: pd.DataFrame
      start: percentile float
      end: percentile float
      cols: list of columns to generate percentile map for

    @returns:
      dict
    """
    percentile_map = {}
    for each in cols:
        percentiles = df[each].quantile([start, end]).values
        percentile_map[each] = percentiles
    return percentile_map


def strip_outliers(
    df: pd.DataFrame,
    percentile_map: dict,
    cols: list = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Open Interest",
    ],
) -> pd.DataFrame:
    """
    Returns a "sanitized" dataframe by stripping outliers based on the columns
    in the input dataframe.

    Using percentile_map as reference for values outside the start-end %tiles.

    Bitwise AND-ing ensures only non-outlier entries exist across.

    @params:
      df: pd.DataFrame
      percentile_map: dict
      cols: list of columns to identify and strip outliers

    @returns:
      pd.DataFrame
    """
    valid_idx = pd.Series(True, index=df.index)
    for col in cols:
        low, high = percentile_map[col]
        valid_idx &= df[col].between(low, high)

    return df[valid_idx].copy()  # deep copy for safety


def add_outlier_flag(
    df: pd.DataFrame,
    percentile_map: dict,
    cols: list = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Open Interest",
    ],
) -> pd.DataFrame:
    """
    Adds columns in the format "is_colname_outlier" indicating a boolean value.

    @params:
      df: pd.DataFrame
      percentile_map: dict
      cols: list of numeric columns to denote outliers

    @returns:
      pd.DataFrame with "is_colname_outlier" columns
    """
    result = df.copy()
    for col in cols:
        low, high = percentile_map[col]
        result[f"is_{col}_outlier"] = ~df[col].between(low, high)
    return result


def process_input_df(
    df: pd.DataFrame,
    symbol: str,
    cols: list = ["Open", "High", "Low", "Close", "Volume", "Open Interest"],
    start_percentile: float = 0.1,
    end_percentile: float = 0.9,
    just_flag: bool = False,
) -> pd.DataFrame:
    """
    Runs preprocessing to eliminate NaNs, identify, extract and return
    "logical" and "non-outlier" rows as a result.

    Note: if just_flag is True, only add flags indicating "logical" and
    "non-outlier" rows else return a new df with "illogical" and "outlier"
    rows stripped away.

    @params:
      df: pd.DataFrame
      symbol: str
      cols: list of numeric columns to process
      start_percentile: float
      end_percentile: float
      just_flag: bool (default: False)

    @returns:
      processed pd.DataFrame
    """
    fut_df = df[df["Symbol"] == symbol]
    fut_df = fut_df.dropna()
    # input 'Timestamp' follows Excel time that can be converted to YYYY-MM-DD
    fut_df["Date"] = pd.to_datetime(fut_df["Timestamp"], unit="D", origin="1899-12-30")

    # if just_flag, add required flags while RETAINING all of the original data
    # else return "processed" data with "logical" and "non-outlier" rows
    if just_flag:
        valid_fut_df = add_logical_flag(fut_df)
        percentile_map = between_percentiles(
            valid_fut_df, cols=cols, start=start_percentile, end=end_percentile
        )
        regular_valid_fut_df = add_outlier_flag(valid_fut_df, percentile_map)
    else:
        valid_fut_df = fut_df[fut_df.apply(is_logical_row, axis=1)]
        percentile_map = between_percentiles(
            valid_fut_df, cols=cols, start=start_percentile, end=end_percentile
        )
        regular_valid_fut_df = strip_outliers(valid_fut_df, percentile_map)

    return regular_valid_fut_df


def process_and_combined_input(
    df: pd.DataFrame,
    cols: list = ["Open", "High", "Low", "Close", "Volume", "Open Interest"],
    start_percentile: float = 0.1,
    end_percentile: float = 0.9,
    just_flag: bool = False,
) -> pd.DataFrame:
    """
    Runs preprocessing to eliminate NaNs, identify, extract and return
    "logical" and "non-outlier" rows as a result.

    @params:
      df: pd.DataFrame
      symbol: str
      cols: list of numeric columns to process
      start_percentile: float
      end_percentile: float
      just_flag: bool (default: False)

    @returns:
      processed pd.DataFrame
    """
    result = pd.DataFrame()
    for symbol in df["Symbol"].unique():
        processed_chunk = process_input_df(
            df=df, symbol=symbol, cols=cols, just_flag=True
        )
        print(f"Processed df for Symbol: {symbol}")
        result = pd.concat([result, processed_chunk])
    return result


def moving_average(df: pd.DataFrame, day_window: int = 20) -> pd.DataFrame:
    df["MA"] = df["Close"].rolling(window=day_window).mean()
    return df


def exponential_moving_average(df: pd.DataFrame, day_window: int = 20) -> pd.DataFrame:
    df["EMA"] = df["Close"].ewm(span=day_window, adjust=False).mean()
    return df


def time_period_ticker_statistics(
    df: pd.DataFrame,
    value_col: str,
    day_window: int = 10000,
) -> dict:
    """
    Filters df to only the last N calendar days (or start of the dataset),
    then computes cumulative return and volatility for the day_window.

    @params:
      df: pd.DataFrame
      value_col: str
      day_window: int

    @returns:
      dict with aforementioned stats for the day_window
    """
    df = df.sort_values(by="Date", ascending=True).reset_index()
    end_date = df["Date"].max()
    # if day_window is larger than the dates within the df, use earliest date found
    start_date = max(end_date - timedelta(days=day_window), df["Date"].iloc[0])

    # subset
    window = df[df["Date"] >= start_date].copy()

    # cumulative return & volatility
    start_price = window[value_col].iloc[0]
    end_price = window[value_col].iloc[-1]

    cumulative_return = 100 * (end_price - start_price) / start_price
    volatility = window[value_col].std()

    return {
        f'Rate of Return for "{value_col}"': cumulative_return,
        f'Volatility / St. Dev for "{value_col}"': volatility,
        "Start": start_date,
        "End": end_date,
    }


def plot_with_ma_ema(df, date_col, value_col, day_window: int) -> None:
    series = df[value_col]

    ma = series.rolling(day_window).mean()
    ema = series.ewm(span=day_window, adjust=False).mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df[date_col], series, label=value_col, linewidth=1.5)
    ax.plot(df[date_col], ma, label=f"MA({day_window})", linewidth=1.5)
    ax.plot(
        df[date_col],
        ema,
        label=f"EMA({day_window})",
        linewidth=1.5,
        linestyle="--",
    )

    ax.set_title(f"{value_col} with MA({day_window}) & EMA({day_window})")
    ax.set_xlabel(date_col)
    ax.set_ylabel(value_col)
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
    return fig, ax
