import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import timedelta
import matplotlib.pyplot as plt

def plot_price_with_actions(injection_dates, withdrawal_dates):
    plt.figure(figsize=(12,5))

    plt.plot(full_series.index, full_series["prices"], label="Gas Price Curve")

    inj_dates = pd.to_datetime(injection_dates)
    wdr_dates = pd.to_datetime(withdrawal_dates)

    plt.scatter(inj_dates, [estimate_price(d) for d in inj_dates],
                color="green", marker="^", s=100, label="Injection")

    plt.scatter(wdr_dates, [estimate_price(d) for d in wdr_dates],
                color="red", marker="v", s=100, label="Withdrawal")

    plt.title("Natural Gas Price Curve with Storage Actions")
    plt.xlabel("Date")
    plt.ylabel("Price ($/MMBtu)")
    plt.legend()
    plt.grid()
    plt.show()

def price_storage_contract_with_logs(
    injection_dates,
    withdrawal_dates,
    volume_per_action,
    max_storage_volume,
    injection_rate,
    withdrawal_rate,
    storage_cost_per_day,
    injection_cost,
    withdrawal_cost
):

    inventory = 0.0
    value = 0.0

    inventory_log = []
    cashflow_log = []

    events = []
    for d in injection_dates:
        events.append((pd.to_datetime(d), "inject"))
    for d in withdrawal_dates:
        events.append((pd.to_datetime(d), "withdraw"))

    events.sort(key=lambda x: x[0])
    last_date = events[0][0]

    for date, action in events:

        days = (date - last_date).days
        if days > 0:
            cost = days * storage_cost_per_day
            value -= cost

        price = estimate_price(date)

        if action == "inject":
            value -= volume_per_action * price
            value -= injection_cost
            inventory += volume_per_action

        else:
            value += volume_per_action * price
            value -= withdrawal_cost
            inventory -= volume_per_action

        inventory_log.append((date, inventory))
        cashflow_log.append((date, value))
        last_date = date

    return round(value,2), inventory_log, cashflow_log

def plot_inventory(inventory_log):
    dates = [x[0] for x in inventory_log]
    volumes = [x[1] for x in inventory_log]

    plt.figure(figsize=(12,4))
    plt.step(dates, volumes, where="post")
    plt.title("Gas Inventory in Storage Over Time")
    plt.xlabel("Date")
    plt.ylabel("Stored Volume (MMBtu)")
    plt.grid()
    plt.show()


def plot_cashflow(cashflow_log):
    dates = [x[0] for x in cashflow_log]
    values = [x[1] for x in cashflow_log]

    plt.figure(figsize=(12,4))
    plt.plot(dates, values, marker="o")
    plt.title("Cumulative Contract Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Contract Value ($)")
    plt.axhline(0, color="black", linestyle="--")
    plt.grid()
    plt.show()


#load the dataset and changing the column name to lowercase for better coding.
df = pd.read_csv('Dataset2/Nat_Gas.csv')
df.columns = df.columns.str.strip().str.lower()  # dates, prices

df["dates"] = pd.to_datetime(df["dates"], format="%m/%d/%y")
df = df.sort_values("dates")
df.set_index("dates", inplace=True)

#building daily curve from the date given
daily_df = df.resample("D").interpolate("linear")

#decomposing
decomp = seasonal_decompose(daily_df["prices"], model="additive", period=365)
trend = decomp.trend.dropna()
seasonal = decomp.seasonal


trend_df = trend.reset_index()
trend_df.columns = ["dates", "trend_price"]
trend_df["t"] = (trend_df["dates"] - trend_df["dates"].min()).dt.days

model = LinearRegression()
model.fit(trend_df[["t"]], trend_df["trend_price"])

#keeping one year as a bandwidth for forecast
last_date = daily_df.index.max()
future_dates = pd.date_range(start=last_date + timedelta(days=1),
                             end=last_date + timedelta(days=365),
                             freq="D")

future_t = (future_dates - trend_df["dates"].min()).days.values.reshape(-1,1)
future_trend = model.predict(future_t)

seasonal_pattern = seasonal[:365].values
future_prices = future_trend + seasonal_pattern

future_df = pd.DataFrame({"prices": future_prices}, index=future_dates)

#final market price curve
full_series = pd.concat([daily_df, future_df])

#function for market price
def estimate_price(date_string):
    date = pd.to_datetime(date_string)

    if date not in full_series.index:
        date = full_series.index[full_series.index.get_indexer([date], method="nearest")[0]]

    return float(full_series.loc[date]["prices"])

#sorage contract pricing
def price_storage_contract(
    injection_dates,
    withdrawal_dates,
    volume_per_action,
    max_storage_volume,
    injection_rate,
    withdrawal_rate,
    storage_cost_per_day,
    injection_cost,
    withdrawal_cost
):

    inventory = 0.0
    value = 0.0

    events = []
    for d in injection_dates:
        events.append((pd.to_datetime(d), "inject"))
    for d in withdrawal_dates:
        events.append((pd.to_datetime(d), "withdraw"))

    events.sort(key=lambda x: x[0])
    last_date = events[0][0]

    for date, action in events:

        # Storage rent
        days = (date - last_date).days
        if days > 0:
            value -= days * storage_cost_per_day

        price = estimate_price(date)

        if action == "inject":
            if volume_per_action > injection_rate:
                raise ValueError("Injection rate exceeded")
            if inventory + volume_per_action > max_storage_volume:
                raise ValueError("Storage capacity exceeded")

            value -= volume_per_action * price
            value -= injection_cost
            inventory += volume_per_action

        else:
            if volume_per_action > withdrawal_rate:
                raise ValueError("Withdrawal rate exceeded")
            if inventory < volume_per_action:
                raise ValueError("Insufficient inventory")

            value += volume_per_action * price
            value -= withdrawal_cost
            inventory -= volume_per_action

        last_date = date

    return round(value, 2)

#sample test case
injections = ["2024-04-01", "2024-05-01", "2024-06-01"]
withdrawals = ["2024-12-15", "2025-01-15", "2025-02-15"]

value, inventory_log, cashflow_log = price_storage_contract_with_logs(
    injection_dates=injections,
    withdrawal_dates=withdrawals,
    volume_per_action=150_000,
    max_storage_volume=600_000,
    injection_rate=200_000,
    withdrawal_rate=200_000,
    storage_cost_per_day=200,
    injection_cost=2000,
    withdrawal_cost=2000
)

print("Estimated contract value ($):", value)

plot_price_with_actions(injections, withdrawals)
plot_inventory(inventory_log)
plot_cashflow(cashflow_log)


#taking user input
while True:
    d = input("\nenter date (YYYY-MM-DD) or type 'exit': ")
    if d.lower() == "exit":
        break
    try:
        print("estimated price:", round(estimate_price(d), 3))
    except:
        print("invalid date format")
