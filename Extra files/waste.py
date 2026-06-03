import pandas as pd
from collections import deque
import os
os.chdir('/Users/kristineandersen/Desktop/Speciale/Thesis_project')
# =====================
# Inputs
# =====================
df = pd.read_csv("Input Files/daily_platelet_demand.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

order_up_to = {
    1: 157,
    2: 156,
    3: 154,
    4: 159,
    5: 159,
    6: 150,
    7: 146,
    8: 138,
}

period_starts = [
    "2018-03-12",
    "2018-06-10",
    "2018-09-08",
    "2018-12-07",
    "2019-03-07",
    "2019-06-05",
    "2019-09-03",
    "2019-12-02",
]

period_starts = pd.to_datetime(period_starts)

EXPIRY_DAYS = 5
LEAD_TIME = 1
INITIAL_INVENTORY = 150


# =====================
# Add period/window
# =====================
df["period"] = pd.cut(
    df["date"],
    bins=list(period_starts) + [df["date"].max() + pd.Timedelta(days=1)],
    labels=range(1, len(period_starts) + 1),
    right=False
)

df = df.dropna(subset=["period"]).copy()
df["period"] = df["period"].astype(int)


# =====================
# FIFO inventory simulation
# =====================
inventory = deque()
pending_orders = deque()

first_day = df["date"].min()
inventory.append({
    "quantity": INITIAL_INVENTORY,
    "expiry_date": first_day + pd.Timedelta(days=EXPIRY_DAYS - 1)
})

results = []

for _, row in df.iterrows():
    date = row["date"]
    demand = int(row["n"])
    period = int(row["period"])
    S = order_up_to[period]

    starting_inventory = sum(batch["quantity"] for batch in inventory)
    starting_pending_orders = sum(order["quantity"] for order in pending_orders)

    # 1) Receive orders arriving today
    received_today = 0

    while pending_orders and pending_orders[0]["arrival_date"] <= date:
        order = pending_orders.popleft()
        received_today += order["quantity"]

        inventory.append({
            "quantity": order["quantity"],
            "expiry_date": date + pd.Timedelta(days=EXPIRY_DAYS - 1)
        })

    inventory_after_receipt = sum(batch["quantity"] for batch in inventory)

    # 2) Remove expired products before demand
    waste = 0
    remaining_inventory = deque()

    while inventory:
        batch = inventory.popleft()

        if batch["expiry_date"] < date:
            waste += batch["quantity"]
        else:
            remaining_inventory.append(batch)

    inventory = remaining_inventory
    inventory_after_expiry = sum(batch["quantity"] for batch in inventory)

    # 3) Check oldest product before demand
    if inventory:
        days_until_expiry_oldest_before_demand = (
            inventory[0]["expiry_date"] - date
        ).days
        oldest_batch_quantity_before_demand = inventory[0]["quantity"]
    else:
        days_until_expiry_oldest_before_demand = None
        oldest_batch_quantity_before_demand = 0

    # 4) Satisfy demand using FIFO
    unmet_demand = demand

    while unmet_demand > 0 and inventory:
        batch = inventory[0]
        used = min(batch["quantity"], unmet_demand)

        batch["quantity"] -= used
        unmet_demand -= used

        if batch["quantity"] == 0:
            inventory.popleft()

    fulfilled_demand = demand - unmet_demand
    ending_inventory_before_order = sum(batch["quantity"] for batch in inventory)

    # 5) Check oldest product after demand
    if inventory:
        days_until_expiry_oldest_after_demand = (
            inventory[0]["expiry_date"] - date
        ).days
        oldest_batch_quantity_after_demand = inventory[0]["quantity"]
    else:
        days_until_expiry_oldest_after_demand = None
        oldest_batch_quantity_after_demand = 0

    # 6) Place order up to level S
    inventory_position_before_order = (
        sum(batch["quantity"] for batch in inventory)
        + sum(order["quantity"] for order in pending_orders)
    )

    order_quantity = max(S - inventory_position_before_order, 0)

    if order_quantity > 0:
        pending_orders.append({
            "quantity": order_quantity,
            "arrival_date": date + pd.Timedelta(days=LEAD_TIME)
        })

    ending_inventory = sum(batch["quantity"] for batch in inventory)
    pending_orders_after_order = sum(order["quantity"] for order in pending_orders)

    # Helpful readable queue snapshots
    inventory_queue = [
        {
            "quantity": batch["quantity"],
            "days_until_expiry": (batch["expiry_date"] - date).days,
            "expiry_date": batch["expiry_date"].date()
        }
        for batch in inventory
    ]

    pending_order_queue = [
        {
            "quantity": order["quantity"],
            "arrival_date": order["arrival_date"].date()
        }
        for order in pending_orders
    ]

    results.append({
        "date": date,
        "period": period,
        "n": demand,
        "order_up_to": S,

        "starting_inventory": starting_inventory,
        "starting_pending_orders": starting_pending_orders,

        "received_today": received_today,
        "inventory_after_receipt": inventory_after_receipt,

        "waste_today": waste,
        "inventory_after_expiry": inventory_after_expiry,

        "fulfilled_demand": fulfilled_demand,
        "unmet_demand": unmet_demand,
        "ending_inventory_before_order": ending_inventory_before_order,

        "inventory_position_before_order": inventory_position_before_order,
        "ordered_today": order_quantity,
        "pending_orders_after_order": pending_orders_after_order,
        "ending_inventory": ending_inventory,

        "days_until_expiry_oldest_before_demand": days_until_expiry_oldest_before_demand,
        "oldest_batch_quantity_before_demand": oldest_batch_quantity_before_demand,
        "days_until_expiry_oldest_after_demand": days_until_expiry_oldest_after_demand,
        "oldest_batch_quantity_after_demand": oldest_batch_quantity_after_demand,

        "inventory_queue_after_day": inventory_queue,
        "pending_order_queue_after_day": pending_order_queue,
    })


results = pd.DataFrame(results)

results.to_csv("inventory_simulation_daily_detailed.csv", index=False)