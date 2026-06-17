# ============================
# The rolling horizon method
# ============================

# ============================
# General Setup
# ============================

# ============================
# Setup
# ============================
using JuMP 
using Gurobi 
using CSV
using DataFrames
using MathOptInterface
using Statistics
using Plots
using Dates

# ============================
# Setup of data for all runs
# ============================

# Cost setup
const O = 411                           # order cost - the centers produce the blood 
const H = 1                             # holding cost
const E = 411                           # outdate cost

# Problem setup
const M_max = 5                         # maximum age of inventory
const S = 1000                          # max inventory level (200 units can be produced per day, 4 days holding)
const L = 1                             # lead time

# set of inventory ages
const M = 1:M_max

# Fetching the demand data for the scenarios
df = CSV.read("daily_platelet_demand_2point5years.csv", DataFrame)

D = df.n                   # retriving the demand values from the data frame

demand_two_periods_prior = D[1:90]             # demand for the first 90 days, used for the first rolling horizon window

average_two_periods_prior = ceil(mean(demand_two_periods_prior))   # average demand for the first 90 days

B = [average_two_periods_prior,0,0,0,0]   # initial inventory levels for each age, based on the average demand of the first 90 days

# ============================
# Getting the rest of the data setup
# ============================

function dataSetupRHCluster1Scenario(df, S)
    A = Matrix(df[:, 2:end - S])
    D_xi_t = A'                       # retriving the demand values from the data frame

    D = ceil.(Int, D_xi_t)              # demand (integer)

    T_max = size(D_xi_t, 2)             # number of time periods
    Xi_max = S                          # number of demand scenarios

    T = 1:T_max
    Xi = 1:Xi_max

    # Fetching the scenario probabilities
    Pxi = Matrix(df[:, end - S + 1:end])
    Pxi = Pxi'                               # retriving the demand values from the data frame

    return T, Xi, T_max, Xi_max, D, Pxi
end

function dataSetupRHCluster(df, S)
    A = Matrix(df[:, 7 + S : 7 + 2*S])
    D_xi_t = A'                       # retriving the demand values from the data frame

    D = ceil.(Int, D_xi_t)              # demand (integer)

    T_max = size(D_xi_t, 2)             # number of time periods
    Xi_max = S                          # number of demand scenarios

    T = 1:T_max
    Xi = 1:Xi_max

    # Fetching the scenario probabilities
    Pxi = Matrix(df[:, 7 : 7 + S])
    Pxi = Pxi'                               # retriving the demand values from the data frame

    return T, Xi, T_max, Xi_max, D, Pxi
end

# ============================
# Setup of method
# ============================

function create_rolling_horizon_model_cluster(window, rolling_horizon_length, T, Xi, D, Pxi, B, G)
    # ============================
    # Model
    # ============================
    model = Model(Gurobi.Optimizer)

    # ============================
    # Variables
    # ============================
    @variable(model, q[xi in Xi, t in T[window:window+rolling_horizon_length-1]] >= 0, Int)
    @variable(model, a[xi in Xi, t in T[window:window+rolling_horizon_length-1], m in M] >= 0)
    @variable(model, is[xi in Xi, t in T[window:window+rolling_horizon_length-1], m in M] >= 0)
    @variable(model, ie[xi in Xi, t in T[window:window+rolling_horizon_length-1], m in M] >= 0)
    @variable(model, v[xi in Xi, t in T[window:window+rolling_horizon_length-1]] >= 0)
    @variable(model, e[xi in Xi, t in T[window:window+rolling_horizon_length-1]] >= 0)
    @variable(model, f[xi in Xi, t in T[window:window+rolling_horizon_length-1]] >= 0)

    @variable(model, b[xi in Xi, t in T[window:window+rolling_horizon_length-1]], Bin)
    @variable(model, s >= 0)

    # ============================
    # Objective
    # ============================

    @objective(model, Min,
        sum(sum((O * q[xi, t] + H * v[xi,t] + E * e[xi,t] + G * f[xi,t])* Pxi[xi, t] for t in T[window:window+rolling_horizon_length-1]) for xi in Xi)
    )

    # ============================
    # Constraints
    # ============================

    # (2)
    @constraint(model, [xi in Xi, t in T[window:window+rolling_horizon_length-1]],
        q[xi,t] >= s - v[xi,t]
    )

    # (3)
    @constraint(model, [xi in Xi, t in T[window:window+rolling_horizon_length-1]],
        q[xi,t] <= s - v[xi,t] + S * (1 - b[xi,t])
    )

    # (4)
    @constraint(model, [xi in Xi, t in T[window:window+rolling_horizon_length-1]],
        q[xi,t] <= S * b[xi,t]
    )

    # (8)
    @constraint(model, [xi in Xi, m in M],
        is[xi,window,m] == B[m]
    )

    # (9)
    @constraint(model, [xi in Xi, t in T[window:window+rolling_horizon_length-1], m in M],
        is[xi,t,m] - ie[xi,t,m] == a[xi,t,m]
    )

    # (10) lead-time arrival into age 1
    @constraint(model, [xi in Xi, t in T[window:window+rolling_horizon_length-1-L]],
        is[xi,t+L,1] == q[xi,t]
    )

    # (11) age flow: remaining inventory moves to next age the next period
    @constraint(model, [xi in Xi, t in T[window:window+rolling_horizon_length-2], m in M[2:end]],
        is[xi,t+1,m] == ie[xi,t,m-1]
    )

    # (12)
    @constraint(model, [xi in Xi, t in T[window:window+rolling_horizon_length-1]],
        v[xi,t] == sum(ie[xi,t,m] for m in M[1:end-1])
    )

    # (13)
    @constraint(model, [xi in Xi, t in T[window:window+rolling_horizon_length-1]],
        e[xi,t] == ie[xi,t,5]
    )

    # (14)
    @constraint(model, [xi in Xi, t in T[window:window+rolling_horizon_length-1]],
        sum(a[xi,t,m] for m in M) + f[xi,t] == D[xi, t]
    )

    # Set MIPGap to 1% (0.01)
    # set_optimizer_attribute(model, "MIPGap", 0.01)

    # ============================
    # Solve the model
    # ============================
    optimize!(model)

    # Check if the model solved successfully
    if termination_status(model) != MOI.OPTIMAL
        return NaN, NaN, NaN
    end

    totalDemand = sum(D[xi, t] for xi in Xi, t in T[window:window+rolling_horizon_length-1])
    totalShortage = sum(value(f[xi,t]) for xi in Xi, t in T[window:window+rolling_horizon_length-1]) 
    serviceLevel = 100 * (totalDemand - totalShortage) / totalDemand

    return serviceLevel, value(s), objective_value(model), value.(ie)

end

# ============================
# Code for running the rolling horizon
# ============================

function runRollingHorizonCluster(T, Xi, D, Pxi, B, T_max)
    # ============================
    # Create tables for vanted stored values
    # ============================

    orderUpToValues = [] 
    serviceLevelValues = [] 
    objectiveValues = []
    ie_values = []  # to store the inventory levels at the end of each window, to be used as input for the next window

    # ============================
    # Run model for each window and store values
    # ============================
    rolling_horizon = 180
    w = 1
    for i in 1:8
        # Update the initial inventory levels B based on the results from the previous window
        if i > 1
            # Get the last DenseAxisArray (from the most recent window)
            ie_last = ie_values[end]

            # Flatten it to a vector (DenseAxisArray supports vec)
            ie_flat = vec(ie_last)

            # Extract the last 5 values
            last5 = ie_flat[end-4:end]
            last5 = last5[1:end-1]
            b1_value = sum(last5) < orderUpToValues[i-1] ? orderUpToValues[i-1] - sum(last5) : 0
            B[1] = b1_value
            B[2:end] = last5
        end
        # Determine the horizon length for the current window (180 for full windows, 90 for the last window if it exceeds T_max)
        horizon = w <= T_max - L - rolling_horizon ? rolling_horizon : 90
        # run the model for the current window and store the results
        sl, sv, ov, ie_vals = create_rolling_horizon_model_cluster(w, horizon, T, Xi, D, Pxi, B)
        push!(orderUpToValues, sv)
        push!(serviceLevelValues, sl)
        push!(objectiveValues, ov)
        push!(ie_values, ie_vals)
        # Update w for the next window
        w += 90
    end
end
# ============================
# With 1 scenario
# ============================

snr = 1

# Fetching the demand data for the scenarios
global snr
dfClus1 = CSV.read("oneScen.csv", DataFrame)

T, Xi, T_max, Xi_max, D_xi_t, Pxi = dataSetupRHCluster1Scenario(dfClus1, snr)


# ============================
# Create tables for vanted stored values for 1 scenario with shortage cost G = 411
# ============================

orderUpToValuesClus1 = []
serviceLevelValuesClus1 = [] 
objectiveValuesClus1 = []

# ============================
# Run model for each window and store values for 1 scenario with shortage cost G = 411
# ============================

orderUpToValuesClus1, serviceLevelValuesClus1, objectiveValuesClus1 = runRollingHorizonCluster(T, Xi, D_xi_t, Pxi, B, T_max, 411)

# ============================
# Save results to CSV for 1 scenario with shortage cost G = 411
# ============================

# Save scalar values (one per window) to a single CSV
timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
df_scalars = DataFrame(
    Window = 1:length(orderUpToValuesClus1),
    OrderUpTo = orderUpToValuesClus1,
    ServiceLevel = serviceLevelValuesClus1,
    Objective = objectiveValuesClus1
)
CSV.write("results_scalars_1scen_s_411_$timestamp.csv", df_scalars)

println("Results for 1 scenario saved to CSV files.")

# ============================
# Create tables for vanted stored values for 1 scenario with shortage cost G = 822
# ============================

orderUpToValuesClus1_822 = []
serviceLevelValuesClus1_822 = [] 
objectiveValuesClus1_822 = []

# ============================
# Run model for each window and store values for 1 scenario with shortage cost G = 822
# ============================

orderUpToValuesClus1_822, serviceLevelValuesClus1_822, objectiveValuesClus1_822 = runRollingHorizonCluster(T, Xi, D_xi_t, Pxi, B, T_max, 822)

# ============================
# Save results to CSV for 1 scenario with shortage cost G = 822
# ============================

# Save scalar values (one per window) to a single CSV
timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
df_scalars = DataFrame(
    Window = 1:length(orderUpToValuesClus1_822),
    OrderUpTo = orderUpToValuesClus1_822,
    ServiceLevel = serviceLevelValuesClus1_822,
    Objective = objectiveValuesClus1_822
)
CSV.write("results_scalars_1scen_s_822_$timestamp.csv", df_scalars)

println("Results for 1 scenario saved to CSV files.")

# ============================
# Create tables for vanted stored values for 1 scenario with shortage cost G = 1233
# ============================

orderUpToValuesClus1_1233 = []
serviceLevelValuesClus1_1233 = [] 
objectiveValuesClus1_1233 = []

# ============================
# Run model for each window and store values for 1 scenario with shortage cost G = 1233
# ============================

orderUpToValuesClus1_1233, serviceLevelValuesClus1_1233, objectiveValuesClus1_1233 = runRollingHorizonCluster(T, Xi, D_xi_t, Pxi, B, T_max, 1233)

# ============================
# Save results to CSV for 1 scenario with shortage cost G = 1233
# ============================

# Save scalar values (one per window) to a single CSV
timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
df_scalars = DataFrame(
    Window = 1:length(orderUpToValuesClus1_1233),
    OrderUpTo = orderUpToValuesClus1_1233,
    ServiceLevel = serviceLevelValuesClus1_1233,
    Objective = objectiveValuesClus1_1233
)
CSV.write("results_scalars_1scen_s_1233_$timestamp.csv", df_scalars)

println("Results for 1 scenario saved to CSV files.")
