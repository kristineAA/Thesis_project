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
const G = 822                           # shortage cost
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

function create_rolling_horizon_model_cluster(window, rolling_horizon_length, T, Xi, D, Pxi, B)
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
    @variable(model, v[xi in Xi, t in T[window:window+rolling_horizon_length-1]] >= 0, Int)
    @variable(model, e[xi in Xi, t in T[window:window+rolling_horizon_length-1]] >= 0)
    @variable(model, f[xi in Xi, t in T[window:window+rolling_horizon_length-1]] >= 0)

    @variable(model, b[xi in Xi, t in T[window:window+rolling_horizon_length-1]], Bin)
    @variable(model, s >= 0, Int)

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
    set_optimizer_attribute(model, "MIPGap", 0.01)

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

    #println("Service Level: ", serviceLevel, "%")
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


    # ============================
    # Print vanted values
    # ============================
    return orderUpToValues, serviceLevelValues, objectiveValues
end

# ============================
# Clusters
# ============================

# ============================
# With 1 scenario
# ============================

snr = 1

# Fetching the demand data for the scenarios
global snr
dfClus1 = CSV.read("oneScen.csv", DataFrame)

T, Xi, T_max, Xi_max, D_xi_t, Pxi = dataSetupRHCluster1Scenario(dfClus1, snr)


# ============================
# Create tables for vanted stored values
# ============================

orderUpToValuesClus1 = []
serviceLevelValuesClus1 = [] 
objectiveValuesClus1 = []

# ============================
# Run model for each window and store values
# ============================

orderUpToValuesClus1, serviceLevelValuesClus1, objectiveValuesClus1 = runRollingHorizonCluster(T, Xi, D_xi_t, Pxi, B, T_max)

# ============================
# Print vanted values
# ============================
println("Objective value: ", objectiveValuesClus1)
println("s = ", orderUpToValuesClus1)   
println("Service levels: ", serviceLevelValuesClus1)

# ============================
# Save results to CSV for 1 scenario
# ============================

# Save scalar values (one per window) to a single CSV
timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
df_scalars = DataFrame(
    Window = 1:length(orderUpToValuesClus1),
    OrderUpTo = orderUpToValuesClus1,
    ServiceLevel = serviceLevelValuesClus1,
    Objective = objectiveValuesClus1
)
CSV.write("results_scalars_1scen_$timestamp.csv", df_scalars)

println("Results for 1 scenario saved to CSV files.")


# ============================
# With 3 scenarios
# ============================

snr = 3

# Fetching the demand data for the scenarios
global snr
dfClus3_1 = CSV.read("3Scen_new/predictions_block_1_2018-03-12_to_2018-09-07.csv", DataFrame)
dfClus3_2 = CSV.read("3Scen_new/predictions_block_2_2018-06-10_to_2018-12-06.csv", DataFrame)
dfClus3_3 = CSV.read("3Scen_new/predictions_block_3_2018-09-08_to_2019-03-06.csv", DataFrame)
dfClus3_4 = CSV.read("3Scen_new/predictions_block_4_2018-12-07_to_2019-06-04.csv", DataFrame)
dfClus3_5 = CSV.read("3Scen_new/predictions_block_5_2019-03-07_to_2019-09-02.csv", DataFrame)
dfClus3_6 = CSV.read("3Scen_new/predictions_block_6_2019-06-05_to_2019-12-01.csv", DataFrame)
dfClus3_7 = CSV.read("3Scen_new/predictions_block_7_2019-09-03_to_2020-02-29.csv", DataFrame)
dfClus3_8 = CSV.read("3Scen_new/predictions_block_8_2019-12-02_to_2020-02-29.csv", DataFrame)

dfClus3 = vcat(dfClus3_1, dfClus3_2, dfClus3_3, dfClus3_4, dfClus3_5, dfClus3_6, dfClus3_7, dfClus3_8)

T, Xi, T_max, Xi_max, D_xi_t, Pxi = dataSetupRHCluster(dfClus3, snr)

# ============================
# Create tables for vanted stored values
# ============================

orderUpToValuesClus3 = []
serviceLevelValuesClus3 = [] 
objectiveValuesClus3 = []

# ============================
# Run model for each window and store values
# ============================

orderUpToValuesClus3, serviceLevelValuesClus3, objectiveValuesClus3 = runRollingHorizonCluster(T, Xi, D_xi_t, Pxi, B, T_max)

# ============================
# Print vanted values
# ============================
println("Objective value: ", objectiveValuesClus3)
println("s = ", orderUpToValuesClus3)   
println("Service levels: ", serviceLevelValuesClus3)

# ============================
# Save results to CSV for 3 scenarios
# ============================

timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")

# Save scalar values (one per window) to a single CSV
df_scalars = DataFrame(
    Window = 1:length(orderUpToValuesClus3),
    OrderUpTo = orderUpToValuesClus3,
    ServiceLevel = serviceLevelValuesClus3,
    Objective = objectiveValuesClus3
)
CSV.write("results_scalars_3scen_$timestamp.csv", df_scalars)

println("Results for 3 scenarios saved to CSV files.")

# ============================
# With 5 scenarios
# ============================

snr = 5

# Fetching the demand data for the scenarios
global snr
dfClus5_1 = CSV.read("5Scen_new/predictions_block_1_2018-03-12_to_2018-09-07.csv", DataFrame)
dfClus5_2 = CSV.read("5Scen_new/predictions_block_2_2018-06-10_to_2018-12-06.csv", DataFrame)
dfClus5_3 = CSV.read("5Scen_new/predictions_block_3_2018-09-08_to_2019-03-06.csv", DataFrame)
dfClus5_4 = CSV.read("5Scen_new/predictions_block_4_2018-12-07_to_2019-06-04.csv", DataFrame)
dfClus5_5 = CSV.read("5Scen_new/predictions_block_5_2019-03-07_to_2019-09-02.csv", DataFrame)
dfClus5_6 = CSV.read("5Scen_new/predictions_block_6_2019-06-05_to_2019-12-01.csv", DataFrame)
dfClus5_7 = CSV.read("5Scen_new/predictions_block_7_2019-09-03_to_2020-02-29.csv", DataFrame)
dfClus5_8 = CSV.read("5Scen_new/predictions_block_8_2019-12-02_to_2020-02-29.csv", DataFrame)

dfClus5 = vcat(dfClus5_1, dfClus5_2, dfClus5_3, dfClus5_4, dfClus5_5, dfClus5_6, dfClus5_7, dfClus5_8)

T, Xi, T_max, Xi_max, D_xi_t, Pxi = dataSetupRHCluster(dfClus5, snr)

# ============================
# Create tables for vanted stored values
# ============================

orderUpToValuesClus5 = []
serviceLevelValuesClus5 = [] 
objectiveValuesClus5 = []

# ============================
# Run model for each window and store values
# ============================

orderUpToValuesClus5, serviceLevelValuesClus5, objectiveValuesClus5 = runRollingHorizonCluster(T, Xi, D_xi_t, Pxi, B, T_max)

# ============================
# Print vanted values
# ============================
println("Objective value: ", objectiveValuesClus5)
println("s = ", orderUpToValuesClus5)   
println("Service levels: ", serviceLevelValuesClus5)

# ============================
# Save results to CSV for 5 scenarios
# ============================

timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")

# Save scalar values (one per window) to a single CSV
df_scalars = DataFrame(
    Window = 1:length(orderUpToValuesClus5),
    OrderUpTo = orderUpToValuesClus5,
    ServiceLevel = serviceLevelValuesClus5,
    Objective = objectiveValuesClus5
)
CSV.write("results_scalars_5scen_$timestamp.csv", df_scalars)

println("Results for 5 scenarios saved to CSV files.")

# ============================
# With 7 scenarios
# ============================

snr = 7

# Fetching the demand data for the scenarios
global snr
dfClus7_1 = CSV.read("7Scen_new/predictions_block_1_2018-03-12_to_2018-09-07.csv", DataFrame)
dfClus7_2 = CSV.read("7Scen_new/predictions_block_2_2018-06-10_to_2018-12-06.csv", DataFrame)
dfClus7_3 = CSV.read("7Scen_new/predictions_block_3_2018-09-08_to_2019-03-06.csv", DataFrame)
dfClus7_4 = CSV.read("7Scen_new/predictions_block_4_2018-12-07_to_2019-06-04.csv", DataFrame)
dfClus7_5 = CSV.read("7Scen_new/predictions_block_5_2019-03-07_to_2019-09-02.csv", DataFrame)
dfClus7_6 = CSV.read("7Scen_new/predictions_block_6_2019-06-05_to_2019-12-01.csv", DataFrame)
dfClus7_7 = CSV.read("7Scen_new/predictions_block_7_2019-09-03_to_2020-02-29.csv", DataFrame)
dfClus7_8 = CSV.read("7Scen_new/predictions_block_8_2019-12-02_to_2020-02-29.csv", DataFrame)

dfClus7 = vcat(dfClus7_1, dfClus7_2, dfClus7_3, dfClus7_4, dfClus7_5, dfClus7_6, dfClus7_7, dfClus7_8)

T, Xi, T_max, Xi_max, D_xi_t, Pxi = dataSetupRHCluster(dfClus7, snr)

# ============================
# Create tables for vanted stored values
# ============================

orderUpToValuesClus7 = []
serviceLevelValuesClus7 = [] 
objectiveValuesClus7 = []

# ============================
# Run model for each window and store values
# ============================

orderUpToValuesClus7, serviceLevelValuesClus7, objectiveValuesClus7 = runRollingHorizonCluster(T, Xi, D_xi_t, Pxi, B, T_max)

# ============================
# Print vanted values
# ============================
println("Objective value: ", objectiveValuesClus7)
println("s = ", orderUpToValuesClus7)   
println("Service levels: ", serviceLevelValuesClus7)

# ============================
# Save results to CSV for 7 scenarios
# ============================

timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")

# Save scalar values (one per window) to a single CSV
df_scalars = DataFrame(
    Window = 1:length(orderUpToValuesClus7),
    OrderUpTo = orderUpToValuesClus7,
    ServiceLevel = serviceLevelValuesClus7,
    Objective = objectiveValuesClus7
)
CSV.write("results_scalars_7scen_$timestamp.csv", df_scalars)

println("Results for 7 scenarios saved to CSV files.")
