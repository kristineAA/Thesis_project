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

# ============================
# Cost Setup
# ============================
O = 411                         # order cost - the centers produce the blood - small fee for snacks and personell
H = 1                           # holding cost
E = 411                         # outdate cost

# ============================
# Problem Setup
# ============================
M_max = 5                       # maximum age of inventory
S = 1000                        # max inventory level (200 units can be produced per day, 4 days holding)
L = 1                           # lead time

# set of inventory ages
M = 1:M_max

# ============================
# Fetching the demand data for the scenarios
# ============================
df = CSV.read("daily_platelet_demand_2point5years.csv", DataFrame)

D = df.n                   # retriving the demand values from the data frame

demand_two_periods_prior = D[1:90]             # demand for the first 90 days, used for the first rolling horizon window

average_two_periods_prior = ceil(mean(demand_two_periods_prior))   # average demand for the first 90 days

B = [average_two_periods_prior,0,0,0,0]   # initial inventory levels for each age, based on the average demand of the first 90 days

# ============================
# Create functions for creating and running the rolling horizon model
# ============================
function create_rolling_horizon_model_cluster(window, rolling_horizon_length, T, D, B, s, G)
    global O, H, E, S, L, M
    # ============================
    # Model
    # ============================
    model = Model(Gurobi.Optimizer)

    # ============================
    # Variables
    # ============================
    @variable(model, q[t in T[window:window+rolling_horizon_length-1]] >= 0, Int)
    @variable(model, a[t in T[window:window+rolling_horizon_length-1], m in M] >= 0, Int)
    @variable(model, is[t in T[window:window+rolling_horizon_length-1], m in M] >= 0, Int)
    @variable(model, ie[t in T[window:window+rolling_horizon_length-1], m in M] >= 0, Int)
    @variable(model, v[t in T[window:window+rolling_horizon_length-1]] >= 0, Int)
    @variable(model, e[t in T[window:window+rolling_horizon_length-1]] >= 0, Int)
    @variable(model, f[t in T[window:window+rolling_horizon_length-1]] >= 0, Int)

    @variable(model, b[t in T[window:window+rolling_horizon_length-1]], Bin)

    # ============================
    # Objective
    # ============================

    @objective(model, Min,
        sum(O * q[t] + H * v[t] + E * e[t] + G * f[t] for t in T[window:window+rolling_horizon_length-1])
    )

    # ============================
    # Constraints
    # ============================

    # (2)
    @constraint(model, [t in T[window:window+rolling_horizon_length-1]],
        q[t] >= s - v[t]
    )

    # (3)
    @constraint(model, [t in T[window:window+rolling_horizon_length-1]],
        q[t] <= s - v[t] + S * (1 - b[t])
    )

    # (4)
    @constraint(model, [t in T[window:window+rolling_horizon_length-1]],
        q[t] <= S * b[t]
    )

    # (8)
    @constraint(model, [m in M],
        is[window,m] == B[m]
    )

    # (9)
    @constraint(model, [t in T[window:window+rolling_horizon_length-1], m in M],
        is[t,m] - ie[t,m] == a[t,m]
    )

    # (10) lead-time arrival into age 1
    @constraint(model, [t in T[window:window+rolling_horizon_length-1-L]],
        is[t+L,1] == q[t]
    )

    # (11) age flow: remaining inventory moves to next age the next period
    @constraint(model, [t in T[window:window+rolling_horizon_length-2], m in M[2:end]],
        is[t+1,m] == ie[t,m-1]
    )

    # (12)
    @constraint(model, [t in T[window:window+rolling_horizon_length-1]],
        v[t] == sum(ie[t,m] for m in M[1:end-1])
    )

    # (13)
    @constraint(model, [t in T[window:window+rolling_horizon_length-1]],
        e[t] == ie[t,5]
    )

    # (14)
    @constraint(model, [t in T[window:window+rolling_horizon_length-1]],
        sum(a[t,m] for m in M) + f[t] == D[t]
    )

    # ============================
    # Solve the model
    # ============================
    optimize!(model)

    # ============================
    # Calculate service level and return wanted values
    # ============================
    totalDemand = sum(D[t] for t in T[window:window+rolling_horizon_length-1])
    totalShortage = sum(value(f[t]) for t in T[window:window+rolling_horizon_length-1]) 
    serviceLevel = 100 * (totalDemand - totalShortage) / totalDemand

    return serviceLevel, value(s), objective_value(model), value.(q), value.(a), value.(is), value.(ie), value.(v), value.(e), value.(f)

end

function runRollingHorizonCluster(T, D, B, T_max, s, G)
    global L
    # ============================
    # Create tables for vanted stored values
    # ============================

    orderUpToValues = []
    serviceLevelValues = [] 
    objectiveValues = []
    q_values = []
    a_values = []
    is_values = []
    ie_values = []
    v_values = []
    e_values = []
    f_values = []

    # ============================
    # Run model for each window and store values
    # ============================
    rolling_horizon = 90
    w = 1
    for i in 1:7
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
        
        # run the model for the current window and store the results
        serviceLevel, s_value, obj_value, q_vals, a_vals, is_vals, ie_vals, v_vals, e_vals, f_vals = create_rolling_horizon_model_cluster(w, rolling_horizon, T, D, B, s[i], G)
        push!(orderUpToValues, s_value)
        push!(serviceLevelValues, serviceLevel)
        push!(objectiveValues, obj_value)
        push!(q_values, q_vals)
        push!(a_values, a_vals)
        push!(is_values, is_vals)
        push!(ie_values, ie_vals)
        push!(v_values, v_vals)
        push!(e_values, e_vals)
        push!(f_values, f_vals)
        GC.gc()        # Force garbage collection to clear memory from the previous model

        w += 90
    end

    # ============================
    # return wanted values
    # ============================
    return serviceLevelValues, objectiveValues, q_values, a_values, is_values, ie_values, v_values, e_values, f_values
end

function retrive_s_level(df)
    A = Matrix(df[:, 2])
    s = A'                       # retriving the demand values from the data frame
    s_levels = ceil.(Int, s)              # demand (integer)

    return s_levels
end

s_levels_moving_average = []
for i in 1:7
    push!(s_levels_moving_average, ceil(mean(D[(90*i+1):(90*(i+1))])))
end

actual_demand = D[181:end]  # actual demand for the last 2.5 years, used for evaluating the service level of the rolling horizon approach

T = 1:length(actual_demand)
T_max = length(actual_demand)

# ============================
# retrive model for 1 scenario with shortage cost G = 411
# ============================
s_levels_predicted_1scen_411 = (CSV.read("results_scalars_1scen_s_411_2026-06-17_00-05-00.csv", DataFrame)).OrderUpTo

s_levels_predicted_1scen_411 = s_levels_predicted_1scen_411[1:7]

serviceLevelValuesTestPredicted_1scen_411, objectiveValuesTestPredicted_1scen_411, q_valuesTestPredicted_1scen_411, a_valuesTestPredicted_1scen_411, is_valuesTestPredicted_1scen_411, ie_valuesTestPredicted_1scen_411, v_valuesTestPredicted_1scen_411, e_valuesTestPredicted_1scen_411, f_valuesTestPredicted_1scen_411 = runRollingHorizonCluster(T, actual_demand, B, T_max, s_levels_predicted_1scen_411, 411)

# ============================
# retrive model for 1 scenario with shortage cost G = 822
# ============================
s_levels_predicted_1scen_822 = (CSV.read("results_scalars_1scen_s_822_2026-06-17_00-05-05.csv", DataFrame)).OrderUpTo

s_levels_predicted_1scen_822 = s_levels_predicted_1scen_822[1:7]

serviceLevelValuesTestPredicted_1scen_822, objectiveValuesTestPredicted_1scen_822, q_valuesTestPredicted_1scen_822, a_valuesTestPredicted_1scen_822, is_valuesTestPredicted_1scen_822, ie_valuesTestPredicted_1scen_822, v_valuesTestPredicted_1scen_822, e_valuesTestPredicted_1scen_822, f_valuesTestPredicted_1scen_822 = runRollingHorizonCluster(T, actual_demand, B, T_max, s_levels_predicted_1scen_822, 822)

# ============================
# retrive model for 1 scenario with shortage cost G = 1233
# ============================
s_levels_predicted_1scen_1233 = (CSV.read("results_scalars_1scen_s_1233_2026-06-17_00-05-07.csv", DataFrame)).OrderUpTo

s_levels_predicted_1scen_1233 = s_levels_predicted_1scen_1233[1:7]

serviceLevelValuesTestPredicted_1scen_1233, objectiveValuesTestPredicted_1scen_1233, q_valuesTestPredicted_1scen_1233, a_valuesTestPredicted_1scen_1233, is_valuesTestPredicted_1scen_1233, ie_valuesTestPredicted_1scen_1233, v_valuesTestPredicted_1scen_1233, e_valuesTestPredicted_1scen_1233, f_valuesTestPredicted_1scen_1233 = runRollingHorizonCluster(T, actual_demand, B, T_max, s_levels_predicted_1scen_1233, 1233)

# ============================
# calculate average shortages for the models with different shortage costs
# ============================
shortageAverage_TestPredicted_1scen_411 = mean([sum(f) for f in f_valuesTestPredicted_1scen_411])
shortageAverage_TestPredicted_1scen_822 = mean([sum(f) for f in f_valuesTestPredicted_1scen_822])
shortageAverage_TestPredicted_1scen_1233 = mean([sum(f) for f in f_valuesTestPredicted_1scen_1233])

# ============================
# calculate average service levels for the models with different shortage costs
# ============================
serviceLevelAverage_TestPredicted_1scen_411 = mean(serviceLevelValuesTestPredicted_1scen_411)
serviceLevelAverage_TestPredicted_1scen_822 = mean(serviceLevelValuesTestPredicted_1scen_822)
serviceLevelAverage_TestPredicted_1scen_1233 = mean(serviceLevelValuesTestPredicted_1scen_1233)

# ============================
# calculate average objective values for the models with different shortage costs
# ============================
objectiveAverage_TestPredicted_1scen_411 = mean(objectiveValuesTestPredicted_1scen_411)
objectiveAverage_TestPredicted_1scen_822 = mean(objectiveValuesTestPredicted_1scen_822)
objectiveAverage_TestPredicted_1scen_1233 = mean(objectiveValuesTestPredicted_1scen_1233)

# ============================
# print desired values for the models with different shortage costs
# ============================

println("Average shortage for G=411: ", shortageAverage_TestPredicted_1scen_411)
println("Average shortage for G=822: ", shortageAverage_TestPredicted_1scen_822)
println("Average shortage for G=1233: ", shortageAverage_TestPredicted_1scen_1233)  

println("Average service level for G=411: ", serviceLevelAverage_TestPredicted_1scen_411, "%")
println("Average service level for G=822: ", serviceLevelAverage_TestPredicted_1scen_822, "%")
println("Average service level for G=1233: ", serviceLevelAverage_TestPredicted_1scen_1233, "%")

println("Average objective value for G=411: ", objectiveAverage_TestPredicted_1scen_411)
println("Average objective value for G=822: ", objectiveAverage_TestPredicted_1scen_822)
println("Average objective value for G=1233: ", objectiveAverage_TestPredicted_1scen_1233)

println("Order-up-to levels for G=411: ", s_levels_predicted_1scen_411)
println("Order-up-to levels for G=822: ", s_levels_predicted_1scen_822)
println("Order-up-to levels for G=1233: ", s_levels_predicted_1scen_1233)