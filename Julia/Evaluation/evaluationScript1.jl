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

# Cost setup
G = 822                         # shortage cost
O = 411                         # order cost - the centers produce the blood - small fee for snacks and personell
H = 1                           # holding cost
E = 411                         # outdate cost

# Problem setup
M_max = 5                       # maximum age of inventory
S = 1000                        # max inventory level (200 units can be produced per day, 4 days holding)
L = 1                           # lead time

# set of inventory ages
M = 1:M_max

# Fetching the demand data for the scenarios
df = CSV.read("daily_platelet_demand_2point5years.csv", DataFrame)

D = df.n                   # retriving the demand values from the data frame

demand_two_periods_prior = D[1:90]             # demand for the first 90 days, used for the first rolling horizon window

average_two_periods_prior = ceil(mean(demand_two_periods_prior))   # average demand for the first 90 days

B = [average_two_periods_prior,0,0,0,0]   # initial inventory levels for each age, based on the average demand of the first 90 days


function create_rolling_horizon_model_cluster(window, rolling_horizon_length, T, D, B, s)
    global G, O, H, E, S, L, M
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

    totalDemand = sum(D[t] for t in T[window:window+rolling_horizon_length-1])
    totalShortage = sum(value(f[t]) for t in T[window:window+rolling_horizon_length-1]) 
    serviceLevel = 100 * (totalDemand - totalShortage) / totalDemand

    return serviceLevel, value(s), objective_value(model), value.(q), value.(a), value.(is), value.(ie), value.(v), value.(e), value.(f)

end

function runRollingHorizonCluster(T, D, B, T_max,s)
    global L
    # ============================
    # Create tables for wanted stored values
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
            
        serviceLevel, s_value, obj_value, q_vals, a_vals, is_vals, ie_vals, v_vals, e_vals, f_vals = create_rolling_horizon_model_cluster(w, rolling_horizon, T, D, B, s[i])
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
    # Return wanted values
    # ============================
    return serviceLevelValues, objectiveValues, q_values, a_values, is_values, ie_values, v_values, e_values, f_values
end

function retrive_s_level(df)
    A = Matrix(df[:, 2])
    s = A'                                  # retriving the demand values from the data frame
    s_levels = ceil.(Int, s)                # demand (integer)

    return s_levels
end

# Fetching the demand data for the scenarios
df = CSV.read("daily_platelet_demand_2point5years.csv", DataFrame)

D = df.n                   # retriving the demand values from the data frame

demand_two_periods_prior = D[1:90]             # demand for the first 90 days, used for the first rolling horizon window

average_two_periods_prior = ceil(mean(demand_two_periods_prior))   # average demand for the first 90 days

B = [average_two_periods_prior,0,0,0,0]   # initial inventory levels for each age, based on the average demand of the first 90 days

s_levels_moving_average = []
for i in 1:7
    push!(s_levels_moving_average, ceil(mean(D[(90*i+1):(90*(i+1))])))
end

s_levels_moving_average_plus_10 = s_levels_moving_average .+ 10
s_levels_moving_average_plus_20 = s_levels_moving_average .+ 20

s_levels_predicted_1scen = (CSV.read("results_scalars_1scen_2026-06-17_08-48-00.csv", DataFrame)).OrderUpTo
s_levels_predicted_3scen = (CSV.read("results_scalars_3scen_2026-06-17_08-51-35.csv", DataFrame)).OrderUpTo
s_levels_predicted_5scen = (CSV.read("results_scalars_5scen_2026-06-17_09-10-15.csv", DataFrame)).OrderUpTo
s_levels_predicted_7scen = (CSV.read("results_scalars_7scen_2026-06-17_12-03-50.csv", DataFrame)).OrderUpTo

s_levels_predicted_1scen = s_levels_predicted_1scen[1:7]
s_levels_predicted_3scen = s_levels_predicted_3scen[1:7]
s_levels_predicted_5scen = s_levels_predicted_5scen[1:7]
s_levels_predicted_7scen = s_levels_predicted_7scen[1:7]

actual_demand = D[181:end]  # actual demand for the last 2.5 years, used for evaluating the service level of the rolling horizon approach

T = 1:length(actual_demand)
T_max = length(actual_demand)


serviceLevelValuesTestMovingAverage, objectiveValuesTestMovingAverage, q_valuesTestMovingAverage, a_valuesTestMovingAverage, is_valuesTestMovingAverage, ie_valuesTestMovingAverage, v_valuesTestMovingAverage, e_valuesTestMovingAverage, f_valuesTestMovingAverage = runRollingHorizonCluster(T, actual_demand, B, T_max, s_levels_moving_average)
serviceLevelValuesTestMovingAverage10, objectiveValuesTestMovingAverage10, q_valuesTestMovingAverage10, a_valuesTestMovingAverage10, is_valuesTestMovingAverage10, ie_valuesTestMovingAverage10, v_valuesTestMovingAverage10, e_valuesTestMovingAverage10, f_valuesTestMovingAverage10 = runRollingHorizonCluster(T, actual_demand, B, T_max, s_levels_moving_average_plus_10)
serviceLevelValuesTestMovingAverage20, objectiveValuesTestMovingAverage20, q_valuesTestMovingAverage20, a_valuesTestMovingAverage20, is_valuesTestMovingAverage20, ie_valuesTestMovingAverage20, v_valuesTestMovingAverage20, e_valuesTestMovingAverage20, f_valuesTestMovingAverage20 = runRollingHorizonCluster(T, actual_demand, B, T_max, s_levels_moving_average_plus_20)
serviceLevelValuesTestPredicted_1scen, objectiveValuesTestPredicted_1scen, q_valuesTestPredicted_1scen, a_valuesTestPredicted_1scen, is_valuesTestPredicted_1scen, ie_valuesTestPredicted_1scen, v_valuesTestPredicted_1scen, e_valuesTestPredicted_1scen, f_valuesTestPredicted_1scen = runRollingHorizonCluster(T, actual_demand, B, T_max, s_levels_predicted_1scen)
serviceLevelValuesTestPredicted_3scen, objectiveValuesTestPredicted_3scen, q_valuesTestPredicted_3scen, a_valuesTestPredicted_3scen, is_valuesTestPredicted_3scen, ie_valuesTestPredicted_3scen, v_valuesTestPredicted_3scen, e_valuesTestPredicted_3scen, f_valuesTestPredicted_3scen = runRollingHorizonCluster(T, actual_demand, B, T_max, s_levels_predicted_3scen)
serviceLevelValuesTestPredicted_5scen, objectiveValuesTestPredicted_5scen, q_valuesTestPredicted_5scen, a_valuesTestPredicted_5scen, is_valuesTestPredicted_5scen, ie_valuesTestPredicted_5scen, v_valuesTestPredicted_5scen, e_valuesTestPredicted_5scen, f_valuesTestPredicted_5scen = runRollingHorizonCluster(T, actual_demand, B, T_max, s_levels_predicted_5scen)
serviceLevelValuesTestPredicted_7scen, objectiveValuesTestPredicted_7scen, q_valuesTestPredicted_7scen, a_valuesTestPredicted_7scen, is_valuesTestPredicted_7scen, ie_valuesTestPredicted_7scen, v_valuesTestPredicted_7scen, e_valuesTestPredicted_7scen, f_valuesTestPredicted_7scen = runRollingHorizonCluster(T, actual_demand, B, T_max, s_levels_predicted_7scen)

# Determine that we only want the results shown for the first 7 windows, to be able to compare the different approaches in the same plot
windows = 1:7

# Create plot for service level comparison
p13 = plot(windows, serviceLevelValuesTestPredicted_1scen, label="1 Scenario", marker=:circle, linewidth=2, color=:blue)
plot!(p13, windows, serviceLevelValuesTestMovingAverage, label="Moving Average", marker=:circle, linewidth=2, color=:pink)
plot!(p13, windows, serviceLevelValuesTestMovingAverage10, label="MA +10", marker=:utriangle, linewidth=2, color=:cyan)
plot!(p13, windows, serviceLevelValuesTestMovingAverage20, label="MA +20", marker=:dtriangle, linewidth=2, color=:purple)
plot!(p13, windows, serviceLevelValuesTestPredicted_3scen, label="3 Scenarios", marker=:utriangle, linewidth=2, color=:green)
plot!(p13, windows, serviceLevelValuesTestPredicted_5scen, label="5 Scenarios", marker=:square, linewidth=2, color=:red)
plot!(p13, windows, serviceLevelValuesTestPredicted_7scen, label="7 Scenarios", marker=:diamond, linewidth=2, color=:orange)
xlabel!(p13, "Window")
ylabel!(p13, "Service Level (%)")
title!(p13, "Service Levels for Predicted Scenarios")

# create plot for order-up-to level comparison
p14 = plot(windows, s_levels_predicted_1scen, label="1 Scenario", marker=:circle, linewidth=2, color=:blue)
plot!(p14, windows, s_levels_predicted_3scen, label="3 Scenarios", marker=:utriangle, linewidth=2, color=:green)
plot!(p14, windows, s_levels_predicted_5scen, label="5 Scenarios", marker=:square, linewidth=2, color=:red)
plot!(p14, windows, s_levels_predicted_7scen, label="7 Scenarios", marker=:diamond, linewidth=2, color=:orange)
plot!(p14, windows, s_levels_moving_average, label="Moving Average", marker=:circle, linewidth=2, color=:pink)
plot!(p14, windows, s_levels_moving_average_plus_10, label="MA +10", marker=:utriangle, linewidth=2, color=:cyan)
plot!(p14, windows, s_levels_moving_average_plus_20, label="MA +20", marker=:dtriangle, linewidth=2, color=:purple)
xlabel!(p14, "Window")
ylabel!(p14, "Order-up-to Level (s)")
title!(p14, "Order-up-to Levels for Predicted Scenarios")

# Calculate total shortage per window for all approaches
shortage_moving_average = [sum(f) for f in f_valuesTestMovingAverage]
shortage_moving_average10 = [sum(f) for f in f_valuesTestMovingAverage10]
shortage_moving_average20 = [sum(f) for f in f_valuesTestMovingAverage20]

# Calculate total shortage per window for predicted scenarios
shortage_predicted_1scen = [sum(f) for f in f_valuesTestPredicted_1scen]
shortage_predicted_3scen = [sum(f) for f in f_valuesTestPredicted_3scen]
shortage_predicted_5scen = [sum(f) for f in f_valuesTestPredicted_5scen]
shortage_predicted_7scen = [sum(f) for f in f_valuesTestPredicted_7scen]

# create plot for shortage comparison
p15 = plot(windows, shortage_predicted_1scen, label="1 Scenario", marker=:circle, linewidth=2, color=:blue)
plot!(p15, windows, shortage_predicted_3scen, label="3 Scenarios", marker=:utriangle, linewidth=2, color=:green)
plot!(p15, windows, shortage_predicted_5scen, label="5 Scenarios", marker=:square, linewidth=2, color=:red)
plot!(p15, windows, shortage_predicted_7scen, label="7 Scenarios", marker=:diamond, linewidth=2, color=:orange)
plot!(p15, windows, shortage_moving_average, label="Moving Average", marker=:circle, linewidth=2, color=:pink)
plot!(p15, windows, shortage_moving_average10, label="MA +10", marker=:utriangle, linewidth=2, color=:cyan)
plot!(p15, windows, shortage_moving_average20, label="MA +20", marker=:dtriangle, linewidth=2, color=:purple)
xlabel!(p15, "Window")
ylabel!(p15, "Total Shortage")
title!(p15, "Shortage for Predicted Scenarios")

# create plot for cost comparison
p16 = plot(windows, objectiveValuesTestPredicted_1scen, label="1 Scenario", marker=:circle, linewidth=2, color=:blue)
plot!(p16, windows, objectiveValuesTestPredicted_3scen, label="3 Scenarios", marker=:utriangle, linewidth=2, color=:green)
plot!(p16, windows, objectiveValuesTestPredicted_5scen, label="5 Scenarios", marker=:square, linewidth=2, color=:red)
plot!(p16, windows, objectiveValuesTestPredicted_7scen, label="7 Scenarios", marker=:diamond, linewidth=2, color=:orange)
plot!(p16, windows, objectiveValuesTestMovingAverage, label="Moving Average", marker=:circle, linewidth=2, color=:pink)
plot!(p16, windows, objectiveValuesTestMovingAverage10, label="MA +10", marker=:utriangle, linewidth=2, color=:cyan)
plot!(p16, windows, objectiveValuesTestMovingAverage20, label="MA +20", marker=:dtriangle, linewidth=2, color=:purple)
xlabel!(p16, "Window")
ylabel!(p16, "Total Cost")
title!(p16, "Costs")

# Calculate total wastage per window for all approaches
sum_e_valuesTestMovingAverage = [sum(e) for e in e_valuesTestMovingAverage]
sum_e_valuesTestMovingAverage10 = [sum(e) for e in e_valuesTestMovingAverage10]
sum_e_valuesTestMovingAverage20 = [sum(e) for e in e_valuesTestMovingAverage20]
sum_e_valuesTestPredicted_1scen = [sum(e) for e in e_valuesTestPredicted_1scen]
sum_e_valuesTestPredicted_3scen = [sum(e) for e in e_valuesTestPredicted_3scen]
sum_e_valuesTestPredicted_5scen = [sum(e) for e in e_valuesTestPredicted_5scen]
sum_e_valuesTestPredicted_7scen = [sum(e) for e in e_valuesTestPredicted_7scen]

# Calculate total ordering per window for all approaches
sum_q_valuesTestMovingAverage = [sum(q) for q in q_valuesTestMovingAverage]
sum_q_valuesTestMovingAverage10 = [sum(q) for q in q_valuesTestMovingAverage10]
sum_q_valuesTestMovingAverage20 = [sum(q) for q in q_valuesTestMovingAverage20]
sum_q_valuesTestPredicted_1scen = [sum(q) for q in q_valuesTestPredicted_1scen]
sum_q_valuesTestPredicted_3scen = [sum(q) for q in q_valuesTestPredicted_3scen]
sum_q_valuesTestPredicted_5scen = [sum(q) for q in q_valuesTestPredicted_5scen]
sum_q_valuesTestPredicted_7scen = [sum(q) for q in q_valuesTestPredicted_7scen]

# Create plot for wastage comparison
p17 = plot(windows, sum_e_valuesTestPredicted_1scen, label="1 Scenario", marker=:circle, linewidth=2, color=:blue)
plot!(p17, windows, sum_e_valuesTestPredicted_3scen, label="3 Scenarios", marker=:utriangle, linewidth=2, color=:green)
plot!(p17, windows, sum_e_valuesTestPredicted_5scen, label="5 Scenarios", marker=:square, linewidth=2, color=:red)
plot!(p17, windows, sum_e_valuesTestPredicted_7scen, label="7 Scenarios", marker=:diamond, linewidth=2, color=:orange)
plot!(p17, windows, sum_e_valuesTestMovingAverage, label="Moving Average", marker=:circle, linewidth=2, color=:pink)
plot!(p17, windows, sum_e_valuesTestMovingAverage10, label="MA +10", marker=:utriangle, linewidth=2, color=:cyan)
plot!(p17, windows, sum_e_valuesTestMovingAverage20, label="MA +20", marker=:dtriangle, linewidth=2, color=:purple)
xlabel!(p17, "Window")
ylabel!(p17, "Wastage")
title!(p17, "Wastage")

# Save all created plots
savefig(p13, "predicted_service_levels_all_scenarios.png")
println("Plot saved as predicted_service_levels_all_scenarios.png")
savefig(p14, "predicted_order_up_to_levels_all_scenarios.png")
println("Plot saved as predicted_order_up_to_levels_all_scenarios.png")
savefig(p15, "predicted_shortage_all_scenarios.png")
println("Plot saved as predicted_shortage_all_scenarios.png")
savefig(p16, "predicted_costs_all_scenarios.png")
println("Plot saved as predicted_costs_all_scenarios.png")
savefig(p17, "predicted_wastage_all_scenarios.png")
println("Plot saved as predicted_wastage_all_scenarios.png")


# create separate plots for only the predicted scenarios, to be able to compare the different number of scenarios without the moving average approaches in the same plot
p8 = plot(windows, serviceLevelValuesTestPredicted_1scen, label="1 Scenario", marker=:circle, linewidth=2, color=:blue)
plot!(p8, windows, serviceLevelValuesTestPredicted_3scen, label="3 Scenarios", marker=:utriangle, linewidth=2, color=:green)
plot!(p8, windows, serviceLevelValuesTestPredicted_5scen, label="5 Scenarios", marker=:square, linewidth=2, color=:red)
plot!(p8, windows, serviceLevelValuesTestPredicted_7scen, label="7 Scenarios", marker=:diamond, linewidth=2, color=:orange)
xlabel!(p8, "Window")
ylabel!(p8, "Service Level (%)")
title!(p8, "Service Levels for Predicted Scenarios")

p9 = plot(windows, s_levels_predicted_1scen, label="1 Scenario", marker=:circle, linewidth=2, color=:blue)
plot!(p9, windows, s_levels_predicted_3scen, label="3 Scenarios", marker=:utriangle, linewidth=2, color=:green)
plot!(p9, windows, s_levels_predicted_5scen, label="5 Scenarios", marker=:square, linewidth=2, color=:red)
plot!(p9, windows, s_levels_predicted_7scen, label="7 Scenarios", marker=:diamond, linewidth=2, color=:orange)
xlabel!(p9, "Window")
ylabel!(p9, "Order-up-to Level (s)")
title!(p9, "Order-up-to Levels for Predicted Scenarios")

p10 = plot(windows, shortage_predicted_1scen, label="1 Scenario", marker=:circle, linewidth=2, color=:blue)
plot!(p10, windows, shortage_predicted_3scen, label="3 Scenarios", marker=:utriangle, linewidth=2, color=:green)
plot!(p10, windows, shortage_predicted_5scen, label="5 Scenarios", marker=:square, linewidth=2, color=:red)
plot!(p10, windows, shortage_predicted_7scen, label="7 Scenarios", marker=:diamond, linewidth=2, color=:orange)
xlabel!(p10, "Window")
ylabel!(p10, "Total Shortage")
title!(p10, "Shortage for Predicted Scenarios")

p11 = plot(windows, objectiveValuesTestPredicted_1scen, label="1 Scenario", marker=:circle, linewidth=2, color=:blue)
plot!(p11, windows, objectiveValuesTestPredicted_3scen, label="3 Scenarios", marker=:utriangle, linewidth=2, color=:green)
plot!(p11, windows, objectiveValuesTestPredicted_5scen, label="5 Scenarios", marker=:square, linewidth=2, color=:red)
plot!(p11, windows, objectiveValuesTestPredicted_7scen, label="7 Scenarios", marker=:diamond, linewidth=2, color=:orange)
xlabel!(p11, "Window")
ylabel!(p11, "Total Cost")
title!(p11, "Costs")

p12 = plot(windows, sum_e_valuesTestPredicted_1scen, label="1 Scenario", marker=:circle, linewidth=2, color=:blue)
plot!(p12, windows, sum_e_valuesTestPredicted_3scen, label="3 Scenarios", marker=:utriangle, linewidth=2, color=:green)
plot!(p12, windows, sum_e_valuesTestPredicted_5scen, label="5 Scenarios", marker=:square, linewidth=2, color=:red)
plot!(p12, windows, sum_e_valuesTestPredicted_7scen, label="7 Scenarios", marker=:diamond, linewidth=2, color=:orange)
xlabel!(p12, "Window")
ylabel!(p12, "Wastage")
title!(p12, "Wastage")

savefig(p8, "predicted_service_levels.png")
println("Plot saved as predicted_service_levels.png")
savefig(p9, "predicted_order_up_to_levels.png")
println("Plot saved as predicted_order_up_to_levels.png")
savefig(p10, "predicted_shortage.png")
println("Plot saved as predicted_shortage.png")
savefig(p11, "predicted_costs.png")
println("Plot saved as predicted_costs.png")
savefig(p12, "predicted_wastage.png")
println("Plot saved as predicted_wastage.png")

# Create summary statistics for all approaches and metrics, to be able to compare the different approaches in a table in the thesis
println("\nSummary statistics for all-model comparison plots:")
for (label, values) in [
        ("Service Level - Moving Average", serviceLevelValuesTestMovingAverage),
        ("Service Level - MA +10", serviceLevelValuesTestMovingAverage10),
        ("Service Level - MA +20", serviceLevelValuesTestMovingAverage20),
        ("Service Level - Predicted 1scen", serviceLevelValuesTestPredicted_1scen),
        ("Service Level - Predicted 3scen", serviceLevelValuesTestPredicted_3scen),
        ("Service Level - Predicted 5scen", serviceLevelValuesTestPredicted_5scen),
        ("Service Level - Predicted 7scen", serviceLevelValuesTestPredicted_7scen),
        ("Order-up-to Level - Moving Average", s_levels_moving_average),
        ("Order-up-to Level - MA +10", s_levels_moving_average_plus_10),
        ("Order-up-to Level - MA +20", s_levels_moving_average_plus_20),
        ("Order-up-to Level - Predicted 1scen", s_levels_predicted_1scen),
        ("Order-up-to Level - Predicted 3scen", s_levels_predicted_3scen),
        ("Order-up-to Level - Predicted 5scen", s_levels_predicted_5scen),
        ("Order-up-to Level - Predicted 7scen", s_levels_predicted_7scen),
        ("Shortage - Moving Average", shortage_moving_average),
        ("Shortage - MA +10", shortage_moving_average10),
        ("Shortage - MA +20", shortage_moving_average20),
        ("Shortage - Predicted 1scen", shortage_predicted_1scen),
        ("Shortage - Predicted 3scen", shortage_predicted_3scen),
        ("Shortage - Predicted 5scen", shortage_predicted_5scen),
        ("Shortage - Predicted 7scen", shortage_predicted_7scen),
        ("Cost - Moving Average", objectiveValuesTestMovingAverage),
        ("Cost - MA +10", objectiveValuesTestMovingAverage10),
        ("Cost - MA +20", objectiveValuesTestMovingAverage20),
        ("Cost - Predicted 1scen", objectiveValuesTestPredicted_1scen),
        ("Cost - Predicted 3scen", objectiveValuesTestPredicted_3scen),
        ("Cost - Predicted 5scen", objectiveValuesTestPredicted_5scen),
        ("Cost - Predicted 7scen", objectiveValuesTestPredicted_7scen),
        ("Wastage - Moving Average", sum_e_valuesTestMovingAverage),
        ("Wastage - MA +10", sum_e_valuesTestMovingAverage10),
        ("Wastage - MA +20", sum_e_valuesTestMovingAverage20),
        ("Wastage - Predicted 1scen", sum_e_valuesTestPredicted_1scen),
        ("Wastage - Predicted 3scen", sum_e_valuesTestPredicted_3scen),
        ("Wastage - Predicted 5scen", sum_e_valuesTestPredicted_5scen),
        ("Wastage - Predicted 7scen", sum_e_valuesTestPredicted_7scen),
        ("Ordering - Moving Average", sum_q_valuesTestMovingAverage),
        ("Ordering - MA +10", sum_q_valuesTestMovingAverage10),
        ("Ordering - MA +20", sum_q_valuesTestMovingAverage20),
        ("Ordering - Predicted 1scen", sum_q_valuesTestPredicted_1scen),
        ("Ordering - Predicted 3scen", sum_q_valuesTestPredicted_3scen),
        ("Ordering - Predicted 5scen", sum_q_valuesTestPredicted_5scen),
        ("Ordering - Predicted 7scen", sum_q_valuesTestPredicted_7scen),
    ]
    println("$label: mean=$(round(mean(values), digits=3)), std=$(round(std(values), digits=3))")
end
