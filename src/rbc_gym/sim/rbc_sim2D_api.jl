using Oceananigans
using Logging
using Statistics


include("rbc_sim2D.jl")

# Global variables to hold simulation state
global simulation = nothing
global model = nothing
global step = 1
global time = 0.0

"""
Initialize a Rayleigh-Bénard simulation with the given parameters
"""
function initialize_simulation(; Ra=10^5, sensors=[48, 8], grid=[96, 64], heaters=12, heater_limit=0.75, dt=1, seed=42, checkpoint_path=nothing, use_gpu=false)
    oceananigans_logger = Oceananigans.Logger.OceananigansLogger(
        stdout,
        Logging.Warn;
        show_info_source=true
    )
    global_logger(oceananigans_logger)

    # Setup simulation parameters
    global N = grid
    global N_obs = sensors
    global L = [2 * pi, 2]
    global domain = L
    global Δb = 1
    global actuators = heaters
    global actuator_limit = heater_limit
    global Δt = dt

    Pr = 0.7
    min_b = 1
    random_kick = 0.01
    Δt_solver = 0.03

    ν = sqrt(Pr / Ra)
    κ = 1 / sqrt(Pr * Ra)

    # Set random seed for reproducibility
    Random.seed!(seed)

    # Initialize action
    global action = zeros(actuators)

    # Initialize simulation components
    grid = define_sample_grid(N, L, use_gpu)
    u_bcs, b_bcs = define_boundary_conditions(min_b, Δb)

    # Create model
    global model = define_model(grid, ν, κ, u_bcs, b_bcs)
    # Initialize model
    if isnothing(checkpoint_path)
        initialize_model(model, min_b, L[2], Δb, random_kick)
    else
        initialize_from_checkpoint(model, checkpoint_path)
    end

    # Setup simulation
    global simulation = Simulation(model, Δt=Δt_solver, stop_time=Δt)
    simulation.verbose = false

    # Reset state
    global step = 1
    global time = 0.0

end

"""
Step the simulation forward by one timestep
"""
function step_simulation(actuation)
    global simulation, model, Δt, N, step, time
    global action = actuation

    if simulation === nothing
        error("Simulation not initialized. Call initialize_simulation first.")
    end

    # Run for one step
    run!(simulation)
    simulation.stop_time += Δt

    time += Δt
    step += 1

    # Check for NaNs
    contains_nans = step_contains_NaNs(model, N)
    if contains_nans
        return false
    end

    return true
end

"""
Get the current state of the simulation
"""
function get_state()
    global model, N

    if model === nothing
        error("Simulation not initialized. Call initialize_simulation first.")
    end

    # Extract fields from the current model
    state = zeros(3, N[1], N[2])
    state[1, :, :] = model.tracers.b[1:N[1], 1, 1:N[2]]
    state[2, :, :] = model.velocities.u[1:N[1], 1, 1:N[2]]
    state[3, :, :] = model.velocities.w[1:N[1], 1, 1:N[2]]

    return state
end

"""
Get the current state of the simulation
"""
function get_observation()
    global N_obs, N

    sensor_positions = [collect(1:Int(N[1] / N_obs[1]):N[1]), collect(1:Int(N[2] / N_obs[2]):N[2])] #TODO einmal oben definieren
    state = get_state()
    return state[:, sensor_positions[1], sensor_positions[2]]
end

"""
Get the current simulation time
"""
function get_info()
    global time, step
    return (time, step)
end

"""
Get nusselt number
"""
function get_nusselt(;state=false)
    global model, L, Δb

    # get data; either state or observation
    if state
        data = get_state()
    else
        data = get_observation()
    end

    T = data[1, :, :]     # temperature field (b‑tracer)
    uy = data[3, :, :]    # vertical velocity

    kappa = model.closure.κ[1]  # thermal diffusivity κ
    H = L[2]                    # domain height

    q_1_mean = mean(T .* uy)
    Tx = mean(T', dims=2)
    q_2 = kappa * mean(array_gradient(Tx))

    return (q_1_mean - q_2) / (kappa * Δb / H)
end


# Export functions
export initialize_simulation, step_simulation, get_state
