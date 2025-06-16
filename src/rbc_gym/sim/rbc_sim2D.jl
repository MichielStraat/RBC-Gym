using Logging
import Random
using Oceananigans
using Statistics
using HDF5
using CUDA
using ArgParse
using ProgressMeter

# script directory
dirpath = string(@__DIR__)


function simulate_2d_rb(dir, seed, random_inits, Ra, Pr, N, L, min_b, Δb, random_kick, Δt, Δt_snap,
    duration, use_gpu)

    ν = sqrt(Pr / Ra) # c.f. line 33: https://github.com/spectralDNS/shenfun/blob/master/demo/RayleighBenard2D.py
    κ = 1 / sqrt(Pr * Ra) # c.f. line 37: https://github.com/spectralDNS/shenfun/blob/master/demo/RayleighBenard2D.py

    totalsteps = Int(div(duration, Δt_snap))

    global domain, action, actuators, actuator_limit
    domain = L
    actuators = 12
    action = zeros(actuators)
    actuator_limit = 0.75

    grid = define_sample_grid(N, L, use_gpu)
    u_bcs, b_bcs = define_boundary_conditions(min_b, Δb)

    model = define_model(grid, ν, κ, u_bcs, b_bcs)

    if !isdir(dir)
        mkpath(dir)
    end
    path = joinpath(dir, "ckpt_ra$(Ra).h5")
    h5_file = h5open(path, "w")

    attrs(h5_file)["num_episodes"] = random_inits
    attrs(h5_file)["start_seed"] = seed
    db = create_dataset(h5_file, "b", datatype(Float64), dataspace(random_inits, size(model.tracers.b)...))
    du = create_dataset(h5_file, "u", datatype(Float64), dataspace(random_inits, size(model.velocities.u)...))
    dw = create_dataset(h5_file, "w", datatype(Float64), dataspace(random_inits, size(model.velocities.w)...))

    for i ∈ 1:random_inits
        println("Simulating random initialization $(i)/$(random_inits)...")

        # Make sure that every random initialization is indeed independend of each other
        # (even when script is restarted)
        Random.seed!(seed + i)

        model = define_model(grid, ν, κ, u_bcs, b_bcs)
        initialize_model(model, min_b, L[2], Δb, random_kick)

        simulation = Simulation(model, Δt=Δt, stop_time=Δt_snap)
        simulation.verbose = false

        success = simulate_model(simulation, model, Δt, Δt_snap, totalsteps, N)

        if (!success)
            return
        end

        db[i, :, :, :] = model.tracers.b
        du[i, :, :, :] = model.velocities.u
        dw[i, :, :, :] = model.velocities.w
    end

    # Save the simulation results to a file
    close(h5_file)
    println("Saved data to: ", h5_file)
end


function define_sample_grid(N, L, use_gpu)
    if use_gpu
        grid = RectilinearGrid(GPU(), size=N, x=(0, L[1]), z=(0, L[2]),
            topology=(Periodic, Flat, Bounded))
    else
        grid = RectilinearGrid(size=(N), x=(0, L[1]), z=(0, L[2]),
            topology=(Periodic, Flat, Bounded))
    end
    return grid
end


function collate_actions_colin(action, x, t)
    global domain, actuators, actuator_limit

    ampl = actuator_limit
    dx = 0.03

    values = ampl .* action
    Mean = mean(values)
    K2 = maximum([1.0, maximum(abs.(values .- Mean)) / ampl])

    segment_length = domain[1] / actuators

    # determine segment of x
    x_segment = Int(floor(x / segment_length) + 1)

    if x_segment == 1
        T0 = 2 + (ampl * action[end] - Mean) / K2
    else
        T0 = 2 + (ampl * action[x_segment-1] - Mean) / K2
    end

    T1 = 2 + (ampl * action[x_segment] - Mean) / K2

    if x_segment == actuators
        T2 = 2 + (ampl * action[1] - Mean) / K2
    else
        T2 = 2 + (ampl * action[x_segment+1] - Mean) / K2
    end

    # x position in the segment
    x_pos = x - (x_segment - 1) * segment_length

    # determine if x is in the transition regions
    if x_pos < dx
        #transition region left
        return T0 + ((T0 - T1) / (4 * dx^3)) * (x_pos - 2 * dx) * (x_pos + dx)^2

    elseif x_pos >= segment_length - dx
        #transition region right
        return T1 + ((T1 - T2) / (4 * dx^3)) * (x_pos - segment_length - 2 * dx) * (x_pos - segment_length + dx)^2

    else
        # middle of the segment
        return T1

    end
end

function bottom_T(x, t)
    global action
    collate_actions_colin(action, x, t)
end

function define_boundary_conditions(min_b, Δb)
    u_bcs = FieldBoundaryConditions(top=ValueBoundaryCondition(0),
        bottom=ValueBoundaryCondition(0))
    b_bcs = FieldBoundaryConditions(top=ValueBoundaryCondition(min_b),
        bottom=ValueBoundaryCondition(bottom_T))
    return u_bcs, b_bcs
end


function define_model(grid, ν, κ, u_bcs, b_bcs)
    model = NonhydrostaticModel(; grid,
        advection=UpwindBiasedFifthOrder(),
        timestepper=:RungeKutta3,
        tracers=(:b),
        buoyancy=Buoyancy(model=BuoyancyTracer()),
        closure=(ScalarDiffusivity(ν=ν, κ=κ)),
        boundary_conditions=(u=u_bcs, b=b_bcs,),
        coriolis=nothing
    )
    return model
end


function initialize_model(model, min_b, Lz, Δb, kick)
    # Set initial conditions
    uᵢ(x, z) = kick * randn()
    wᵢ(x, z) = kick * randn()
    bᵢ(x, z) = clamp(min_b + (Lz - z) * Δb / 2 + kick * randn(), min_b, min_b + Δb)

    # Send the initial conditions to the model to initialize the variables
    set!(model, u=uᵢ, w=wᵢ, b=bᵢ)
end

function initialize_from_checkpoint(model, path)
    h5_file = h5open(path, "r")

    n = attrs(h5_file)["num_episodes"]
    idx = rand(1:n)

    bb = read(h5_file, "b")[idx,:,:,:]
    uu = read(h5_file, "u")[idx,:,:,:]
    ww = read(h5_file, "w")[idx,:,:,:]

    println("Loading checkpoint with index: $idx from file: $path")
    set!(model, u = uu, w = ww, b = bb)
    close(h5_file)
end


function simulate_model(simulation, model, Δt, Δt_snap, totalsteps, N)
    cur_time = 0.0
    @showprogress 2 "Simulating..." for i in 1:totalsteps
        run!(simulation)
        simulation.stop_time += Δt_snap
        cur_time += Δt_snap

        if (step_contains_NaNs(model, N))
            printstyled("[ERROR] NaN values found!\n"; color=:red)
            return false
        end
    end

    return true
end


function array_gradient(a)
    result = zeros(length(a))

    for i in 1:length(a)
        if i == 1
            result[i] = a[i+1] - a[i]
        elseif i == length(a)
            result[i] = a[i] - a[i-1]
        else
            result[i] = (a[i+1] - a[i-1]) / 2
        end
    end

    result
end


function step_contains_NaNs(model, N)
    contains_nans = (any(isnan, model.tracers.b[1:N[1], 1, 1:N[2]]) ||
                     any(isnan, model.velocities.u[1:N[1], 1, 1:N[2]]) ||
                     any(isnan, model.velocities.w[1:N[1], 1, 1:N[2]]))
    return contains_nans
end


function parse_arguments()
    s = ArgParseSettings(description="Simulates 2D Rayleigh-Bénard.")

    @add_arg_table s begin
        "--sim_name"
        help = "The name of the simulation. Defaults to the following scheme \
        x<N[1]>_z<N[2]>_Ra<Ra>_Pr<Pr>_t<Δt>_snap<Δt_snap>_dur<duration>"
        default = nothing
        "--dir"
        help = "The path to the directory to store the simulations in."
        default = joinpath(dirpath, "data")
        "--seed"
        help = "Random seed for the simulation."
        arg_type = Int
        default = 42
        "--random_inits"
        help = "The number of random initializations to simulate"
        arg_type = Int
        default = 1
        "--Ra"
        help = "Rayleigh number"
        arg_type = Int
        default = 10^5
        "--Pr"
        help = "Prandtl number"
        arg_type = Float64
        default = 0.7
        "--N"
        help = "The size of the grid [width, height]"
        arg_type = Int
        nargs = 2
        default = [128, 64]
        "--L"
        help = "The spatial dimensions of the domain [width, height]"
        arg_type = Float64
        nargs = 2
        default = [2 * pi, 2]
        "--min_b"
        help = "The temperature of the top plate"
        arg_type = Float64
        default = 0
        "--delta_b"
        help = "The temperature difference between the bottom and top plate"
        arg_type = Float64
        default = 1
        dest_name = "Δb"
        "--random_kick"
        help = "The amplitude of the random initial perturbation (kick)"
        arg_type = Float64
        default = 0.2
        "--delta_t"
        help = "Time delta between simulated steps"
        arg_type = Float64
        default = 0.01
        dest_name = "Δt"
        "--delta_t_snap"
        help = "Time delta between saved snapshots"
        arg_type = Float64
        default = 0.3
        dest_name = "Δt_snap"
        "--duration"
        help = "The duration of a simulation"
        arg_type = Int
        default = 300
        "--use_cpu"
        help = "Runs the simulation on CPU when argument given."
        action = :store_false
        dest_name = "use_gpu"
    end

    return parse_args(s)
end

is_script = abspath(PROGRAM_FILE) == @__FILE__
if (is_script)
    args = parse_arguments()
    simulate_2d_rb(
        args["dir"],
        args["seed"],
        args["random_inits"],
        args["Ra"],
        args["Pr"],
        args["N"],
        args["L"],
        args["min_b"],
        args["Δb"],
        args["random_kick"],
        args["Δt"],
        args["Δt_snap"],
        args["duration"],
        args["use_gpu"],)
end