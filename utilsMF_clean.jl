using Polynomials
using LinearAlgebra
using Plots
using Plots.PlotMeasures
using Distributions, ProgressMeter, LaTeXStrings

# Define different parameters
using Parameters

# STDP Network structure with default parameters
@with_kw struct STDPNetwork_lim
       N                :: Int64   = 2000    # number of neurons
       alpha_m          :: Float64 = 0.05   # minimum firing rate of neurons from 0 to 1
       alpha_M          :: Float64 = 1.     # maximum firing rate of neurons from 0 to 1
       beta             :: Float64 = 1.     # firing rate of neurons from 1 to 0 (mean time 2ms)
       delta_w          :: Int64   = 1      # size of weight jumps
       taupos           :: Float64 = 1.5    # time constant for potentiation (see STDP curve)
       tauneg           :: Float64 = 2.     # time constant for depression (see STDP curve)
       p0_plus          :: Float64 = .8     # base rate for potentiation
       p0_moins         :: Float64 = .6     # base rate for depression
       dt               :: Float64 = .05    # time step and grid size for xi(s)
       tmax             :: Float64 = 5.     # final time (ms)
       tmin             :: Float64 = 0.0    # starting time (ms)
       sigma            :: Float64 = 1.5    # slope of the sigmoid
       theta            :: Float64 = 0.     # threshold theta such that f(2*theta) = alpha_M
       w_max            :: Int64   = 10     # maximum possible weight
       w_min            :: Int64   = -10    # minimum possible weight
       W_init           :: Int64   = 0      # initial value of weights: uniform({-W_init,...,0,...,W_init})
       p_V_init         :: Float64 = 0.213  # at t=0, neurons potentials follow Bernoulli(p_V_init)
       lambda           :: Float64 = 2.     # parameter of the exponential law of S_0
       m_s              :: Int64   = 15     # maximum value of s in xi
       M_s              :: Int64   = Int(floor(m_s / dt)) # xi(s) where s belongs to [0:dt:dt*M_s]
       M_s_max          :: Int64   = Int(floor(M_s))  # initially s belongs to [0:dt:dt*M_s_max]
end

# Neuron firing rate function for an input current equal to x
# Uses a sigmoid function to map input current to firing rate
f(x, n::STDPNetwork_lim) =  @. (n.alpha_M - n.alpha_m) / (1 + exp(-n.sigma*(x - n.theta))) + n.alpha_m

# Probability of potentiation and depression independent of weights
# These functions decay exponentially with time since last spike
p_plus(s, taupos, p0_plus) = @. exp(-s/taupos) * p0_plus
p_moins(s, tauneg, p0_moins) = @. exp(-s/tauneg) * p0_moins

# Calculate the input current for neuron i based on xi_1_t distribution
function currentI(i, xi_1_t, w_min, w_max)
    @views dot(w_min:1:w_max, sum(xi_1_t[i,:,:], dims = 1))
end

# Initialize the system with initial conditions
#Gamma(2.5, 2.)    LogNormal(0.8, 1)
function cond_initiales(net::STDPNetwork_lim, d = Gamma(2.5, 2.))
    @unpack p_V_init, M_s, w_max, w_min, N, W_init, M_s_max, dt = net

    # Draw initial neuron states V_0 from Bernoulli distribution
    V_0 = rand(N)
    V_t = V_0 .< p_V_init
    V_t_part = V_0 .< p_V_init

    # Draw initial spike times S_0
    S_0 = rand(N)
    S_t_part = Vector{Float64}(undef, N)

    # Initialize S_t using Gamma distribution
    S_t = rand(d, N)
    S_t_part[findall(V_t .== 0)] .= S_t[findall(V_t .== 0)]

    # Initialize xi_0 and xi_1 distributions
    xi_0_t = zeros(N, M_s, w_max - w_min + 1)
    xi_1_t = zeros(N, M_s, w_max - w_min + 1)

    # Weight distribution vector (uniform on [-W_init,..., W_init])
    W_0 = zeros(2*w_max + 1)
    W_0[w_max - W_init + 1: w_max + W_init + 1] .= 1 / (2*W_init + 1)

    # Initialize xi_0: zero at S=0 and with density on [dt, M_s_max * dt]
    densite(x) = Distributions.pdf(d, x)
    # Normalize by truncating the density at M_s_max
    norm_uniform_0 = sum(dt*densite(k * dt) for k in 1:M_s_max-1)
    for i = 1:N
        xi_0_t[i,2:M_s_max,:] .= densite.(collect(1:M_s_max-1) * dt ) * W_0' / norm_uniform_0
    end
    @error "Check xi_0_t normalization" sum(xi_0_t[1,:,:]) * dt

    # For the calculation of xi_1(0)
    I_t = ones(N,1) * dot((w_min:w_max), W_0)
    m_etoile = Int.(floor.(S_t .* (1 .- V_t) ./ dt)) .+ 1
    m_etoile[m_etoile .> M_s] .= M_s
    count_m_etoile = zeros(Int64, M_s)
    for k=1:N
        count_m_etoile[m_etoile[k]] += 1
    end

    # Calculate xi_1(0)
    for j in axes(xi_1_t, 3), i in axes(xi_1_t, 1)
        xi_1_t[i,1,j] = dt * sum(f(I_t[k], net) * xi_0_t[i, m_etoile[k], j] / count_m_etoile[m_etoile[k]] for k in 1:N)
    end

    # Calculate full xi_1: a distribution with the remaining mass
    xi_1_0 = sum(xi_1_t[1,1,:])

    # Use exponential law with parameter xi_1_0
    norm_uniform_1 = sum(dt*xi_1_0*exp.(- collect(0:M_s_max-1) * dt * xi_1_0))
    for i = 1:N
        xi_1_t[i,1:M_s_max,:] .=  xi_1_0 * exp.(- collect(0:M_s_max-1) * dt * xi_1_0) * W_0' / norm_uniform_1
    end
    @error "Check xi_1_t normalization" sum(xi_1_t[1,:,:]) * dt

    # Draw remaining S_0 (those for which V_0=1) according to xi_1_0 law
    S_1 = rand(sum(V_t))
    S_t[findall(V_t .== 1)]         .= -log.(S_1) ./ xi_1_0
    S_t_part[findall(V_t .== 1)]    .= -log.(S_1) ./ xi_1_0

    # Draw weights of the particle system uniformly on [-W_init,W_init]
    @error "I think W_t_part = zeros(N,N)"
    W_cdt_init = rand(N^2)
    W_cdt_init .= Int.(floor.((2 .* W_init .+ 1) .* W_cdt_init .- W_init ))
    W_cdt_init = Float64.(W_cdt_init)
    W_t_part   = reshape(W_cdt_init,N,N)

    # Final adjustment of weights
    p_V_emp = sum(V_t)/N
    xi_0_t .= (1 - p_V_emp) * xi_0_t
    xi_1_t .= p_V_emp * xi_1_t
    @error "Check total probability mass" sum(xi_0_t[1,:,:]) * dt + sum(xi_1_t[1,:,:]) * dt

    return (;V_t,S_t,xi_0_t,xi_1_t,V_t_part,S_t_part,W_t_part,xi_1_0)
end

# Drift evolution of xi distributions (mean-field approximation)
function drift_xi!(net::STDPNetwork_lim, S_t, xi_0_t, xi_1_t, copy_xi_0_t, copy_xi_1_t, a_t_0, I_t)
    @unpack_STDPNetwork_lim net

    # Boundary term at S = M_s dt
    @. @views xi_0_t[:,M_s,:] = copy_xi_0_t[:,M_s-1,:] * (1 - dt * a_t_0[M_s-1]) +
            copy_xi_0_t[:,M_s,:] * (1 - dt * a_t_0[M_s]) +
            (copy_xi_1_t[:,M_s,:] + copy_xi_1_t[:,M_s-1,:]) * dt * beta

    @. @views xi_1_t[:, M_s, :] = (copy_xi_1_t[:,M_s,:] + copy_xi_1_t[:,M_s - 1,:]) * (1 - dt * beta)

    # Term for all 0 < S < M_s dt
    @. @views xi_1_t[:,2:M_s-1,:] = copy_xi_1_t[:,1:M_s-2,:] * (1 - dt * beta)

    @. @views xi_0_t[:,2:M_s-1,:] = copy_xi_0_t[:,1:M_s-2,:] * (1 - dt * a_t_0[1:M_s-2])' +
            copy_xi_1_t[:,1:M_s-2,:] * dt * beta

    # Boundary term at S = 0
    xi_0_t[:,1,:] .= 0.

    # Update xi_1 at S=0 with depression and potentiation effects
    @views xi_1_t[:,1,1:w_max - w_min] .= sum(
            (p_moins(S_t, tauneg, p0_moins) .* copy_xi_0_t[:,:,2:w_max - w_min + 1] .+
            (1 .- p_moins(S_t, tauneg, p0_moins)) .* copy_xi_0_t[:,:,1:w_max - w_min]) .* a_t_0',
            dims = 2 )[:,1,:]

    @views xi_1_t[:,1,w_max - w_min + 1] .= sum( a_t_0' .*
            (1 .- p_moins(S_t, tauneg, p0_moins)) .* copy_xi_0_t[:,:, w_max - w_min+1],
            dims = 2)[:]

    @views xi_1_t[:,1,1] .= xi_1_t[:,1,1] .+ sum( a_t_0' .*
            p_moins(S_t, tauneg, p0_moins) .* copy_xi_0_t[:,:, 1],
            dims = 2)[:]

    @. @views xi_1_t[:,1,:] = dt * xi_1_t[:,1,:]
end

# Handle jump terms when neurons spike
function jump_terms!(net::STDPNetwork_lim, ind_saut_neur, tau_t, V_t, S_t, xi_0_t, xi_1_t)
    @unpack_STDPNetwork_lim net

    # Update neuron states and spike times
    @. @views V_t = (V_t - ind_saut_neur)^2
    @. @views S_t = (S_t + dt) * (1 - ind_saut_neur * V_t) + (dt - tau_t) * ind_saut_neur * V_t

    # Get indices of spiking neurons
    ind_spike = getindex.(findall( ind_saut_neur .* V_t .== 1),1)

    # Update xi_0 distribution for spiking neurons with potentiation effects
    @. @views xi_0_t[ind_spike,:,w_max - w_min + 1 ] = xi_0_t[ind_spike,:,w_max - w_min + 1 ] +
            p_plus( 0:dt:m_s-dt, taupos, p0_plus)' * xi_0_t[ind_spike,:,w_max - w_min]

    @. @views xi_0_t[ind_spike,:,2:w_max - w_min ] =
        p_plus(0:dt:m_s-dt, taupos, p0_plus)' * xi_0_t[ind_spike,:,1:w_max - w_min - 1] +
            (1 - p_plus(0:dt:m_s-dt, taupos, p0_plus))' * xi_0_t[ind_spike,:,2:w_max - w_min ]

    @. @views xi_0_t[ind_spike,:, 1 ] = (1 - p_plus(0:dt:m_s-dt, taupos, p0_plus)') * xi_0_t[ind_spike,:,1]

    # Update xi_1 distribution for spiking neurons with potentiation effects
    @. @views xi_1_t[ind_spike,:,w_max - w_min + 1 ] = xi_1_t[ind_spike,:,w_max - w_min + 1 ] +
        p_plus(dt/2:dt:m_s-dt/2, taupos, p0_plus)' * xi_1_t[ind_spike,:,w_max - w_min]

    @. @views xi_1_t[ind_spike,:,2:w_max - w_min ] =
        p_plus(dt/2:dt:m_s-dt/2, taupos, p0_plus)' * xi_1_t[ind_spike,:,1:w_max - w_min - 1] +
            (1 - p_plus(dt/2:dt:m_s-dt/2, taupos, p0_plus))' * xi_1_t[ind_spike,:,2:w_max - w_min ]

    @. @views xi_1_t[ind_spike,:,1 ] = (1 - p_plus(dt/2:dt:m_s-dt/2, taupos, p0_plus))' * xi_1_t[ind_spike,:,1]
end

# Update particle system (discrete simulation)
function update_part!(net::STDPNetwork_lim,V_t_part,S_t_part,W_t_part,I_t_part)
    @unpack_STDPNetwork_lim net

    # Calculate jump rates for particles
    taux_saut_part = f(I_t_part, net) .* (1 .- V_t_part) .+ beta .* V_t_part
    tau_t_part = -log.(rand(N))./taux_saut_part
    ind_saut_neur_part = tau_t_part .< dt

    # Update neuron states
    @. V_t_part = (V_t_part - ind_saut_neur_part)^2
    ind_spike_part = getindex.(findall( ind_saut_neur_part .* V_t_part .== 1),1)
    n_spike = length(ind_spike_part)

    # Update weights based on spike events with potentiation/depression
    bern_plus = rand(n_spike * N)
    bern_plus = reshape(bern_plus, n_spike, N)
    bern_moins = rand(n_spike * N)
    bern_moins = reshape(bern_moins, N, n_spike)

    # Potentiation: increase weights for outgoing connections from spiking neurons
    @views W_t_part[ind_spike_part,:] .= W_t_part[ind_spike_part,:] .+
        (W_t_part[ind_spike_part,:] .< w_max ) .*
        (bern_plus .< ones(n_spike,1) * p_plus(S_t_part, taupos, p0_plus)')

    # Depression: decrease weights for incoming connections to spiking neurons
    @views W_t_part[:,ind_spike_part] .= W_t_part[:,ind_spike_part] .-
            (W_t_part[:,ind_spike_part] .> w_min ) .*
            (bern_moins .< p_moins(S_t_part, tauneg, p0_moins) * ones(1,n_spike))

    # Update spike times
    @. @views S_t_part = (S_t_part + dt) * (1 - ind_saut_neur_part * V_t_part) +
                        (dt - tau_t_part) * ind_saut_neur_part * V_t_part
end

# Main simulation function for mean-field approximation
function simu_traj_MF!(net::STDPNetwork_lim, initialCondition; verbose = true)
    # Initialization - extract necessary parameters (automatic)
    @unpack_STDPNetwork_lim net
    @unpack V_t, S_t, xi_0_t, xi_1_t = initialCondition
    ii_max          = Int.(floor.((tmax - tmin)/dt))

    # Initialize arrays for storing results
    moy_taux_saut   = Vector{Float64}(undef, ii_max)
    Esp_V           = Vector{Float64}(undef, ii_max)
    Esp_S           = Vector{Float64}(undef, ii_max)
    Esp_V_xi        = Vector{Float64}(undef, ii_max)
    Esp_S_xi        = Vector{Float64}(undef, ii_max)
    Esp_W_xi        = Vector{Float64}(undef, ii_max)
    times           = Vector{Float64}(undef, ii_max)
    n_jump          = Vector{Float64}(undef, ii_max)
    m_etoile        = Vector{Int64}(undef, N)
    I_t             = Vector{Float64}(undef, N)
    a_t_0           = zeros(Float64,M_s)
    count_m_etoile  = zeros(Float64,M_s)
    t_current       = 0.

	# Temporary variables for calculation
    copy_xi_1_t = copy(xi_1_t)
    copy_xi_0_t = copy(xi_0_t)
    copy_a_t_0 = copy(a_t_0)

    # Time loop
    @time @showprogress for ii = 1:ii_max
        # Calculate input intensity I for each neuron (previous time step)
        for i=1:N
            @views I_t[i] = dt * currentI(i, xi_1_t, w_min, w_max)
        end

        # Record quantities for all time
        moy_taux_saut[ii]   = sum(f(I_t,net) .* (1 .- V_t) .+ beta .* V_t) / N
        Esp_V[ii]           = sum(V_t) / N
        Esp_S[ii]           = sum(S_t) / N
        Esp_V_xi[ii]        = sum(xi_1_t) * dt / N
        Esp_S_xi[ii]        = dot(collect(dt/2:dt:m_s-dt/2),sum(sum(xi_1_t + xi_0_t, dims = 1), dims = 3)) * dt / N
        Esp_W_xi[ii]        = dot( collect(w_min:w_max), ( sum(sum(xi_1_t + xi_0_t, dims = 1),dims = 2) ) ) * dt / N
        times[ii]           = t_current

        # Calculate quantities for next time step
        # Calculate m_etoile, and retrieve indices where m_etoile[k] > 0
        @. m_etoile = Int(floor(S_t * (1 - V_t) / dt)) + 1
        m_etoile[m_etoile .> M_s] .= M_s

        # Calculate a_t^0 (firing rate approximation)
        a_t_0 .= 0.
        count_m_etoile .= 0.
        for k=1:N
            a_t_0[m_etoile[k]] += f(I_t[k], net)
            count_m_etoile[m_etoile[k]] += 1
        end
        ind_m_non_0 = findall(count_m_etoile .> 0)
        @. a_t_0[ind_m_non_0] = a_t_0[ind_m_non_0] / count_m_etoile[ind_m_non_0]
        copyto!(copy_a_t_0, a_t_0)

        # Approximate a_t_0 by a 5th degree polynomial
        fit_a = Polynomials.fit(ind_m_non_0 .* dt .- dt, a_t_0[ind_m_non_0], 5)
        ind_m = collect(0:dt:m_s-dt)
        @. @views a_t_0 = fit_a[0] + fit_a[1]*ind_m + fit_a[2]*ind_m^2 +
            fit_a[3]*ind_m^3 + fit_a[4]*ind_m^4 + fit_a[5]*ind_m^5

        # Copy current state for drift calculation
        copyto!(copy_xi_1_t, xi_1_t)
        copyto!(copy_xi_0_t, xi_0_t)

        # Apply changes due to jumps
        taux_saut = f(I_t, net) .* (1 .- V_t) .+ beta .* V_t
        tau_t = -log.(rand(N)) ./ taux_saut
        ind_saut_neur = tau_t .< dt
        n_jump[ii] = sum(ind_saut_neur)

        # Apply changes due to drift terms on xi distributions
        drift_xi!(net, S_t, xi_0_t, xi_1_t, copy_xi_0_t, copy_xi_1_t, a_t_0, I_t)
        jump_terms!(net, ind_saut_neur, tau_t, V_t, S_t, xi_0_t, xi_1_t)

        t_current += dt

        # Verbose output for debugging
        verbose && println(maximum(I_t), " ", minimum(xi_0_t), " ", minimum(xi_1_t))
        verbose && println(sum(xi_0_t + xi_1_t) * dt / N  , " ")
    end

    return (;moy_taux_saut, Esp_V, Esp_S,
        V_t,S_t,xi_0_t,xi_1_t,
        Esp_V_xi, Esp_S_xi, Esp_W_xi,
        n_jump, times, I_t, a_t_0, copy_a_t_0)
end

# Simulation function for particle system (discrete neurons)
function simu_traj_particle!(net::STDPNetwork_lim, initialCondition; verbose = true)
    # Initialization - extract necessary parameters (automatic)
    @unpack_STDPNetwork_lim net
    @unpack V_t_part, S_t_part, W_t_part = initialCondition
    ii_max          = Int(floor((tmax - tmin)/dt))

    # Initialize arrays for storing results
    Esp_V_part      = Vector{Float64}(undef, ii_max)
    Esp_S_part      = Vector{Float64}(undef, ii_max)
    Esp_W_part      = Vector{Float64}(undef, ii_max)
    times           = Vector{Float64}(undef, ii_max)
    I_t_part        = Vector{Float64}(undef, N)
    t_current       = 0.

    # Time loop
    @time @showprogress for ii = 1:ii_max
        # Calculate input intensity I for each neuron (previous time step)
        I_t_part .= W_t_part * V_t_part / N

        # Record quantities for all time
        Esp_V_part[ii]      = sum(V_t_part) / N
        Esp_S_part[ii]      = sum(S_t_part) / N
        Esp_W_part[ii]      = sum(W_t_part) / (N^2)
        times[ii]           = t_current

        # Apply changes due to jumps for the particle system
        update_part!(net, V_t_part, S_t_part, W_t_part, I_t_part)
        t_current += dt
    end

    return (;V_t_part,S_t_part,W_t_part,
        I_t_part, Esp_V_part, Esp_S_part, Esp_W_part, times)
end
