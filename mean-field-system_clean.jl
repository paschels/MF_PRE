using Revise
cd(@__DIR__)
using Pkg; pkg"activate ."

# Include the utility functions for mean-field simulation
includet("utilsMF_clean.jl")
fntlg = Plots.font(16)
default(titlefont=fntlg, guidefont=fntlg, tickfont=fntlg, legendfont=fntlg,
	lw=3, #color_palette = test
	)

#############
# Initialize network parameters
# Create STDP network with 2000 neurons, 15ms simulation time, and 0.05ms time step
net = STDPNetwork_lim(N = 5000, tmax = 500., dt = 5e-2, m_s=15)

# Generate initial conditions
# This sets up the initial state of neurons, weights, and distributions
CI = @time cond_initiales(net);

# Important: CI is modified in-place, so we need to pass a copy to preserve original state
# Run mean-field simulation
result_MF = simu_traj_MF!(net, deepcopy(CI); verbose = false)
# Run particle-based simulation for comparison
result_particle = simu_traj_particle!(net, deepcopy(CI))

using JLD2
save("resultats-MF-part-vs-limit-0-500ms_ms15_N5000.jld2", "result_MF", result_MF, "result_particle", result_particle)

######################################################################
# Expectation trajectories: particle system vs mean-field
######################################################################
# E[V_t]: mean membrane potential
plot(result_particle.times, result_particle.Esp_V_part, label=L"\overline{V_t^{N}}", legend=:bottomright)
	plot!(result_MF.times, result_MF.Esp_V, label=L"\overline{V_t^{*}}")
	plot!(result_MF.times, result_MF.Esp_V_xi, label=L"\mathbb{E}[\overline{\xi_t^{*}}(\ \cdot , \mathbb{R}^+ , \mathbb{Z})]",
      xlabel=L"time\ (ms)", ylabel=L"potential")
savefig("../MF_PRE/images-svg/esperance-V-final-500ms.svg")

# E[S_t]: mean time since last spike
plot(result_particle.times, result_particle.Esp_S_part, label=L"\overline{S_t^{N}}")
	plot!(result_MF.times, result_MF.Esp_S, label=L"\overline{S_t^{*}}")
	plot!(result_MF.times, result_MF.Esp_S_xi, label=L"\mathbb{E}[\overline{\xi_t^{*}}(\{0,1\},\cdot , \mathbb{Z})]",
      xlabel=L"time\ (ms)", ylabel=L"ms")
savefig("../MF_PRE/images-svg/esperance-S-final-500ms.svg")

# E[W_t]: mean synaptic weight
plot(result_particle.times, result_particle.Esp_W_part, label=L"\overline{W_t^{N}}")
	plot!(0:net.dt:net.tmax-net.dt, result_MF.Esp_W_xi, label=L"\mathbb{E}[\overline{\xi_t^{*}}(\{0,1\}, \mathbb{R}^+ , \cdot)]",
      xlabel=L"time\ (ms)", ylabel=L"weight", legend=:bottomright)
savefig("../MF_PRE/images-svg/esperance-W-final-500ms.svg")


######################################################################
# Input current distribution
######################################################################

histogram(result_particle.I_t_part, label=L"\hat{P}^N_{I^N}", normed=true, nbins=100, linetype = :stephist)
	histogram!(result_MF.I_t, label=L"\hat{P}^N_{I^*}", normed=true, nbins=100, linetype = :stephist,
           xlabel=L"synaptic\ current", ylabel=L"density\ function", legend=:top)
savefig("../MF_PRE/images-svg/distribution-I-final-500ms.svg")

######################################################################
# Distribution comparisons: time since last spike (S), split by state V
######################################################################
i0 = 1  # neuron index used throughout for per-neuron diagnostics
# Inactive neurons (V = 0)
begin
    dt, m_s = net.dt, net.m_s
    V_t_part, S_t_part = result_particle.V_t_part, result_particle.S_t_part
    V_t, S_t, xi_0_t   = result_MF.V_t, result_MF.S_t, result_MF.xi_0_t

    histogram(S_t_part[findall(V_t_part .< 1)], normed = true, nbins = 0:4*dt:m_s-dt, linetype = :stephist,
              label=L"\hat{P}^N_{S^N|V^N=0}", xlims = (0, m_s))
    histogram!(S_t[findall(V_t .< 1)], normed = true, nbins = 0:4*dt:m_s-dt, linetype = :stephist,
               label=L"\hat{P}^N_{S^*|V^*=0}")
    plot!(collect(0:dt:m_s-dt), 1/(dt*sum(xi_0_t[i0,:,:])) * sum(xi_0_t[i0,:,:], dims=2)[:],
          xlabel=L"ms", ylabel=L"density\ function",
          label=L"\xi_{t}^{i,*}(0, \cdot, \mathbb{Z})")
end
savefig("../MF_PRE/images-svg/distribution-S-final-V0-500ms.svg")

# Active neurons (V = 1)
begin
    N, dt, m_s       = net.N, net.dt, net.m_s
    V_t_part, S_t_part = result_particle.V_t_part, result_particle.S_t_part
    V_t, S_t, xi_1_t   = result_MF.V_t, result_MF.S_t, result_MF.xi_1_t

    histogram(S_t_part[findall(V_t_part .> 0)], nbins = 0:dt:m_s-dt, normed = true, linetype = :stephist,
              label=L"\hat{P}^N_{S^N|V^N=1}")
    histogram!(S_t[findall(V_t .> 0)], nbins = 0:dt:m_s-dt, normed = true, linetype = :stephist,
               label=L"\hat{P}^N_{S^*|V^*=1}")
    plot!(collect(0:dt:m_s-dt), 1/(dt*sum(xi_1_t[i0,:,:])) * sum(xi_1_t[i0,:,:], dims=2)[:],
          xlabel=L"ms", ylabel=L"density\ function",
          label=L"\xi_{t}^{i,*}(1, \cdot, \mathbb{Z})", xlims = (0, 8))
end
savefig("../MF_PRE/images-svg/distribution-S-final-V1-500ms.svg")

######################################################################
# Analysis of the firing-rate approximation a_t^0, recomputed directly
# from the final MF state (independent check of what the simulation
# computes internally)
######################################################################

# m_etoile: discretized time since last spike, for inactive neurons only
m_0 = (V_t .== 0)
m_etoile = min.(Int.(floor.(S_t .* (1 .- V_t) / dt)) .+ 1, net.M_s)

# Empirical firing rate a_t^0 as a function of time since last spike
a_t_0 = zeros(net.M_s)
count_m_etoile = zeros(net.M_s)
for k in findall(m_etoile .> 1)
    a_t_0[m_etoile[k]] += f(result_MF.I_t[k], net)
    count_m_etoile[m_etoile[k]] += 1
end
ind_m_non_null = findall(count_m_etoile .> 0)
a_t_0[ind_m_non_null] ./= count_m_etoile[ind_m_non_null]

# Polynomial approximation of a_t_0 (used to smooth it for the MF equations)
order_poly = 5
ind_fit = findall(count_m_etoile .> 0)
fit_a = Polynomials.fit(ind_fit*dt .- dt, a_t_0[ind_fit], order_poly)
ind_m = collect(0:dt:net.m_s-dt)
@. @views a_t_0 = fit_a[0] + fit_a[1]*ind_m + fit_a[2]*ind_m^2 +
                  fit_a[3]*ind_m^3 + fit_a[4]*ind_m^4 + fit_a[5]*ind_m^5

# Compare empirical points against the kernel / polynomial / exponential approximations
ind_S_less_ms = m_0 .* (S_t .< net.m_s)
scatter(S_t[ind_S_less_ms], f(result_MF.I_t[ind_S_less_ms], net),
        label=L"\alpha(I_{t_f}^{i,*})\ |\ V_t^{i,*} = 0",
        xlabel=L"time\ since\ last\ spike\ [ms]", ylabel=L"synaptic\ current")
		plot!(dt:dt:net.m_s, a_t_0, label=L"Poly.\ approx.")
