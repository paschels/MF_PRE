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

# Visualize the initial xi_0 distribution for the first neuron
# Shows how the distribution varies with weight (W) and time since last spike (S)
heatmap(CI.xi_0_t[1,:,:], xlabel = "W", ylabel ="S", title = "ξ₀¹")

# Important: CI is modified in-place, so we need to pass a copy to preserve original state
# Run mean-field simulation
result_MF = simu_traj_MF!(net, deepcopy(CI); verbose = false)
# Run particle-based simulation for comparison
result_particle = simu_traj_particle!(net, deepcopy(CI))

# Plot comparison of mean time since last spike (S) between methods
plot(result_particle.times, result_particle.Esp_S_part,label="\$ E[S_t^{i,N}] \$")
	plot!(result_MF.times,result_MF.Esp_S, label="\$ E[S_t^{i,*}] \$")
	plot!(result_MF.times,result_MF.Esp_S_xi,label=L" E[\xi_t^{i,*}(\{0,1\},\cdot , Z)]) ", xlabel=L" time\ (ms) ", ylabel=L"ms")

# Plot comparison of mean weight evolution between methods
plot(result_MF.times,result_MF.Esp_W_xi,label="Esp_W_xi")
	plot!(0:net.dt:net.tmax-net.dt,result_particle.Esp_W_part,label="Esp_W_part",legend=:bottomright)

########## BEGIN TEST - DISTRIBUTION COMPARISONS #################
i0 = 1  # Focus on first neuron for detailed analysis

# Compare distributions of time since last spike for inactive neurons (V=0)
histogram(result_particle.S_t_part[result_particle.V_t_part .< 1])
	histogram!(result_MF.S_t[result_MF.V_t .< 1])

# Compare distributions of time since last spike for active neurons (V=1)
histogram(result_particle.S_t_part[result_particle.V_t_part .> 0])
	histogram!(result_MF.S_t[result_MF.V_t .> 0])

# Detailed comparison of S distribution for inactive neurons (V=0)
# Shows empirical distribution from particle system vs mean-field approximation
begin
    @unpack dt, m_s = net
    (;V_t_part, S_t_part) = result_particle
    (;V_t, S_t, xi_0_t) = result_MF
    i0 = 1
    histogram(S_t_part[findall(V_t_part .< 1)],normed = true,nbins=0:4*dt:m_s-dt,linetype = :stephist,
              label="\$ initial\\ part. \\ syst. \\ (S_t^{i,N})\$", xlims = (0,m_s))
    	histogram!(S_t[findall(V_t .< 1)],normed = true, nbins=0:4*dt:m_s-dt, linetype = :stephist,
               label="\$ MF(N)\\ part. \\ syst.\\ (S_t^{i,*}) \$")
    	plot!(collect(0:dt:m_s-dt),1/(dt*sum(xi_0_t[i0,:,:]))*sum(xi_0_t[i0,:,:],dims=2)[:],
          xlabel="\$ ms \$",ylabel="\$ density\\ function \$",
          label="\$ MF(N)\\ part. \\ syst.\\ (\\xi_{t}^{i, *})\$")
end
#savefig("../images-svg/distribution-S-final-V0-500ms.svg")

# Detailed comparison of S distribution for active neurons (V=1)
begin
    @unpack N, dt, m_s = net
    (;V_t_part, S_t_part) = result_particle
    (;V_t, S_t, xi_0_t, xi_1_t) = result_MF
    histogram(S_t_part[findall(V_t_part .> 0)],nbins=0:dt:m_s-dt,normed = true,linetype = :stephist,
              label="\$ initial\\ part. \\ syst. \\ (S_t^{i,N})\$")
    	histogram!(S_t[findall(V_t .> 0)],nbins=0:dt:m_s-dt,normed = true,linetype = :stephist,
               label="\$ MF\\ part. \\ syst.\\ (S_t^{i,*}) \$")
    	plot!(collect(0:dt:m_s-dt),1/(dt*sum(xi_1_t[i0,:,:]))*sum(xi_1_t[i0,:,:],dims=2)[:],
          xlabel="\$ ms \$",ylabel="\$ density\\ function \$",
          label="\$ MF(N)\\ part. \\ syst.\\ (\\xi_{t}^{i, *})\$", xlims = (0,8))
end
#savefig("../images-svg/distribution-S-final-V1-500ms.svg")

# Diagnostic checks for distribution normalization
length(findall(V_t .> 0))
sum(xi_1_t[1,:,:]) .* dt *N
length(findall(V_t .< 1))
sum(xi_0_t[1,:,:]) .* dt * N
length(S_t[findall(S_t .> m_s)])/N

# Analyze the firing rate approximation a_t_0
ind_m_non_0 = findall(copy_a_t_0 .> 0)
time_x = ind_m_non_0 * dt
scatter(time_x,copy_a_t_0[ind_m_non_0])
	plot!(collect(0:dt:m_s-dt),a_t_0)

# Plot comparison of mean membrane potential (V) evolution
begin
    plot(result_particle.times,result_particle.Esp_V_part,label="\$ E[V_t^{i,N}] \$",legend=:bottomright)
    	plot!(result_MF.times,result_MF.Esp_V, label="\$ E[V_t^{i,*}] \$")
    	plot!(result_MF.times,result_MF.Esp_V_xi,label=L" E[\xi_t^{i,*}(\cdot , R^+ , Z)]) ",
    xlabel=L" time\ (ms) ", ylabel=L" potential ")
end
savefig("../images-svg/esperance-V-final-500ms.svg")

# Plot comparison of mean time since last spike (S) evolution
begin
    plot(result_particle.times,result_particle.Esp_S_part,label="\$ E[S_t^{i,N}] \$")
    	plot!(result_MF.times,result_MF.Esp_S, label="\$ E[S_t^{i,*}] \$")
    	plot!(result_MF.times,result_MF.Esp_S_xi,label=L" E[\xi_t^{i,*}(\{0,1\},\cdot , Z)]) ",
        xlabel=L" time\ (ms) ", ylabel=L"ms")
end
savefig("../images-svg/esperance-S-final-500ms.svg")

# Plot comparison of mean weight (W) evolution
begin
    plot(result_particle.times,result_particle.Esp_W_part,label="\$ E[W_t^{ij,N}] \$")
    	plot!(0:net.dt:net.tmax-net.dt,result_MF.Esp_W_xi,label=L" E[\xi_t^{i,*}(\{0,1\}, R^+ , \cdot)]) ",
        xlabel=L" time\ (ms) ", ylabel=L"weight",legend=:bottomright)
end
savefig("../images-svg/esperance-W-final-500ms.svg")

# Compare input current distributions between methods
histogram(result_particle.I_t_part,label=L"I_t^{i,N}",normed=true,nbins=100, linetype = :stephist)
	histogram!(result_MF.I_t,label=L"I_t^{i,*}",normed=true,nbins=100, linetype = :stephist,
    xlabel=L" synaptic \ current ", ylabel=L"density\ function",legend=:top)
savefig("../images-svg/distribution-I-final-500ms.svg")

# Analyze weight distributions for active neurons
histogram(reshape(W_t_part[findall(V_t_part .> 0),:],N*length(findall(V_t_part .> 0)),1),
    normed = true,nbins=w_min-1:1:w_max)
# Compare with mean-field weight distribution for active state
histoW_xi_1 = sum(sum(xi_1_t,dims=1),dims=2)[:] / sum(xi_1_t)
	plot!(vcat(w_min-1,collect(w_min-1:1:w_max),w_max), vcat(0,histoW_xi_1[1],histoW_xi_1,0),
    legend = false,linetype=:steppost)

# Analyze weight distributions for inactive neurons
histogram(reshape(W_t_part[findall(V_t_part .< 1),:],N*length(findall(V_t_part .< 1)),1),
    normed = true,nbins=w_min-1:1:w_max)
# Compare with mean-field weight distribution for inactive state
histoW_xi_0 = sum(sum(xi_0_t,dims=1),dims=2)[:] / sum(xi_0_t)
	plot!(vcat(w_min-1,collect(w_min-1:1:w_max),w_max), vcat(0,histoW_xi_0[1],histoW_xi_0,0),
    legend = false,linetype=:steppost)

#### ANALYSIS OF FIRING RATE APPROXIMATION ####
# Calculate m_etoile - discretized time since last spike for inactive neurons
m_0 = (V_t.==0)
m_etoile = min.(Int.(floor.(S_t .* (1 .- V_t) / dt)) .+ 1, M_s)

# Calculate a_t^0 - empirical firing rate as function of time since last spike
a_t_0 = zeros(net.M_s)
count_m_etoile = zeros(M_s)
for k in findall(m_etoile .> 1)
	a_t_0[m_etoile[k]] += f(I_t[k], net)
	count_m_etoile[m_etoile[k]] += 1
end
ind_m_non_null = findall(count_m_etoile .> 0)
a_t_0[ind_m_non_null] ./= count_m_etoile[ind_m_non_null]

# Approximate a_t_0 using polynomial fitting
# This smooths the empirical firing rate for use in the mean-field equations
order_poly = 5
min_occ = 0
ind_m_min_occ = findall(count_m_etoile .> min_occ)
ind_fit = ind_m_min_occ

# Try both polynomial and exponential fits for comparison
fit_a1 = Polynomials.fit(ind_fit*dt .- dt,log.(a_t_0[ind_fit] .-  minimum(a_t_0[ind_fit]) .+ 0.01),order_poly)
ind_m = collect(0:dt:m_s-dt)
fit_a = Polynomials.fit(ind_fit*dt .- dt, a_t_0[ind_fit] ,5)
ind_m = collect(0:dt:m_s-dt)
@. @views a_t_0 = fit_a[0] + fit_a[1]*ind_m + fit_a[2]*ind_m^2 +
	fit_a[3]*ind_m^3 + fit_a[4]*ind_m^4 + fit_a[5]*ind_m^5

# Exponential approximation for comparison
current_s_exp = minimum(a_t_0[ind_fit]) .+ exp.(fit_a1[0] .+ fit_a1[1]*ind_m)

# Kernel density estimation of the firing rate
dist_current_s = smooth(S_t[m_0], f(I_t[m_0], net),bwsj(S_t[m_0]),0:dt:m_s)

# Plot comparison of different firing rate approximations
ind_S_less_ms = m_0.*(S_t.<m_s)
scatter(S_t[ind_S_less_ms],f(I_t[ind_S_less_ms],net),
        label=L"\alpha (I_{t_f}^{i, *}) \| V_t^{i,*} = 0",
        xlabel=L"time \ since \ last \ spike \ [ms]", ylabel=L"synaptic\ current")
plot!(0:dt:m_s,dist_current_s, label=L" kernel\ approx.",
      xlabel=L"time \ since \ last \ spike \ [ms]", ylabel=L"synaptic\ current")
plot!(dt:dt:m_s,a_t_0, label=L"Poly.\ approx.")
plot!(dt:dt:m_s,current_s_exp, label=L" exp.\ approx. (occ>0)")
