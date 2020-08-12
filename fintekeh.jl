using Plots
using LinearAlgebra
using Optim
using Distributions
using ForwardDiff

function generate_brownian_trajectory(x0, drift, diffusion, tf, no_samples)
	# Generates biased drift-diffusion trajectory
	# x0 is a (1,dim) array of initial values
	# drift is a (1,dim) vector of drift terms
	# diffusion is a contant (for now)
	dim = length(x0)
	dt = tf/no_samples
	rand_vals=  randn(no_samples, dim)
	times = dt*[i for i=1:no_samples, j=1:dim] 
	x = x0.+ transpose(drift).*times + cumsum(sqrt(2*diffusion*dt) * rand_vals, dims=1)
	return x

end

function generate_cauchy_trajectory(x0, loc, scale, tf, no_samples)
	## Generates levy flights with step size chosen from
	# A truncated power law survivor func
	# P(U > u) = { 1 if u<=1; u^{-D} for  u > 1}
	dim = length(x0)
	times = dt* [ i for i=1:no_samples, j=1:dim]

	dist = Cauchy(loc, scale* dt)
	rand_vals = rand(dist, no_samples, dim)
	x = x0 .+ cumsum(rand_vals, dims=1)
end

function infer_drift_diffusion(xt, dt=1, hessian::Bool=false)
	# Bayesian parameter inference on drift and diffusion parameters
	## Max A Posterior (MAP) estimate
	sol = optimize(p -> mlogp_driftdiffusion(xt, p, dt),[-Inf,0], [Inf,Inf], [-1.,1.], Fminbox( LBFGS() ); autodiff = :forward )
	if hessian == true
		hess = ForwardDiff.hessian(p -> mlogp_driftdiffusion(xt, p, dt), sol.minimizer)
		return sol, hess
	end

	## Fischer information Matrix (for error estimates)
end

function infer_cauchy(xt, dt=1, hessian::Bool=false)

	sol = optimize(p -> mlogp_cauchy(xt, p, dt),[-Inf,0], [Inf,Inf], [1.,1.], Fminbox( LBFGS() ); autodiff = :forward )
	if hessian == true
		hess = ForwardDiff.hessian(p -> mlogp_cauchy(xt, p, dt), sol.minimizer)
		return sol, hess
	end
end

function mlogp_cauchy(xt, p, dt)
	# Calculates log posterior given a drift diffusion model
	# p = (drift, diffusion)
	dim = size(xt)[1]
	logp = 0
	for i in 1:dim-1
		dx = xt[i+1,:] - xt[i,:]
		prob = (dx .- p[1]).^2 .+ (dt*p[2])^2 
		logp += - log.(prob)[1] + log(p[2]) ## exp to force positivity in smooth way
	end
	return -logp
end

function mlogp_driftdiffusion(xt, p, dt)
	# Calculates log posterior given a drift diffusion model
	# p = (drift, diffusion)
	dim = size(xt)[1]
	logp = 0
	for i in 1:dim-1 #NOTE: exclude first one
		dx = xt[i+1,:] - (xt[i,:] .+ p[1] * dt)
		logp += -dot(dx,dx)/(2 * p[2] )
	end
	D = p[2] * Matrix{Float64}(I, dim, dim)
	logp += - 0.5 * logdet(D) ## exp to force positivity in smooth way
	return -logp
end

function isamp_evidence(xt, p_inf, points = 100)
	## Calculate Bayesian model evidence 
	## p(x_t| M) = ∫ p(x| θ, M) p(θ | M) dθ
	## Assume flat prior on the parameters of the model for now
	## Calculated using MCMC
	
	# Importance sampling: Generate random values
	dim = size(p_inf)[1]
	dist = MvNormal(dim, 1)
	rvals = rand(dist, points)
	
	evidence = 0
	for i=1:points
		r = rvals[:,i]
		r[2] = abs(r[2]) ## no neg diffusio/scalen
		evidence += exp.(-mlogp_driftdiffusion(xt, r, dt))/pdf(dist, r)
		#println(evidence)
	end
	evidence /= points
	return evidence#, rvals
end
	
	
function time_correlator(xt)
	y = [mean(xt[1+j:end].*xt[1:end-j]) - mean(xt[1+j:end]) * mean(xt[1:end-j]) for j in 1:size(xt)[1]-1 ]
end


## Test the functions
x0 = [0]
drift = 2#*collect(1:5)
diff = 0.5
loc = 0.56
scale = 2.17
tf = 10
no_samples = 100

dt = tf/no_samples


xt=generate_brownian_trajectory(x0, drift, diff, tf, no_samples)
sol, hess = infer_drift_diffusion(xt, dt, true)
bars = sqrt.(diag(inv(hess)))
println("drift is $(sol.minimizer[1]) +/- $(bars[1]) and diff is $(sol.minimizer[2]) +/- $(bars[2])") 
#plot(res)
xc=generate_cauchy_trajectory(x0, loc, scale, tf, no_samples)
sol_c, hess_c = infer_cauchy(xc, dt, true)
bars_c = sqrt.(diag(inv(hess_c)))
println("loc is $(sol_c.minimizer[1]) +/- $(bars_c[1]) and scale is $(sol_c.minimizer[2]) +/- $(bars_c[2])") 

## evidence_calc
#
e1 = isamp_evidence(xt, sol.minimizer)
print(e1)
