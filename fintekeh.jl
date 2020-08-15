using Plots
using LinearAlgebra
using Optim
using Distributions
using ForwardDiff

struct BrownianModel
end

struct CauchyModel
end

struct GeometricModel
end

function generate_trajectory(x0, drift, diffusion, tf, no_samples, model::BrownianModel)
	# Generates drift-diffusion trajectory
	# x0 is a (1,dim) array of initial values
	# drift is a (1,dim) vector of drift terms
	# diffusion is a constant (for now)
	dim = length(x0)
	dt = tf/no_samples
	rand_vals=  randn(no_samples, dim)
	times = dt*[i for i=1:no_samples, j=1:dim] 
	x = x0.+ transpose(drift).*times + cumsum(sqrt(diffusion*dt) * rand_vals, dims=1)
	return x

end

function generate_trajectory(x0, loc, scale, tf, no_samples, model::CauchyModel)
	## Generates discontinuous levy flights with step size chosen from
	# Cauchy distribution
	dim = length(x0)
	times = dt* [ i for i=1:no_samples, j=1:dim]

	dist = Cauchy(loc, scale* dt)
	rand_vals = rand(dist, no_samples, dim)
	x = x0 .+ cumsum(rand_vals, dims=1)
end

function generate_trajectory(x0, p_drift, vol, tf, no_samples, model::GeometricModel)
	# Generates geomtric brownian motion
	# using analytic form rather than direct simulation
	# S_t = S_0 exp( (μ - σ^2/2)*t  σ W_t) 
	dim = length(x0)
	dt = tf/no_samples
	rand_vals=  randn(no_samples, dim)
	times = dt*[i for i=1:no_samples, j=1:dim] 
	x = x0 * transpose( exp.( (p_drift - vol^2/2)*times + vol *cumsum(rand_vals, dims=1)) )
	return x
end

function infer(xt, model, dt=1, hessian::Bool=false)
	# Bayesian parameter inference on drift and diffusion parameters
	## Max A Posterior (MAP) estimate
	sol = optimize(p -> mlogp(xt, p, dt, model),[-Inf,0], [Inf,Inf], [-1.,1.], Fminbox( LBFGS() ); autodiff = :forward ) ## note: seems to predict 2*diff ??
	if hessian == true
		hess = ForwardDiff.hessian(p -> mlogp(xt, p, dt, model), sol.minimizer)
		return sol, hess
	end

	## Fischer information Matrix (for error estimates)
end

##function infer_cauchy(xt, dt=1, hessian::Bool=false)
#
#	sol = optimize(p -> mlogp_cauchy(xt, p, dt),[-Inf,0], [Inf,Inf], [1.,1.], Fminbox( LBFGS() ); autodiff = :forward )
#	if hessian == true
#		hess = ForwardDiff.hessian(p -> mlogp_cauchy(xt, p, dt), sol.minimizer)
#		return sol, hess
#	end
#end


function mlogp(xt, p, dt, model::CauchyModel)
	# Calculates log posterior given a drift diffusion model
	# Appropriate normalization given
	# p = (drift, diffusion)
	dim = size(xt)[1]-1
	logp = 0
	for i in 1:dim
		dx = xt[i+1,:] - xt[i,:]
		prob = (dx .- p[1]).^2 .+ (dt*p[2])^2 
		logp += - log.(prob)[1] + log(dt*p[2]) - log(π) 
	end
	return -logp
end

function mlogp(xt, p, dt, model::BrownianModel)
	# Calculates log posterior given a drift diffusion model
	# p = (drift, diffusion)
	dim = size(xt)[1]-1
	D = dt* p[2] * Matrix{Float64}(I, dim, dim)
	invD = inv(D)
	dx = [ xt[i+1,:] - (xt[i,:] .+ p[1] * dt) for i=1:dim]

	logp = - 0.5* transpose(dx) * invD * dx
	logp += - 0.5 * logdet(D) - (dim/2) * log(2*π) ## exp to force positivity in smooth way
	return -logp
end

function isamp_evidence(xt, model, p_inf, inf_hess=I, points = 1000)
	## Calculate Bayesian model evidence 
	## p(x_t| M) = ∫ p(x| θ, M) p(θ | M) dθ
	## Assume flat prior on the parameters of the model for now
	## Calculated using MCMC
	
	# Importance sampling: Generate random values
	dim = size(p_inf)[1]
	invsig = 0.5 *( transpose(inf_hess) + inf_hess)
	dist = MvNormal(p_inf, inv(invsig) )
	rvals = rand(dist, points)
	
	evidence = 0
	for i=1:points
		r = rvals[:,i]
		r[2] = abs(r[2]) ## no neg diff/scale wlog
		evidence += exp.(-mlogp(xt, r, dt, model))/pdf(dist, r)
		#println(evidence)
	end
	evidence /= points
	return evidence#, rvals
end

#function isamp_evidence_cauchy(xt, p_inf,inf_hess=I,  points = 1000)
#	## Calculate Bayesian model evidence 
#	## p(x_t| M) = ∫ p(x| θ, M) p(θ | M) dθ
#	## Assume flat prior on the parameters of the model for now
#	## Calculated using MCMC
#	
#	# Importance sampling: Generate random values
#	dim = size(p_inf)[1]
#	sig = 0.5 *( transpose(inf_hess) + inf_hess)
#	dist = MvNormal(p_inf, sig)
#	rvals = rand(dist, points)
#	
#	evidence = 0
#	for i=1:points
#		r = rvals[:,i]
#		r[2] = abs(r[2]) ## no neg diffusio/scalen
#		evidence += exp.(-mlogp_cauchy(xt, r, dt))/pdf(dist, r)
#		#println(evidence)
#	end
#	evidence /= points
#	return evidence#, rvals
#end

	
function time_correlator(xt)
	y = [mean(xt[1+j:end].*xt[1:end-j]) - mean(xt[1+j:end]) * mean(xt[1:end-j]) for j in 1:size(xt)[1]-1 ]
end


## Test the functions
x0 = [1]
drift = 0.93#*collect(1:5)
diff = 0.50

loc = 0.56
scale = 0.17
tf = 10
no_samples = 100

dt = tf/no_samples


xc=generate_trajectory(x0, loc, scale, tf, no_samples, CauchyModel())
xt=generate_trajectory(x0, drift, diff, tf, no_samples, BrownianModel())
xg=generate_trajectory(x0, drift, diff, tf, no_samples, GeometricModel())
data = xc

sol_b, hess_b = infer(data, BrownianModel(), dt, true)
bars_b = sqrt.(diag(inv(hess_b)))
println("drift is $(sol_b.minimizer[1]) +/- $(bars_b[1]) and diff is $(sol_b.minimizer[2]) +/- $(bars_b[2])") 
##
sol_c, hess_c = infer(data, CauchyModel(), dt, true)
bars_c = sqrt.(diag(inv(hess_c)))
println("loc is $(sol_c.minimizer[1]) +/- $(bars_c[1]) and scale is $(sol_c.minimizer[2]) +/- $(bars_c[2])") 

## evidence_calc
#
e1 = isamp_evidence(data, BrownianModel(), sol_b.minimizer, hess_b)
e2 = isamp_evidence(data, CauchyModel(), sol_c.minimizer, hess_c)
print(e1, "\n", e2)
