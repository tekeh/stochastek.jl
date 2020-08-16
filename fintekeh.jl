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
struct OUModel
end

eps = 1e-4
################ STOCHASTIC TRAJECTORY SIMULATION ################
function generate_trajectory(x0, model::BrownianModel, drift, diffusion, tf, no_samples)
	# Generates drift-diffusion trajectory
	dim = length(x0)
	dt = tf/no_samples
	rand_vals=  randn(no_samples, dim)
	times = dt*[i for i=1:no_samples, j=1:dim] 
	x = x0.+ transpose(drift).*times + cumsum(sqrt(diffusion*dt) * rand_vals, dims=1)
	return x

end

function generate_trajectory(x0, model::CauchyModel, loc, scale, tf, no_samples)
	## Generates discontinuous levy flights with step size chosen from
	# Cauchy distribution
	dim = length(x0)
	times = dt* [ i for i=1:no_samples, j=1:dim]

	dist = Cauchy(loc, scale* dt)
	rand_vals = rand(dist, no_samples, dim)
	x = x0 .+ cumsum(rand_vals, dims=1)
end

function generate_trajectory(x0, model::GeometricModel, p_drift, vol, tf, no_samples)
	# Generates geomtric brownian motion
	# using analytic form rather than direct simulation
	# S_t = S_0 exp( (μ - σ^2/2)*t  σ W_t) 
	dim = length(x0)
	dt = tf/no_samples
	rand_vals=  randn(no_samples, dim)
	times = dt*[i for i=1:no_samples, j=1:dim] 
	x = x0 * transpose(exp.( (p_drift - vol^2/2)*times + vol *cumsum(rand_vals, dims=1)))
	return reshape(x, (100,1))
end

function generate_trajectory(x0, model::OUModel, drift, diffusion, tf, no_samples)
	# Generates  OU (mean reverting) process trajectory
	dim = length(x0)
	dt = tf/no_samples
	rand_vals=  randn(no_samples, dim)
	xt = Array{Float64,2}(undef, no_samples, dim)
	xt[1,:] = x0
	
	## simulate, due to lack of exact sol
	for i in 1:no_samples-1
		dx = - drift * dt * xt[i] + sqrt(diffusion * dt) * rand_vals[i]
		xt[i+1] = xt[i] + dx
	end
	return xt

end

######### LOG P FUNCTIONS ###########################
#
function mlogp(xt, p, dt, model::BrownianModel)
	# Calculates log posterior given a drift diffusion model
	# p = (drift, diffusion)
	dim = size(xt)[1]-1
	D = dt* p[2] * Matrix{Float64}(I, dim, dim)
	invD = inv(D)
	dx = [ xt[i+1] - (xt[i] .+ p[1] * dt) for i=1:dim]

	logp = - 0.5* transpose(dx) * invD * dx
	logp += - 0.5 * logdet(D) - (dim/2) * log(2*π) ## exp to force positivity in smooth way
	return -logp
end

function mlogp(xt, p, dt, model::CauchyModel)
	# Calculates log posterior given a drift diffusion model
	# Appropriate normalization included
	# p = (loc, scale)
	dim = size(xt)[1]-1
	logp = 0
	for i in 1:dim
		dx = xt[i+1] - xt[i]
		prob = (dx .- p[1]).^2 .+ (dt*p[2])^2 
		logp += - log.(prob)[1] + log(dt*p[2]) - log(π) 
	end
	return -logp
end

function mlogp(xt, p, dt, model::GeometricModel)
	# Calculates log posterior given a geometric diffusion model
	# Appropriate normalization included
	# p = [%_drift, %_volatility]
	#
	if any(t->t<0, xt) ## note:raise exception here 
		#println("Warning: GEOMETRIC model can not account for negative values in the time-series")
		return Inf 
	end
	dim = size(xt)[1]-1
	logp = 0
	for i in 1:dim
		logp += -(log.(xt[i+1]) .- log.(xt[i]) .- (p[1] - 0.5 * p[2]^2) * dt).^2 ./(2 * p[2]^2 * dt)  .- log.(xt[i+1]) ## Beware floating-point madness in divisions
	end
	logp += -(dim/2) * log.(2*π*dt*p[2]^2)
	return -logp
end

function mlogp(xt, p, dt, model::OUModel)
	# Calculates log posterior given a no drift OU process
	dim = size(xt)[1]-1
	logp=0
	for i in 1:dim
		logp += - (p[1]/(2*p[2])) * (xt[i+1] - xt[i] * exp(-p[1]*dt))^2/(1 - exp(-2*p[1]*dt))
	end
	logp += (dim/2)*log(p[1]) - (dim/2)*log(2π * p[2] * (1 - exp(-2*p[1]*dt)))
	return -logp
end

############## INFERENCE AND EVIDENCE #########################
#
function infer(xt, model, dt=1, hessian::Bool=false)
	# Bayesian parameter inference on drift and diffusion parameters
	## Max A Posterior (MAP) estimate
	sol = optimize(p -> mlogp(xt, p, dt, model),[eps,eps], [Inf,Inf], [1.,1.], Fminbox( LBFGS() ); autodiff = :forward ) ## note: seems to predict 2*diff ??
	if hessian == true
		hess = ForwardDiff.hessian(p -> mlogp(xt, p, dt, model), sol.minimizer)
		return sol, hess
	end
end
	
function isamp_evidence(xt, model, p_inf, inf_hess=I, points = 1000)
	## Calculate Bayesian model evidence 
	## p(x_t| M) = ∫ p(x| θ, M) p(θ | M) dθ
	## Assume flat prior on the parameters of the model for now
	## Calculated using importance sampling
	
	dim = size(p_inf)[1]

	invsig = 0.5 *( transpose(inf_hess) + inf_hess)
	sig = inv(invsig)
	sig = 0.5*(sig + transpose(sig))
	dist = MvNormal(p_inf, sig )
	rvals = rand(dist, points)
	
	evidence = 0
	for i=1:points
		r = rvals[:,i]
		r[1] = abs(r[1]) ## note: this should be made more general to handle different ranges in models
		r[2] = abs(r[2]) ## no neg diff/scale wlog
		evidence += exp.(-mlogp(xt, r, dt, model))/pdf(dist, r)
		#println(evidence)
	end
	evidence /= points
	return evidence#, rvals
end


function time_correlator(xt)
	y = [mean(xt[1+j:end].*xt[1:end-j]) - mean(xt[1+j:end]) * mean(xt[1:end-j]) for j in 1:size(xt)[1]-1 ]
end

## Test the functions
x0 = [1]
drift = 0.93
diff = 0.50

loc = 0.09
scale = 0.17

p_drift = 0.41
p_vol = 0.73

tf = 10
no_samples = 100

dt = tf/no_samples


xc=generate_trajectory(x0, CauchyModel(), loc, scale, tf, no_samples)
xb=generate_trajectory(x0, BrownianModel(), drift, diff, tf, no_samples)
xg=generate_trajectory(x0, GeometricModel(), p_drift, p_vol, tf, no_samples)
xo=generate_trajectory(x0, OUModel(), drift, diff, tf, no_samples)
data = xg

sol_b, hess_b = infer(data, BrownianModel(), dt, true)
try
	bars_b = sqrt.(diag(inv(hess_b)))
	println("BROWNIAN:\t drift is $(sol_b.minimizer[1]) +/- $(bars_b[1]) and diff is $(sol_b.minimizer[2]) +/- $(bars_b[2])") 
catch e
	println("BROWNIAN curvature @ MAP estimate is very flat or negative...")
	global hess_b = I
	bars_b = ["nan", "nan"]
	println("BROWNIAN:\t drift is $(sol_b.minimizer[1]) +/- $(bars_b[1]) and diff is $(sol_b.minimizer[2]) +/- $(bars_b[2])") 
end

##
sol_c, hess_c = infer(data, CauchyModel(), dt, true)
try
	bars_c = sqrt.(diag(inv(hess_c)))
	println("CAUCHY:\tloc is $(sol_c.minimizer[1]) +/- $(bars_c[1]) and scale is $(sol_c.minimizer[2]) +/- $(bars_c[2])") 
catch e
	println("CAUCHY curvature @ MAP estimate is very flat or negative...")
	global hess_c = I
	bars_c = ["nan", "nan"]
	println("CAUCHY:\tloc is $(sol_c.minimizer[1]) +/- $(bars_c[1]) and scale is $(sol_c.minimizer[2]) +/- $(bars_c[2])") 
end
##
sol_g, hess_g = infer(data, GeometricModel(), dt, true)
try
	bars_g = sqrt.(diag(inv(hess_g)))
	println("GEOMETRIC:\tp_drift is $(sol_g.minimizer[1]) +/- $(bars_g[1]) and p_vol is $(sol_g.minimizer[2]) +/- $(bars_g[2])") 
catch e
	println("GEOMETRIC curvature @ MAP estimate is very flat or negative...")
	global hess_g = I
	bars_g = ["nan", "nan"]
	println("GEOMETRIC:\tp_drift is $(sol_g.minimizer[1]) +/- $(bars_g[1]) and p_vol is $(sol_g.minimizer[2]) +/- $(bars_g[2])") 
end
##
sol_o, hess_o = infer(data, OUModel(), dt, true)
try
	bars_o = sqrt.(diag(inv(hess_o)))
	println("OU:\tp_drift is $(sol_o.minimizer[1]) +/- $(bars_o[1]) and p_vol is $(sol_o.minimizer[2]) +/- $(bars_o[2])") 
catch e
	println("OU curvature @ MAP estimate is very flat or negative...")
	global hess_o = I
	bars_o = ["nan", "nan"]
	println("OU\tdrift is $(sol_o.minimizer[1]) +/- $(bars_o[1]) and diff is $(sol_o.minimizer[2]) +/- $(bars_o[2])") 
end

## evidence_calc
#
eb = isamp_evidence(data, BrownianModel(), 	sol_b.minimizer, hess_b)
ec = isamp_evidence(data, CauchyModel(), 	sol_c.minimizer, hess_c)
eg = isamp_evidence(data, GeometricModel(), 	sol_g.minimizer, hess_g)
eo = isamp_evidence(data, OUModel(), 	sol_o.minimizer, hess_o)
print("EVIDENCE VALUES \n\nBROWNIAN:\t $(eb) \nCAUCHY:\t  $(ec)\nGEOMETRIC:\t $(eg) \nOU:\t $(eo)")
