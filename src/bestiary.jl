## Model Bestiary. Defines the Julia structs which are model initialisers. Such structures can be passed to other functions to control the behaviour of the method

eps = 1e-6
struct BrownianModel
	string::String
	param_no::Int64
	params_lower::Array{Float64, 1}
	params_upper::Array{Float64, 1}
end
	BrownianModel() = BrownianModel("BROWNIAN", 2, [-Inf, eps], [Inf, Inf])

struct CauchyModel
	string::String
	param_no::Int64
	params_lower::Array{Float64, 1}
	params_upper::Array{Float64, 1}
end
	CauchyModel() = CauchyModel("CAUCHY", 2, [-Inf, eps], [Inf, Inf])

struct GeometricModel
	string::String
	param_no::Int64
	params_lower::Array{Float64, 1}
	params_upper::Array{Float64, 1}
end
	GeometricModel() = GeometricModel("GEOMETRIC", 2, [-Inf, eps], [Inf, Inf])

struct OUModel
	string::String
	param_no::Int64
	params_lower::Array{Float64, 1}
	params_upper::Array{Float64, 1}
end
	OUModel() = OUModel("OU", 2, [eps, eps], [Inf, Inf])

## defined constants

################ STOCHASTIC TRAJECTORY SIMULATION ################
function generate_trajectory(x0, model::BrownianModel, params, tf, no_samples)
	# Generates drift-diffusion trajectory
	dim = length(x0)
	drift, diffusion = params
	dt = tf/no_samples
	rand_vals=  randn(no_samples, dim)
	times = dt*[i for i=1:no_samples, j=1:dim] 
	x = x0.+ transpose(drift).*times + cumsum(sqrt(diffusion*dt) * rand_vals, dims=1)
	return x

end

function generate_trajectory(x0, model::CauchyModel, params, tf, no_samples)
	## Generates discontinuous levy flights with step size chosen from
	# Cauchy distribution
	dim = length(x0)
	loc, scale = params
	dt = tf/no_samples
	times = dt* [ i for i=1:no_samples, j=1:dim]

	dist = Cauchy(loc, scale* dt)
	rand_vals = rand(dist, no_samples, dim)
	x = x0 .+ cumsum(rand_vals, dims=1)
end

function generate_trajectory(x0, model::GeometricModel, params, tf, no_samples)
	# Generates geomtric brownian motion
	# using analytic form rather than direct simulation
	# S_t = S_0 exp( (μ - σ^2/2)*t  σ W_t) 
	dim = length(x0)
	p_drift, p_vol = params
	dt = tf/no_samples
	rand_vals=  randn(no_samples, dim)
	times = dt*[i for i=1:no_samples, j=1:dim] 
	x = x0 * transpose(exp.( (p_drift - p_vol^2/2)*times + p_vol *cumsum(rand_vals, dims=1)))
	return reshape(x, (no_samples,1))
end

function generate_trajectory(x0, model::OUModel, params, tf, no_samples)
	# Generates  OU (mean reverting) process trajectory
	dim = length(x0)
	drift, diffusion = params
	dt = tf/no_samples
	rand_vals = randn(no_samples, dim)
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

