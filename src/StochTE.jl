module StochTE

using Plots
using LinearAlgebra
using Optim
using Distributions
using ForwardDiff

export forecast, 
	infer, 
	rand_reject, 
	evidence,
	generate_trajectory,
	mlogp,
	BrownianModel,
	CauchyModel,
	GeometricModel,
	OUModel

include("bestiary.jl")

############## INFERENCE AND EVIDENCE #########################
#
function forecast(xt, model, dt, n_obs, Δ_future, n_traj=100)
	# produces a stochastic forecast of the data, according to selected model
	# from n_obs to n_obs + Δ_future
	sol, hess = infer(xt, model, dt, true)
	x0 = xt[end]

	no_points = floor(Δ_future/dt)
	x_pred = Array{Float64, 2}(undef, n_traj, no_points)
	for i=1:n_traj
		x_pred[i,:] = generate_trajectory(x0, model, sol.minimizer, Δ_future, no_points)
	end
	return x_pred
end

function infer(xt, model, dt=1, hessian::Bool=false)
	# Bayesian parameter inference on drift and diffusion parameters
	## Max A Posterior (MAP) estimate
	sol = optimize(p -> mlogp(xt, p, dt, model), model.params_lower, model.params_upper, [1.,1.], Fminbox( LBFGS() ); autodiff = :forward ) ## note: seems to predict 2*diff ??
	if hessian == true
		hess = ForwardDiff.hessian(p -> mlogp(xt, p, dt, model), sol.minimizer)
		return sol, hess
	end
end
	
function rand_reject(model, dist, points)
	## Sampling from a distribution with rejection if value 
	## is outside of the bounds of the passed model.
	## Equivalent to a multivariate tuncated dist
	rvals = Array{Float64, 2}(undef, model.param_no, points)
	i=1
	while i < points+1
		rval = rand(dist)
		if model.params_lower < rval && model.params_upper > rval
			rvals[:,i] = rval
			i +=1
		end
	end
	return rvals
end

function evidence(xt, model, p_inf, dt = 1, inf_hess=I, points = 1000)
	## Calculate Bayesian model evidence 
	## p(x_t| M) = ∫ p(x| θ, M) p(θ | M) dθ
	## Assume flat prior on the parameters of the model for now
	## Calculated using importance sampling
	
	invsig = 0.5 *( transpose(inf_hess) + inf_hess)
	sig = inv(invsig)
	sig = 0.5*(sig + transpose(sig))
	dist = MvNormal(p_inf, sig )
	rvals = rand_reject(model, dist, points)
	
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
end
