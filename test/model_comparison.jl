using LinearAlgebra
## Note: must clean up this test
tf = 50
no_samples = 100
dt = tf/no_samples
data = generate_trajectory([1], BrownianModel(), [1, .1], tf, no_samples)
evals = []

for m in models
	println(m)
	name = m().string
	sol, hess = infer(data, m(), dt, true)
	try
		bars = sqrt.(diag(inv(hess)))
		#println("$(name):\t drift is $(sol.minimizer[1]) +/- $(bars[1]) and diff is $(sol.minimizer[2]) +/- $(bars[2])") 
	catch e
		#println("$(name) curvature @ MAP estimate is very flat or negative...")
		hess = I
		bars = ["nan", "nan"]
		#println("$(name):\t drift is $(sol.minimizer[1]) +/- $(bars[1]) and diff is $(sol.minimizer[2]) +/- $(bars[2])") 
	end
	e = evidence(data, m(), sol.minimizer, dt, hess)
	push!(evals, e)
end

max_e, loc_max = findmax(transpose(evals))

@test models[loc_max] == BrownianModel ## should predict Brownian as best
