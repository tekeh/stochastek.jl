## Test the functions
p1 = 0.93
p2 = 0.50

no_samples=100
tf = 10
dt = tf/no_samples

#models = [BrownianModel, CauchyModel, GeometricModel, OUModel]
for m in models
	@testset "$(m().string)" begin ## effectively tests initialization
		xt = @test_nowarn generate_trajectory([1], m(), [p1,p2], tf, no_samples)
		@test typeof(mlogp(xt,[p1,p2], dt, m())) == Float64 
	end
end


