using StochTE
using Test

tests = ["bestiary_tests", "model_comparison"]
models = [BrownianModel, CauchyModel, GeometricModel, OUModel, ARModel]

for t in tests
	@testset "$t" begin
		include("$(t).jl")
	end
end

