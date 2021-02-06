-- Genetic algorithm test
-- The goal: Get 0.5

math.randomseed(os.time())
local NNL = require "NNL"
local population = NNL.genalg.population.new(20)

local y = {0.5}
local x = {1, 0, 1}

print(population.Networks[1]:Forward(x)[1])

for generation = 1, 10 do
	for i, brain in ipairs (population.Networks) do
		local output = brain:Forward(x)
		brain.Fitness = 1/math.abs(y[1]-output[1])
	end
	local fitness = population:GetTotalFitness()
	print ("GENERATION "..generation..": "..fitness)
	population:Evolve()
end

local best = population:GetBestNetworks(1)[1]
print(best:Forward(x)[1])