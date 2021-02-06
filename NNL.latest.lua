--[[


STYLE GUIDE

- PascalCase for configurations that user accesses
- this_case or lowercase for variables and private functions 
- semicolon (;) for dictionaries and commas for arrays
- Keep between 1-5 tabs, or at least try to avoid chevrons
- Prefer ternary operators for setting values unless it's more convenient to use if


A simple NN I wrote when I came across this article, just to get a better understanding:
https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

API

NNL.nn.new(settings)
NNL.nn.fromJSON(string)
nn:toJSON()

nn:Forward(array inputs)
nn:Cost(array expected)
nn:Learn()

]]


-- NN Library

local NNL = {}

local function tshallowcopy(t)
	local nt = {}
	for i, v in pairs (t) do
		nt[i] = v
	end
	return nt
end

local function tdeepcopy(orig, copies) -- thanks lua users wiki
    copies = copies or {}
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        if copies[orig] then
            copy = copies[orig]
        else
            copy = {}
            copies[orig] = copy
            for orig_key, orig_value in next, orig, nil do
                copy[tdeepcopy(orig_key, copies)] = tdeepcopy(orig_value, copies)
            end
            setmetatable(copy, tdeepcopy(getmetatable(orig), copies))
        end
    else 
        copy = orig
    end
    return copy
end

local debugmode = false
function NNL.dprint(...)
	if debugmode then print(...) end
end
function NNL.tprint(t)
	for i, v in pairs(t) do
		local Type = type(v)
		if Type=="table" then
			print(i); NNL.tprint(v)
		else
			print(i,v)
		end
	end
end


-- NEURAL NETWORK

NNL.nn = {}
local nn = NNL.nn
nn.__index = nn

-- JSON API, needs another library
local jsonconfig = {
	encode = function(...) return game.HttpService:JSONEncode(...) end;
	decode = function(str) return game.HttpService:JSONDecode(str) end;
	compress = nil;
	decompress = nil;
}

function nn.fromJSON(str)
	str = jsonconfig.decompress and jsonconfig.decompress(str) or str 
	local self = jsonconfig.decode(str)
	self._lastinputs = {}
	for _,layer in ipairs (self.layers) do
		for _, node in ipairs(layer) do
			node.o, node.d = 0, 0
		end
	end
	return self
end

function nn:toJSON()
	local copy = tshallowcopy(self)
	-- Reset the last output and deltas
	copy._lastinputs = nil
	for _,layer in ipairs (self.layers) do
		for _, node in ipairs(layer) do
			node.o, node.d = nil, nil
		end
	end
	local json = jsonconfig.encode(copy)
	json = jsonconfig.compress and jsonconfig.compress(json) or json 
	return json
end

-- nn

local activation_functions = {
	sigmoid = function(x, deriv)
		return deriv and 1/(1+math.exp(-x)) * (1-1/(1+math.exp(-x)))
		or 1/(1+math.exp(-x))
	end;
	tanh = function (x, deriv)
		return deriv and math.pow(math.tanh(x),2)
		or math.tanh(x)
	end;
	relu = function (x, deriv)
		return deriv and (x>0 and 1 or x==0 and 0.5)
		or math.max(0, x)
	end;
	leakyrelu = function(x, deriv)
		return deriv and (x>=0 and 1 or 0.1)
		or math.max(0.1*x, x)
	end;
}

--[[
hmm what can this be
local optimizers = {
	stochasticgradientdescent = function(node)
		return change
	end;
	momentum = function(node)
		return change
	end;
	adam = function(node)
		return change
	end;
}]]

local function random01()
	local pv = 1000000000000000
	return math.random(0,pv)/pv
end

local function blanknode()
	return {w={}; d=0; o=0}
end

function nn.new(newsettings)
	local self = {} 
	setmetatable(self,nn)
	newsettings = newsettings or {}
	self.settings = {
		OutputNodes = newsettings.OutputNodes or 1;
		InputNodes = newsettings.InputNodes or 3;
		HiddenLayers = newsettings.HiddenLayers or 2;
		HiddenNodes = newsettings.HiddenNodes or 2;
		HiddenActivation = newsettings.HiddenActivation or "sigmoid";
		OutputActivation = newsettings.InputActivation or "sigmoid";
		LearningRate = newsettings.LearningRate or 0.2;
	}
	self._lastinputs = {}
	--[[
		inputs = {1,2,3,4,5}
			w: the synapse weight, b: the bias, d: the last delta
		nn.layers = {
			{{w={0,1,1,1,1}; b=10}, {w={0,1,1,1,1}}, {w={0,1,1,1,1}}},
			{{w={0,1,1,1,1}; b=10}, {w={0,1,1,1,1}}, {w={0,1,1,1,1}}}
		}
		#weights == #inputs
	]]
	self.layers = {}	  -- Stores the layers to the left of a layer
	for n_syn = 1, self.settings.HiddenLayers do 
		local hiddenlayer = {}
		for n_node = 1, self.settings.HiddenNodes do 
			local node = blanknode()
			-- #weights should be #activations
			local weights_needed = n_syn==1 and self.settings.InputNodes or self.settings.HiddenNodes
			for n_w = 1, weights_needed do 
				node.w[n_w] = random01()
			end
			table.insert(hiddenlayer, node)
		end
		table.insert(self.layers, hiddenlayer)
	end
	local outputlayer = {}
	for n_node = 1, self.settings.OutputNodes do 
		local node = blanknode()
		local weights_needed = self.settings.HiddenNodes
		for n_w = 1, weights_needed do 
			node.w[n_w] = random01()
		end
		table.insert(outputlayer, node)
	end
	table.insert(self.layers, outputlayer)

	return self
end

function nn:_activate(weights, inputs)	  
	local activation = 0 
	-- scalar (inputs .  weights)
	for i=1, #inputs do 
		local wa = weights[i] * inputs[i]
		activation = activation + wa
	end
	return activation
end

function nn:Forward(inputs)
	local outputs = {}
	self._lastinputs = inputs
	local last_activations = inputs  -- Use this instead of outputs for readability
	for i, layer in ipairs(self.layers) do 
		local activations = {}
		for i2, node in ipairs(layer) do 
			local activation = self:_activate(node.w, last_activations)
			local a_function = 
				i==#self.layers and activation_functions[self.settings.OutputActivation]
				or activation_functions[self.settings.HiddenActivation]
			activation = a_function(activation)	 -- sig (wa)
			activations[i2] = activation	
			node.o = activation					  -- Keep the output for backpropagation
		end
		last_activations = activations				  -- forward it to the next layer
	end
	outputs = last_activations
	return outputs
end


--	local deriv_function = activation_functions[self.HiddenActivation]
--	local error = (correct - output) 
--	local delta = error * deriv_function(output)

--[[

1. Forward propagate. l1 = activation(dot(layer0, layer0))
2. Calculate the error square. error = (y - l1)^2
3. Find the needed delta. 
	lOutput_deltas = error * activation(lOutput, deriv)
	lHidden_deltas = 
4. Update weights in the layer. layer0 = dot(l0 in columns, l1_deltas)

df = f * (1-f)

		nn.layers = {
			{{w={0,1,1,1,1}; b=10; o=0.4; d=0.2}, {w={0,1,1,1,1}}, {w={0,1,1,1,1}}},
			{{w={0,1,1,1,1}; b=10; o=0.4; d=0.2}, {w={0,1,1,1,1}}, {w={0,1,1,1,1}}}
		}

]]

function nn:Cost(expected)
	for i = #self.layers, 1, -1 do 
		local layer = self.layers[i]
		local errors = {}
		-- Calculate the deltas for each node
		if i==#self.layers then 
			-- this is an output layer
			local error = 0 -- Total error for this layer's synapse
			for i2, thisnode in ipairs(layer) do 
				error = error + expected[i2]-thisnode.o
			end
			table.insert(errors, error)
		else
			-- A hidden layer
			for i2, thisnode in ipairs (layer) do 
				local right_layer = self.layers[i+1] -- Get the layer to the right
				local error = 0						 -- Total error of the right layer, the 'output'
				for i2, rightnode in ipairs (right_layer) do 
					-- Getting the weighted errors of each node in the output
					-- error = connecting weight x output node error x slope
					error = error + rightnode.w[i2] * rightnode.d
				end
				table.insert(errors, error)
			end
		end
		-- Finally, apply the deltas
		for i2, node in ipairs (layer) do 
			local deriv_function = 
				i==#self.layers and activation_functions[self.settings.OutputActivation]
				or activation_functions[self.settings.HiddenActivation]
			node.d = errors[i2] * deriv_function(node.o)
		end
	end
end

function nn:Learn()
	-- Use after nn:Cost()
	-- This will apply all the deltas to the weights
	-- If this was called before a forward propagation, jail

	local inputs = self._lastinputs
	for i, layer in ipairs(self.layers) do
		-- New = Current + dot (inputs . deltas) 
		for i2, node in ipairs(layer) do
			for i3, w in ipairs (node.w)do 
				local learning_rate = self.settings.LearningRate
				local change = inputs[i2] * node.d * learning_rate
				node.w[i3] = w + change
			end
		end
		-- Set their last output as the next layer's input
		inputs = {}
		for i2, node in ipairs(layer) do
			inputs[i2] = node.o
		end
	end
end


--[[ Test this

math.randomseed(os.time())
local NNL = require "NNL"
local nn = NNL.nn.new()

local y = {0.5}
local x = {1,0,1}

local epochs = 1000
for i=1, epochs do 
	nn:Forward(x)
	nn:Cost(y)
	if i % 10 ==0 then
		nn:Learn() 
	end
end

print(nn:Forward(x)[1])

-- JSON API

local saved = nn:toJSON()
local loaded = NNL.nn.fromJSON(saved)
print(nn:Forward(x)[1])

]]

-- GENETIC ALGORITHM
-- Meant to be in the same file as NNL, but separated so I can review everything individually

NNL.genalg = {}
local genalg = NNL.genalg
--[[ 
	From Jobro13's Neuro (https://github.com/jobro13/Neuro/blob/master/Neuro/src/genetics.lua)
	For glossary: 
	Weight -> Gene
	Node -> Chromosome
]]
genalg.Settings = {
	CrossoverChance = 0.7;
	MutateChance = 0.3;
	MaxMutateChange = 0.2;
	BestNetworksCount = 4;
}

function genalg.Crossover(node1, node2)
	if not (random01() <= genalg.Settings.CrossoverChance) then
		return node1, node2
	end
	local newnode = blanknode()
	local nweights = #node1.w
	local crossoverpoint = math.random(2, math.max(2, nweights-1))	-- At what point do the weights switch
	for i=1, nweights do
		if i < crossoverpoint then -- Use node1's weights
			newnode.w[i] = node1.w[i]
		else -- Use node2's weights
			newnode.w[i] = node2.w[i]
		end
	end
	return newnode
end

function genalg.Mutate(node)
	local newnode = blanknode()
	for i, weight in ipairs (node.w) do
		if random01() <= genalg.Settings.MutateChance then
			newnode.w[i] = genalg.Settings.MaxMutateChange * random01()
		else
			newnode.w[i] = weight
		end
	end
	return newnode
end

function genalg.Meiosis(node1, node2)
	-- Crossover, then mutate
	local newnode1, newnode2 = genalg.Crossover(node1, node2), genalg.Crossover(node1, node2)
	newnode1, newnode2 = genalg.Mutate(newnode1), genalg.Mutate(newnode2)
	return newnode1, newnode2
end

function genalg.Breed(network1, network2)
	for i, layer1 in ipairs (network1.layers) do
		local layer2 = network2.layers[i]
		for i2, node1 in ipairs(layer1) do
			local node2 = layer2[i2]
			local newnode1, newnode2 = genalg.Meiosis(node1, node2)
			layer1[i2] = newnode1;
			layer2[i2] = newnode2
		end
	end
end

genalg.population = {}
local population = genalg.population
population.__index = population

function population.new(networks, networksettings)
	local self = {
		Networks = {};
		Size = 0;
	}
	setmetatable(self,population)
	local Type = type(networks)
	if Type=="number" then
		local t = {}
		for i=1, networks do 
			table.insert(t, NNL.nn.new(networksettings))
		end
		networks = t
		self.Size = #networks
	end
	for _, network in pairs (networks) do 
		network.Fitness = network.Fitness or 0;	-- Add a new network property for fitness
	end
	self.Networks = networks
	return self
end

function population:Insert(network)
	table.insert(self.Networks, network)
end

function population:GetTotalFitness()
	local total = 0
	for i, network in ipairs (self.Networks) do
		total = total + network.Fitness
	end
	return total
end

function population:GetBestNetworks(n)
	n = n or genalg.Settings.BestNetworkCount
	table.sort(self.Networks, function(a,b)
		return b.Fitness < a.Fitness
	end)
	local best = {}
	for i=1, n do 
		best[i] = self.Networks[i]
	end
	return best
end

function population:Evolve(bestnetworks)
	bestnetworks = bestnetworks or self.Networks
	local newbrains = {}
	table.sort(self.Networks, function(a,b)
		return b.Fitness < a.Fitness
	end)
	table.insert(newbrains, tdeepcopy(self.Networks[1]))
	for i=1, #self.Networks-1, 2 do
		local network1, network2 = bestnetworks[i], bestnetworks[i+2]
		if not network2 then 
			table.insert(newbrains, network1)
			break
		end
		genalg.Breed(network1, network2)
		table.insert(newbrains, network1)
		table.insert(newbrains, network2)
	end
	self.Networks = newbrains
end

--[[ Test this

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

]]

return NNL
