--[[

NNL 
Author: g_captain
API Reference (06/02/2021 DD/MM/YYYY)

function NNL.dprint(...)
function NNL.tprint(t)

# NEURAL NETWORK

NNL.nn = {}

Activation functions:
	sigmoid,
	tanh,
	relu,
	leakyrelu

JSON configurations:
	function encode(str)
	function decode(str)
	function compress(str)
	function decompress(str)

function nn.fromJSON(str)
function nn:toJSON()

function nn.new(newsettings)
	Default settings:
		OutputNodes = 1;
		InputNodes = 3;
		HiddenLayers = 2;
		HiddenNodes = 2;
		HiddenActivation = "sigmoid";
		OutputActivation = "sigmoid";
		LearningRate = 0.2;
	
function nn:Forward(inputs)
function nn:Cost(expected)
function nn:Learn()

# GENETIC ALGORITHM

NNL.genalg:
genalg.Settings = {
	CrossoverChance = 0.7;
	MutateChance = 0.3;
	MaxMutateChange = 0.2;
	BestNetworksCount = 4;
}
function genalg.Crossover(node1, node2)
function genalg.Mutate(node)
function genalg.Meiosis(node1, node2)
function genalg.Breed(network1, network2)

genalg.population:
function population.new(networks, networksettings)
function population:Insert(network)
function population:GetTotalFitness()
function population:GetBestNetworks(n)
function population:Evolve(bestnetworks)

]]

local NNL = {}

local function tshallowcopy(t)
	local nt = {}
	for i, v in pairs (t) do
		nt[i] = v
	end
	return nt
end

local function tdeepcopy(orig, copies)
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

-- local json = require "modules/json"
local jsonconfig = {
	encode = function(s) return game.HttpService:JSONEncode(s) end; --json.encode;
	decode = function(s) return game.HttpService:JSONDecode(s) end; --json.decode;
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
	setmetatable(self, nn)
	return self
end

function nn:toJSON()
	local copy = tdeepcopy(self)
	setmetatable(copy, {})
	-- Reset all the backprop node data
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

local function random01()
	local pv = 100000000
	return math.random(0, pv)/pv
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
	self.layers = {}	
	for n_syn = 1, self.settings.HiddenLayers do 
		local hiddenlayer = {}
		for n_node = 1, self.settings.HiddenNodes do 
			local node = blanknode()
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
	for i=1, #inputs do 
		local wa = weights[i] * inputs[i]
		activation = activation + wa
	end
	return activation
end

function nn:Forward(inputs)
	local outputs = {}
	self._lastinputs = inputs
	local last_activations = inputs  
	for i, layer in ipairs(self.layers) do 
		local activations = {}
		for i2, node in ipairs(layer) do 
			local activation = self:_activate(node.w, last_activations)
			local a_function = 
				i==#self.layers and activation_functions[self.settings.OutputActivation]
				or activation_functions[self.settings.HiddenActivation]
			activation = a_function(activation)	 
			activations[i2] = activation	
			node.o = activation					 
		end
		last_activations = activations	
	end
	outputs = last_activations
	return outputs
end

function nn:Cost(expected)
	for i = #self.layers, 1, -1 do 
		local layer = self.layers[i]
		local errors = {}
		if i==#self.layers then 
			local error = 0
			for i2, thisnode in ipairs(layer) do 
				error = error + expected[i2]-thisnode.o
			end
			table.insert(errors, error)
		else
			for i2, thisnode in ipairs (layer) do 
				local right_layer = self.layers[i+1] 
				local error = 0	
				for i2, rightnode in ipairs (right_layer) do 
					error = error + rightnode.w[i2] * rightnode.d
				end
				table.insert(errors, error)
			end
		end
		for i2, node in ipairs (layer) do 
			local deriv_function = 
				i==#self.layers and activation_functions[self.settings.OutputActivation]
				or activation_functions[self.settings.HiddenActivation]
			node.d = errors[i2] or errors[1] * deriv_function(node.o)
		end
	end
end

function nn:Learn()
	local inputs = self._lastinputs
	for i, layer in ipairs(self.layers) do
		for i2, node in ipairs(layer) do
			for i3, w in ipairs (node.w)do 
				local learning_rate = self.settings.LearningRate
				local change = inputs[i2] * node.d * learning_rate
				node.w[i3] = w + change
			end
		end
		inputs = {}
		for i2, node in ipairs(layer) do
			inputs[i2] = node.o
		end
	end
end

-- GENETIC ALGORITHM

NNL.genalg = {}
local genalg = NNL.genalg
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
	local crossoverpoint = math.random(2, math.max(2, nweights-1))
	for i=1, nweights do
		if i < crossoverpoint then 
			newnode.w[i] = node1.w[i]
		else 
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
		network.Fitness = network.Fitness or 0;
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

return NNL
