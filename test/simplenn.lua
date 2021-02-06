-- Simple NN test
-- The goal: Get 0.5

math.randomseed(os.time())
local NNL = require "NNL"
local nn = NNL.nn.new()

local y = {0.5}
local x = {1,0,1}

local epochs = 1000
for i=1, epochs do 
	nn:Forward(x)
	nn:Cost(y)
	if i % 10 == 0 then
		nn:Learn() 
	end
end

print(nn:Forward(x)[1])

-- JSON API
local saved = nn:toJSON()
local loaded = NNL.nn.fromJSON(saved)
print(nn:Forward(x)[1])