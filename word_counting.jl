# word counting exercise


using CSV
using DataFrames

data = CSV.read("very_fake_diet_data.csv") |> DataFrame

function Counter(d::DataFrame)
	outp = Dict{String, Int64}()
	for i = 1:size(data,1)
		for j in split(data[i,1])
			if haskey(outp, j)
				outp[j] += 1
			else 
				outp[j] = 1 
			end 
		end 
	end 
	return outp
end 

Counter(data)