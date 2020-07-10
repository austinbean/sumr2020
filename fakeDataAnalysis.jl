using CSV #Add CSV package
using Flux
using DataFrames

data = CSV.read("/very_fake_diet_data.csv") |> DataFrame!
show(data, true) #shows all of the columns of the dataframe
print(data)

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

lexiconFreq = Counter(data)
#Question #1: After assigning the counter function to a variable, and I pull out the "Key"  property, I get a bunch of undefined values...where is that coming from?
#Quesiton #2: How do I import functions from other documents into this one?

#Examine the most common words
sort(collect(lexiconFreq), by = tuple -> last(tuple), rev=true)

#Obtain the unique words
uniqueWords = collect(keys(lexiconFreq))

#Create oneHotVectors from these words
oneHotWords = map(word -> Flux.onehot(word, uniqueWords), uniqueWords)

#Create a dictionary of words to oneHotVectors
oneHotDict = Dict(uniqueWords .=> oneHotWords)
oneHotDict["1"]


    #BELOW CODE IS JUNK, PLS IGNORE
    # oneHotSentences = Array{Array, Flux.OneHotVector}
    # oneHotSentences = [Flux.OneHotVector[]]
    # Vector{Flux.OneHotVector}
    # Flux.OneHotVector[]
    # oneHotSentences = Vector{Vector{Flux.OneHotVector}}
    # typeof(oneHotSentences[1])
    # for i = 1:size(data,1)
    #     for j in split(data[i,1])
    #         println(i)
    #     end
    # end
    #
    # for i = 1:size(data,1)
    #     tempVectOHV = Flux.OneHotVector[]
    #     for j in split(data[i,1])
    #         oneHotSentences[i].push(oneHotDict[j])
    #     end
    # end
    # tempVectOH1V = Bool[]
    # tempVectOHV1 = Flux.OneHotVector[]

#Create a vector of one hot vectors, for each sentence
#function prepareData()
    oneHotSentences = Vector{Flux.OneHotVector}[]
    for i = 1:size(data, 1)
        tempVectOHV = Flux.OneHotVector[]
        for j in split(data[i, 1])
            push!(tempVectOHV, oneHotDict[j])
        end
        append!(oneHotSentences, tempVectOHV)
    end
    zip(oneHotSentences, data[:,3])
    #return
#end

#For loop that outputs the oneHotVector for word in a sentence
    for j in split(data[1,1])
        oneHotDict[j]
    end

    #For loop that outputs the oneHotVector for word in a sentence
        for j in split(data[1,1])
            push!(tempVectOHV, oneHotDict[j])
        end

#Build an RNN model
Chain(LSTM(N, 128), LSTM(128, 128), Dense(128, N))
