using CSV #Add CSV package
using Flux
using Flux: onehot, chunk, batchseq, throttle, logitcrossentropy
using DataFrames
using Parameters: @with_kw
using Lathe.preprocess: TrainTestSplit

#Set WD
cd("/Users/mousaghannnam/Documents/Data Science/sumr2020/")

# Hyperparameter arguments
@with_kw mutable struct Args
    lr::Float64 = 1e-2	# Learning rate
    seqlen::Int = 50	# Length of batchseqences
    nbatch::Int = 50	# number of batches text is divided into
    throttle::Int = 1	# Throttle timeout
end

function getData()
	println("hello there! #1")
	#Load Data
	myData = CSV.read("./very_fake_diet_data.csv") |> DataFrame!
    println("hello there! #2")
	#Function for creating dictionary of word frequencies
		function Counter(d::DataFrame)
			outp = Dict{String, Int64}()
			for i = 1:size(myData,1)
				for j in split(myData[i,1])
					if haskey(outp, j)
						outp[j] += 1
					else
						outp[j] = 1
					end
				end
			end
			return outp
		end
		println("hello there! #3")
		lexiconFreq = Counter(myData) #Table of word frequencies

		#Obtain the unique words
		uniqueWords = collect(keys(lexiconFreq))
		N = length(uniqueWords)

		#Create oneHotVectors from these words
		oneHotWords = map(word -> Flux.onehot(word, uniqueWords), uniqueWords)

		#Create a dictionary of words to oneHotVectors
		oneHotDict = Dict(uniqueWords .=> oneHotWords)
		oneHotDict["1"] #Test

		#Create a vector of one hot vectors, for each sentence
		function getData(myData)
		    x = Tuple{Flux.OneHotMatrix,Int64}[]
		    for i = 1:size(myData, 1)
		        tempSentence = split(myData[i, 1])
		        tempMatrix = Flux.onehotbatch(tempSentence, uniqueWords)
		        tempTup = (tempMatrix, myData[i, 3])
		        push!(x, tempTup)
		    end
			return x
		end

		#Create our input vector
		inputData = getData(myData)

		#Split up training and test data
		#Creating df with just 1 and 3rd row, for binary task
		trainD,testD = TrainTestSplit(inputData, 0.9)
		trainData = inputData[trainD]
		testData = inputData[testD]

		return trainData, testData, N
end

function build_model(N)
    return Chain(
            LSTM(N, 128),
            LSTM(128, 128),
            Dense(128, N))
end


# Function to construct model
function train(; kws...)
    # Initialize the parameters
    args = Args(; kws...)

    # Get Data
    trainData, testData, N = getData()
	print(size(trainData))

    # Constructing Model
    m = build_model(N)

    function loss(xs, ys)
      l = sum(logitcrossentropy.(m.(xs), ys))
      return l
    end
	println("hello there! #4")
    ## Training
    opt = ADAM(args.lr)
    tx, ty = testData[5]
    evalcb = () -> @show loss(tx, ty)
	println("hello there! #5")
	p = params(m)
	Flux.train!(loss,p, trainData, opt)
	println("hello there! #6")
end

m = train()




#Load Data
myData = CSV.read("./very_fake_diet_data.csv") |> DataFrame!
show(myData, true) #shows all of the columns of the dataframe
#print(myData)

#Function for creating dictionary of word frequencies
	function Counter(d::DataFrame)
		outp = Dict{String, Int64}()
		for i = 1:size(myData,1)
			for j in split(myData[i,1])
				if haskey(outp, j)
					outp[j] += 1
				else
					outp[j] = 1
				end
			end
		end
		return outp
	end

lexiconFreq = Counter(myData) #Table of word frequencies
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
oneHotDict["1"] #Test

#Create a vector of one hot vectors, for each sentence
function getData(myData)
    x = Tuple{Flux.OneHotMatrix,Int64}[]
    for i = 1:size(myData, 1)
        tempSentence = split(myData[i, 1])
        tempMatrix = Flux.onehotbatch(tempSentence, uniqueWords)
        tempTup = (tempMatrix, myData[i, 3])
        push!(x, tempTup)
    end
	return x
end

#Create our input vector
inputData = getData(myData)

#Split up training and test data
#Creating df with just 1 and 3rd row, for binary task
trainD,testD = TrainTestSplit(inputData, 0.9)
trainData = inputData[trainD]
testData = inputData[testD]

N = length(uniqueWords)

m = Chain(LSTM(N, 128), LSTM(128, 128),Dense(128, N))

function build_model(N)
    return Chain(
            LSTM(N, 128),
            LSTM(128, 128),
            Dense(128, N))
end


function loss(xs, ys)
  l = sum(logitcrossentropy.(m.(xs), ys))
  return l
end







## Training
η, β = 0.001, (0.9, 0.999)
opt = ADAM(η, β)
tx, ty = testData[5]
evalcb = () -> @show loss(tx, ty)
Flux.train!(loss, params(m), trainData, opt, cb = throttle(evalcb, 15))




##  Quantitative Output
