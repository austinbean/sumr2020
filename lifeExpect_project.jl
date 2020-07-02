using CSV #Add CSV package
using Flux
using Flux: throttle
using Flux: accuracy
using GLM # not sure if i really need this right now
using Tracker
using Plots
using DataFrames
using StatsBase

df = CSV.File("/Users/mousaghannnam/Documents/Data Science/julia_test_folder/lifeExpect.csv") |> DataFrame!

filter_data = filter( row -> !ismissing(row.GDP), df)
filter_data = filter( row -> !ismissing(row."infant deaths"), filter_data)
#Gather data from df in Array of Float form so that sigoid function works
gdp = convert(Array{Union{Float32, Missing}, 1}, filter_data[:, 17])
deaths = convert(Array{Union{Float32, Missing}, 1}, filter_data."infant deaths")
d1 = Flux.Data.DataLoader(gdp, deaths)

#Take 70% of the data to train, 30% for test data
training_gdp = convert(Array{Union{Float32, Missing},1}, filter_data[1:1737,17])
training_deaths = convert(Array{Union{Float32, Missing},1}, filter_data[1:1737,6])
test_gdp = convert(Array{Union{Float32, Missing},1}, filter_data[1737:2490,17])
test_deaths = convert(Array{Union{Float32, Missing},1}, filter_data[1737:2490,6])

d_train = Flux.Data.DataLoader(training_gdp, training_deaths)
d_test = Flux.Data.DataLoader(test_gdp, test_deaths)

data_train = zip(training_gdp,training_deaths)
data_test = zip(test_gdp, test_deaths)

## Build and train the model
model = Dense(1,1,σ)
loss_f(x,y) = Flux.mse(model([x]),y)
opt = Descent(0.1)
evalcb = () -> @show(sum([loss_f(i[1],i[2]) for i in data_train])) #Function for outputting the sum of the total loss_f
evalcb = () -> @show(sum([loss_f(i[1],i[2]) for i in data_train])/length(data_train)) #Function for outputting the MSE

Flux.train!(loss_f, Flux.params(model), data_train , opt, cb = throttle(evalcb, 0.5)) #Prints the total loss at the end of training
#Flux.train!(loss_f, Flux.params(model), data_train , opt, cb = sum([loss_f(i[1],i[2]) for i in data_train])) #Prints the total loss at the end of training

#MSE for the TEST daata...
return_MSE(data_train)

#Create function for plotting line of best fit
model_line(x)= model.W[1]*x + model.b[1]

#Plot training data, fitted with MSE
plotly() #Create backend
scatter(training_gdp, training_deaths, markersize = 2, label = "Deaths") #Add only test data
xlabel!("GDP")
ylabel!("Deaths")
title!("GDP vs Deaths")
plot!(model_line, linewidth = 1.5, label = "MSE", color= :red, xlims = (0,maximum(test_gdp)), ylims = (0, maximum(test_deaths))) #Regression line

#Plot test data, fitted with MSE (MSE calculated from training data)
plotly() #Create backend
scatter(test_gdp, test_deaths, markersize = 2, label = "Deaths") #Add only test data
xlabel!("GDP")
ylabel!("Deaths")
title!("GDP vs Deaths")
plot!(model_line, linewidth = 1.5, label = "MSE", color= :red, xlims = (0,maximum(test_gdp)), ylims = (0, maximum(test_deaths))) #Regression line


## Another function that returns MSE
function return_MSE(dataset) #The input "dataset"  is a zipped tuple! Same as the Flux.train! fx
    #loss = 0
    #pred_array = Float32[]
    total_loss = sum([loss_f(i[1],i[2]) for i in dataset])
    MSE = total_loss / length(dataset)
    return (MSE) #Flux's MSE fx is being called one step at a time, so we have to find average of total loss at end
end

#Samples entire dataset for 30% of samples, and returns MSE
#Returns the MSE over n number of runs
function random_runs(n_runs)
    runResults = Float32[]
    for i in 1:n_runs
        rand_indice = sample(1:length(gdp), 790, replace=false)
        random_data = zip(gdp[rand_indice], deaths[rand_indice])
        append!(runResults, return_MSE(random_data))
    end
    return(runResults)
end

#Determines the average of the MSE over n number of runs
n_runs = 10
sum(random_runs(n_runs))/n_runs

# Attempt to take a bunch of steps of random subsets of data to train the model on, just to compare their total loss
# for step in 1:10
#     rand_indice = sample(1:length(gdp), 1737, replace=false)
#     Flux.train!(loss_f, Flux.params(model), data_train , opt, cb = throttle(evalcb, 0.5)) #Prints the total loss at the end of training
# end


## Trying to generate plot of predicted vs actual values
pred_array = Float32[]
for i in test_gdp
    append!(pred_array, model([i]))
end

performance_testdf = DataFrame(y_actual = test_deaths, y_predicted = pred_array )
#Did I do something wrong? Or, are these my actual results
test_plot = scatter(performance_testdf[!,:y_actual],performance_testdf[!,:y_predicted], title = "Predicted value vs Actual value on Test Data", ylabel = "Predicted value", xlabel = "Actual value", legend = false)

## Trying to create a vector of the outputs of the model...
#Is there any easier way??
pred_array = Float32[]
for i in training_gdp
    append!(pred_array, model([i]))
end
