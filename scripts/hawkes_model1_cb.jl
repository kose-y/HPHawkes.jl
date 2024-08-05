using OpenCL

device = cl.devices()[1]
ctx    = cl.Context(device)
queue  = cl.CmdQueue(ctx)
#device, ctx, queue = cl.create_compute_context() # a simple context. defaults to the first GPU. 

using HPHawkes
using GeoStats, GeoIO
# import GLMakie as Mke
using Shapefile, GeoInterface
using RData, DataFrames
# load data
include("resample.jl")
@time df = load("dataComplete.rds")
using StableRNGs, StatsBase
RealType = Float64
m = parse(Int, ARGS[2]) # number of samples, must be a multiple of 8 for now
n_mcmc_samples = 20000
seed_mcmc = parse(Int, ARGS[1])
seed_data = 1234
idx = sort(sample(StableRNG(seed_data), 1:14506538, m, replace=false))
df_cut = df[idx, :]
locations = zeros(RealType, 2, m) # To use float8/double8 vector type
for i in 1:m
    locations[1, i] = df_cut.lat[i]
    locations[2, i] = df_cut.lon[i]
end

println("m: $m, seed_data: $seed_data, seed_mcmc : $seed_mcmc")

times = convert(Vector{RealType}, df_cut.day);
df_cut.fips[df_cut.fips .== 2261] .= 2063 # valdez to Chugash census ares
geom = GeoIO.load("cb_2021_us_county_500k/cb_2021_us_county_500k.shp")


# creates buffers and compiles kernels
d = HPHawkes.HawkesStorage{RealType}(ctx, locations, times; 
    device=device,
    queue=queue)

T = RealType
# computes loglikelihood
HPHawkes.loglik1_cb(d::HPHawkes.HawkesStorage{T}, 
    one(T), #sigmaXprec
    one(T), #tauXprec 
    one(T), #tauTprec, 
    one(T), #omega 
    one(T), #theta::T
    one(T), #mu0::T, 
    2)# dimX

function log_prior(x)
    #L = length(x)
    #- sum(x .^ 2 ./ (1:L)) / 2.0
    - sum(x .^ 2) / 2.0
end

function log_likelihood(d::HPHawkes.HawkesStorage{T},x) where T
#     T = RealType
    x_ = exp.(x)
# computes loglikelihood
    HPHawkes.loglik1_cb(d, 
    convert(T, x_[1]), #sigmaXprec
    convert(T, 1/sqrt(3531905)), #tauXprec  # land area of US.
    convert(T, x_[2]), #tauTprec, 
    convert(T, x_[3]), #omega 
    convert(T, x_[4]), #theta::T
    convert(T, x_[5]), #mu0::T, 
    2)
#     0
end

function log_posterior(d, x)
    log_prior(x) + log_likelihood(d,x)
end

function delta(n)
    n^(-0.5)
end

function mh(data, iterations,parameter_count,target=0.5;
    up_dat=100, sigma_init_scale=nothing, init=zeros(parameter_count), rng=Random.GLOBAL_RNG, 
    fix = Int[], out_sample="samples.csv", out_ll="likelihoods.csv")
    if sigma_init_scale === nothing 
        sigma       = ones(parameter_count)         # proposal standard deviation
    else
        sigma = sigma_init_scale
    end
    #d           = Normal(0.0, sigma) # proposal distribution
    locations = zeros(RealType, 2, m)
    io_sample = open(out_sample, "w")
    io_ll = open(out_ll, "w")
    adapt_bound = 10
    accept      = zeros(Int, parameter_count)
    props       = zeros(Int, parameter_count)
    accept_cum      = zeros(Int, parameter_count)
    props_cum       = zeros(Int, parameter_count)
    acceptances = 0
    if !isnothing(up_dat)
        resample!(df_cut, geom, rng)
        for i in 1:m
            locations[1, i] = df_cut.lat[i]
            locations[2, i] = df_cut.lon[i]
        end
        cl.copy!(data.queue, data.locations_buff, locations)
    end
    write(io_sample, join(["x$i" for i in 1:parameter_count], ",") * "\n")
    write(io_ll, "loglikelihood\n")
    states = zeros(iterations,parameter_count)
    states[1, :] .= init
    likelihoods = zeros(iterations)
    curr_log_prob = log_posterior(data, states[1,:])
    likelihoods[1] = curr_log_prob
    choice_set = setdiff(1:parameter_count, fix)
    write(io_sample, join(["$s" for s in states[1, :]], ",") * "\n")
    write(io_ll, "$(curr_log_prob)\n")
    for i = 2:iterations
        p          = rand(rng, choice_set)      # choose parameter to update
        prop_state = states[i-1,:]   
        d          = Normal(0.0, sigma[p])        
        prop_state[p] += rand(rng, d,1)[1]                 # proposal
        prop_log_prob = log_posterior(data, prop_state) # get proposal target value

        u = log(rand(rng, 1)[1])
        mh_ratio = prop_log_prob - curr_log_prob
        if u < mh_ratio
            states[i,:]   .= prop_state
            accept[p]  += 1
            accept_cum[p] += 1
            acceptances  += 1
            curr_log_prob = prop_log_prob
        else
            states[i,:] .= @view(states[i-1,:])
        end
        likelihoods[i] = curr_log_prob
        props[p] += 1
        props_cum[p] += 1

        write(io_sample, join(["$s" for s in states[i, :]], ",") * "\n")
        write(io_ll, "$(curr_log_prob)\n")

        if mod(props[p], adapt_bound) == 0 
            accept_ratio = accept[p] / adapt_bound
            accept_ratio /= target
            if accept_ratio > 2.0
                accept_ratio = 2.0
            elseif accept_ratio < 0.5
                accept_ratio = 0.5
            end
            sigma[p] *= accept_ratio

            accept[p] = 0
            props[p]  = 0
        end

        if mod(i, 100) == 0
            print(i, " ", sigma, "\n")
            println("Acceptance rate: ", acceptances/(i-1))
            println("Acceptance rate by param: ", accept_cum ./ props_cum)
            println("ll: ", likelihoods[i])
            flush(stdout)
            flush(io_ll)
            flush(io_sample)
        end        
        if mod(i, 20) == 0
            println("current ESS: $([ess(states[1:i, j]) for j in 1:parameter_count])")
            flush(stdout)
            flush(io_ll)
            flush(io_sample)
        end

        if !isnothing(up_dat) && i % up_dat == 0
            resample!(df_cut, geom, rng)
            for i in 1:m
                locations[1, i] = df_cut.lat[i]
                locations[2, i] = df_cut.lon[i]
            end
            cl.copy!(data.queue, data.locations_buff, locations)
            curr_log_prob = log_posterior(data, states[i, :])
        end
    end
    println("Acceptance rate: ", acceptances/(iterations-1))
    close(io_sample)
    close(io_ll)
    states, likelihoods
end

using Distributions, Random, OpenCL, RData, DataFrames, StableRNGs, MCMCDiagnosticTools

@time output, likelihoods = mh(d, 1, 5; sigma_init_scale = [0.066, 0.04, 0.047, 0.042, 0.03], 
    init = [2.887874555211505,2.642478497601152,0.5771171460454807,-1.1322555774716694,-0.34242143082969173], rng=StableRNG(98765))
sigma_init_scale = [0.02, 0.015, 0.04, 0.02, 0.007]
init = RealType[1.9037795056869586,0.9284792623957603,2.335945127406837,0.001946839079597651,-5.398866098357411]
init = RealType[2.887874555211505,2.642478497601152,0.5771171460454807,-1.1322555774716694,-0.34242143082969173] # added line
init = RealType[-4.271055374906369,2.505017688169231,-1.1917072802049045,-1.908899301310454,-0.056930439918434037] # added line
init = RealType[-2.5299629965244947,-1.1659843800475833,3.4439313657252537,-0.0013794908080485375,-9.074269201937454]# added line
init = RealType[-0.38622952557021784,-0.10342191244637755,3.469222356404636,-0.0008099222965228196,-8.081485856773918]# added line
rng = StableRNG(seed_mcmc)
for p in 1:length(init)
    d          = Normal(0.0, sigma_init_scale[p])
    init[p] += rand(rng, d,1)[1]       
end
@time output, likelihoods = mh(d, n_mcmc_samples, 5, 0.5; 
    sigma_init_scale = [0.02, 0.005, 0.04, 0.02, 0.007], 
    # previous run:[3.3336567063789873, 2.5326886447474943, 2.8378434523074203, 0.26308791862540115, -1.6241749548949507, -0.17847108531786635]
    init = init, #[3.523848718296068,2.7566077282597305,2.905555023402919,0.24415411341086493,-1.7860446799481307,-0.14859995537980203], 
    rng=rng,
    fix=Int[],
    out_sample="samples_$(m)samples_$(n_mcmc_samples)_$(seed_data)_$(seed_mcmc)_model1_cb.csv",
    out_ll = "likelihoods_$(m)samples_$(n_mcmc_samples)_$(seed_data)_$(seed_mcmc)_model1_cb.csv",
    up_dat=n_mcmc_samples + 3)
# using CSV
# CSV.write("samples_1msamples_20000.csv", DataFrame(output, :auto))

# CSV.write("likelihoods_1msamples_20000.csv", DataFrame(reshape(likelihoods, :, 1), :auto))
