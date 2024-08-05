using OpenCL

using MPI
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)
const root = 0
device = cl.devices()[rank % 8 + 1]
ctx    = cl.Context(device)
queue  = cl.CmdQueue(ctx)
#device, ctx, queue = cl.create_compute_context() # a simple context. defaults to the first GPU. 

function partition(n, npartitions)
    part_size = n ÷ npartitions
    rem = n % npartitions
    r = vcat([0], [part_size + (i <= rem ? 1 : 0) for i in 1:npartitions])
    return cumsum(r)
end

using HPHawkes
using GeoStats, GeoIO
# import GLMakie as Mke
using Shapefile, GeoInterface
using RData, DataFrames
# load data
include("resample.jl")
@time df = load("dataComplete.rds")
using StableRNGs, StatsBase
RealType = Float32
m = 1_000_000 # number of samples, must be a multiple of 8 for now
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
popdensity = convert(Vector{RealType}, df_cut.popDens)
df_cut.fips[df_cut.fips .== 2261] .= 2063 # valdez to Chugash census ares
geom = GeoIO.load("cb_2021_us_county_500k/cb_2021_us_county_500k.shp")


# creates buffers and compiles kernels
d = HPHawkes.HawkesStorage{RealType}(ctx, locations, times, popdensity; 
    device=device,
    queue=queue)

partitions = partition(d.m, size)
# println(partitions)

T = RealType
# computes loglikelihood
HPHawkes.loglik2_cb(d::HPHawkes.HawkesStorage{T}, 
    one(T), #sigmaXprec
    one(T), #tauXprec 
    one(T), #tauTprec, 
    one(T), #omega 
    one(T), #theta::T
    one(T), #mu0::T, 
    2; # dimX
    first_idx=partitions[rank + 1],
    last_idx=partitions[rank + 2])

function log_prior(x)
    #L = length(x)
    #- sum(x .^ 2 ./ (1:L)) / 2.0
    - sum(x .^ 2) / 2.0
end

function log_likelihood(d::HPHawkes.HawkesStorage{T},x, partitions; root=0, comm=MPI.COMM_WORLD) where T
#     T = RealType
    x_ = exp.(x)
# computes loglikelihood
    local_lik = HPHawkes.loglik2_cb(d, 
    convert(T, x_[1]), #sigmaXprec is offest x_[1] + tauXprec.  This ensures that sigmaXprec is larger than tauXprec.
        convert(T, 1/sqrt(3531905)), #tauXprec 
        convert(T, x_[2]), #tauTprec, 
        convert(T, x_[3]), #omega 
        convert(T, x_[4]), #theta::T
        convert(T, x_[5]), #mu0::T, 
        2; # dimX
        first_idx=partitions[rank + 1],
        last_idx=partitions[rank + 2]
    )
    # println("rank $rank: $local_lik")
    MPI.Reduce(local_lik, +, root, comm)
end

function log_posterior(d, x, partitions; root=0, comm=MPI.COMM_WORLD)
    ll = log_likelihood(d,x, partitions; root=root, comm=comm)
    if MPI.Comm_rank(comm) == root
        return log_prior(x) + ll
    else
        return nothing
    end
end

function delta(n)
    n^(-0.5)
end

function mh(data, iterations,parameter_count,target=0.5;
    up_dat=nothing, sigma_init_scale=nothing, init=zeros(parameter_count), rng=Random.GLOBAL_RNG, 
    fix = Int[], out_sample="samples.csv", out_ll="likelihoods.csv", root=0, comm=MPI.COMM_WORLD, 
    partitions=partition(data.m, MPI.Comm_size(comm)))
    #d           = Normal(0.0, sigma) # proposal distribution
    prop_state = Array{Float64}(undef, parameter_count)
    if MPI.Comm_rank(comm) == root
        if sigma_init_scale === nothing 
            sigma       = ones(parameter_count)         # proposal standard deviation
        else
            sigma = sigma_init_scale
        end
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
        end
        write(io_sample, join(["x$i" for i in 1:parameter_count], ",") * "\n")
        write(io_ll, "loglikelihood\n")
        states = zeros(iterations,parameter_count)
        states[1, :] .= init
        prop_state .= init
        likelihoods = zeros(iterations)
    end
    MPI.Bcast!(locations, root, comm)
    cl.copy!(data.queue, data.locations_buff, locations)
    MPI.Bcast!(prop_state, root, comm)
    curr_log_prob = log_posterior(data, prop_state, partitions; comm=comm, root=root)
    if MPI.Comm_rank(comm) == root
        likelihoods[1] = curr_log_prob
        choice_set = setdiff(1:parameter_count, fix)
        write(io_sample, join(["$s" for s in states[1, :]], ",") * "\n")
        write(io_ll, "$(curr_log_prob)\n")
    end 

    for i = 2:iterations
        if MPI.Comm_rank(comm) == root
            p          = rand(rng, choice_set)      # choose parameter to update
            prop_state .= states[i-1,:]   
            d          = Normal(0.0, sigma[p])        
            prop_state[p] += rand(rng, d,1)[1]                 # proposal
        end
        MPI.Bcast!(prop_state, root, comm)
        prop_log_prob = log_posterior(data, prop_state, partitions; comm=comm, root=root) # get proposal target value
        if MPI.Comm_rank(comm) == root
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

            if mod(i, 10) == 0
                print(i, " ", sigma, "\n")
                println("Acceptance rate: ", acceptances/(i-1))
                println("Acceptance rate by param: ", accept_cum ./ props_cum)
                println("ll: ", likelihoods[i])
                flush(stdout)
                flush(io_ll)
                flush(io_sample)
            end        
            if mod(i, 20) == 0
                # println("current ESS: $([ess(states[1:i, j]) for j in 1:parameter_count])")
                # flush(stdout)
                # flush(io_ll)
                # flush(io_sample)
            end

            if !isnothing(up_dat) && i % up_dat == 0
                resample!(df_cut, geom, rng)
                for i in 1:m
                    locations[1, i] = df_cut.lat[i]
                    locations[2, i] = df_cut.lon[i]
                end
                prop_state .= states[i,:] 
            end
        end
        if !isnothing(up_dat) && i % up_dat == 0
            MPI.Bcast!(prop_state, root, comm)
            MPI.Bcast!(locations, root, comm)
            cl.copy!(data.queue, data.locations_buff, locations)
            curr_log_prob = log_posterior(data, prop_state, partitions; comm=comm, root=root)
            println("new logprob: $curr_log_prob")
        end
    end
    if MPI.Comm_rank(comm) == root
        println("Acceptance rate: ", acceptances/(iterations-1))
        close(io_sample)
        close(io_ll)
    else
        states=nothing
        likelihoods=nothing
    end
    states, likelihoods
end

using Distributions, Random, OpenCL, RData, DataFrames, StableRNGs, MCMCDiagnosticTools

@time output, likelihoods = mh(d, 1, 5; sigma_init_scale = [0.066, 0.04, 0.047, 0.042, 0.03], 
    init = [-0.3939072870143555,-0.053087595728158805,3.0372927867175656,0.0009028630309366971,-7.179528994354247], rng=StableRNG(98765))
if rank == 0
    println(likelihoods)
end
# println("rank $rank: $(d.likContribs)")
sigma_init_scale = [0.02, 0.005, 0.04, 0.02, 0.007]
init = RealType[0.02419075010987857,2.4112342210999214,3.3324837076534344,-2.4054477628972135,-0.09265020291382649]
# init = RealType[-0.3939072870143555,-0.053087595728158805,3.0372927867175656,0.0009028630309366971,-7.179528994354247]
rng = StableRNG(seed_mcmc)
for p in 1:length(init)
    dist_inner          = Normal(0.0, sigma_init_scale[p])
    init[p] += rand(rng, dist_inner,1)[1]       
end
@time output, likelihoods = mh(d, n_mcmc_samples, 5, 0.5; 
    sigma_init_scale = sigma_init_scale,
    init=init, 
    up_dat=nothing,
    rng=rng,
    fix=Int[],
    out_sample="samples_$(m)samples_$(n_mcmc_samples)_$(seed_data)_$(seed_mcmc)_model2_cb.csv",
    out_ll = "likelihoods_$(m)samples_$(n_mcmc_samples)_$(seed_data)_$(seed_mcmc)_model2_cb.csv")
# using CSV
# CSV.write("samples_1msamples_20000.csv", DataFrame(output, :auto))

# CSV.write("likelihoods_1msamples_20000.csv", DataFrame(reshape(likelihoods, :, 1), :auto))
