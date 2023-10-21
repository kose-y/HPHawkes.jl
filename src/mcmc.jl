using Distributions, Random, Plots, StatsPlots, OpenCL, RData, DataFrames

function update_data(seed)
    run(`Rscript ../covid_hosp/code/update_locs.R -seed $seed "&"`)
end

function log_prior(x)
    #L = length(x)
    #- sum(x .^ 2 ./ (1:L)) / 2.0
    - sum(x .^ 2) / 2.0
end

function log_likelihood(d,x)
#     T = RealType
#     x = .exp(x)
# # computes loglikelihood
#     HPHawkes.loglik(d::HPHawkes.HawkesStorage{T}, 
#     x[1], #sigmaXprec
#     x[2], #tauXprec 
#     x[3], #tauTprec, 
#     x[4], #omega 
#     x[5], #theta::T
#     x[6], #mu0::T, 
#     2)
    0
end

function log_posterior(x)
    log_prior(x) + log_likelihood(d,x)
end

function delta(n)
    n^(-0.5)
end


function mh(iterations,parameter_count,target=0.5,up_dat=nothing)
    sigma       = [1.0, 1.0, 1.0, 1.0]          # proposal standard deviation
    #d           = Normal(0.0, sigma) # proposal distribution
    adapt_bound = 50
    accept      = [0, 0, 0, 0]
    props       = [0, 0, 0, 0]
    acceptances = 0
    if !isnothing(up_dat)
        seed = 1
        #rm("../covid_hosp/seeds.txt")
        open("../covid_hosp/seeds.txt", "w") do file
            write(file, string(1))
        end
    end

    states = zeros(iterations,parameter_count)
    states[1,:] = rand(d,parameter_count) 
    curr_log_prob = log_posterior(states[1,:])

    for i = 2:iterations
        p          = rand(1:parameter_count)      # choose parameter to update
        prop_state = states[i-1,:]   
        d          = Normal(0.0, sigma[p])        
        prop_state[[p]] += rand(d,1)                 # proposal
        prop_log_prob = log_posterior(prop_state) # get proposal target value

        u = log(rand(1)[1])
        mh_ratio = prop_log_prob - curr_log_prob
        if u < mh_ratio
            states[i,:]   = prop_state
            accept[p]  += 1.0
            acceptances  += 1.0
            curr_log_prob = prop_log_prob
        else
            states[i,:] = states[i-1,:]
        end
        props[p] += 1

        if mod(props[p], adapt_bound) == 0 
            accept_ratio = accept[p] / adapt_bound
            if accept_ratio > target
                sigma[p] *= 1 + delta(i)
            else
                sigma[p] *= 1 - delta(i)
            end

            accept[p] = 0
            props[p]  = 0
        end

        if mod(i, 10000) == 0
            print(i, " ", sigma, "\n")
        end

        if !isnothing(up_dat)
            s = open("../covid_hosp/seeds.txt") do f
                parse(Int, split(readline(f), " ")[end])
            end

            if s > seed 
                seed += 1
                # load data
                df = load("dataComplete.rds")
                Threads.@spawn update_data(seed)
            end
        end
    end
    print("Acceptance rate: ", acceptances/(iterations-1))
    states
end

output = mh(1000000,4)

plot(output[:,4])
# density(output[:,1])
# plot!(Normal(0,1))


