using Distributions, Random, Plots, StatsPlots, OpenCL, RData, DataFrames

function update_data(seed)
    run(`Rscript ../covid_hosp/code/update_locs.R -seed $seed "&"`)
end

function log_prior(x)
    #L = length(x)
    #- sum(x .^ 2 ./ (1:L)) / 2.0
    - sum(x .^ 2) / 2.0
end

function log_likelihood(d::HPHawkes.HawkesStorage{T},x) where T
#     T = RealType
    x_ = exp.(x)
# computes loglikelihood
    HPHawkes.loglik(d, 
    convert(T, x_[1] + x_[2]), #sigmaXprec is offest x_[1] + tauXprec.  This ensures that sigmaXprec is larger than tauXprec.
    convert(T, x_[2]), #tauXprec 
    convert(T, x_[3]), #tauTprec, 
    convert(T, x_[4]), #omega 
    convert(T, x_[5]), #theta::T
    convert(T, x_[6]), #mu0::T, 
    2)
#     0
end

function log_posterior(x)
    log_prior(x) + log_likelihood(d,x)
end

function delta(n)
    n^(-0.5)
end


function mh(data, iterations,parameter_count,target=0.5;
    up_dat=nothing, sigma_init_scale=nothing, init=zeros(parameter_count))
    if sigma_init_scale === nothing 
        sigma       = ones(parameter_count)         # proposal standard deviation
    else
        sigma = sigma_init_scale
    end
    #d           = Normal(0.0, sigma) # proposal distribution
    adapt_bound = 50
    accept      = zeros(Int, parameter_count)
    props       = zeros(Int, parameter_count)
    acceptances = 0
    if !isnothing(up_dat)
        seed = 1
        #rm("../covid_hosp/seeds.txt")
        open("../covid_hosp/seeds.txt", "w") do file
            write(file, string(1))
        end
    end

    states = zeros(iterations,parameter_count)
    states[1, :] .= init
    likelihoods = zeros(iterations)
    curr_log_prob = log_posterior(data, states[1,:])
    likelihoods[1] = curr_log_prob

    for i = 2:iterations
        p          = rand(1:parameter_count)      # choose parameter to update
        prop_state = states[i-1,:]   
        d          = Normal(0.0, sigma[p])        
        prop_state[p] += rand(d,1)[1]                 # proposal
        prop_log_prob = log_posterior(data, prop_state) # get proposal target value

        u = log(rand(1)[1])
        mh_ratio = prop_log_prob - curr_log_prob
        if u < mh_ratio
            states[i,:]   = prop_state
            accept[p]  += 1
            acceptances  += 1
            curr_log_prob = prop_log_prob
        else
            states[i,:] .= @view(states[i-1,:])
        end
        likelihoods[i] = curr_log_prob
        props[p] += 1

        if mod(props[p], adapt_bound) == 0 
            accept_ratio = accept[p] / adapt_bound 
            accept_ratio /= target

            if accept_ratio > 2.0
                accept_ratio = 2.0
            end

            if accept_ratio < 0.5
                accept_ratio = 0.5
            end

            sigma[p] *= accept_ratio

            # if accept_ratio > target
            #     sigma[p] *= 1 + delta(i)
            # else
            #     sigma[p] *= 1 - delta(i)
            # end

            accept[p] = 0
            props[p]  = 0
        end

        if mod(i, 100) == 0
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
    println("Acceptance rate: ", acceptances/(iterations-1))
    states, likelihoods
end


@time output, likelihoods = mh(d, 50000, 6; sigma_init_scale = [0.03, 0.03, 0.04, 0.05, 0.05, 0.03], init = [1.0, 2.23, 3.31, 1.94, 1.01, -0.35])

plot(output[:,4])
# density(output[:,1])
# plot!(Normal(0,1))


I