using Distributions, Random, Plots, StatsPlots

 d = Normal(0.0, 1.0)

function log_posterior(x,sigma=1)
    - sum(x .^ 2)/(2*sigma^2)
end

# log_posterior(rand(d,20))

function mh(iterations,parameter_count)
    d = Laplace(0.0, 1.0)
    accept = zeros(iterations)
    states = zeros(iterations,parameter_count)
    states[1,:] = rand(d,parameter_count) 
    curr_log_prob = log_posterior(states[1,:])

    for i = 2:iterations
        p          = rand(1:parameter_count)      # choose parameter to update
        prop_state = states[i-1,:]           
        prop_state[[p]] += rand(d,1)                 # proposal
        prop_log_prob = log_posterior(prop_state) # get proposal target value

        u = log(rand(1)[1])
        mh_ratio = prop_log_prob - curr_log_prob
        if u < mh_ratio
            states[i,:] = prop_state
            accept[i]   = 1.0
            curr_log_prob = prop_log_prob
        else
            states[i,:] = states[i-1,:]
        end
    end
    states
end

output = mh(1000000,4)

plot(output[:,1])
density(output[:,2])
plot!(Normal(0,1))

