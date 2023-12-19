module HPHawkes
using LinearAlgebra, OpenCL, Pipe
# Write your package code here.
include("kernels.jl")
include("struct.jl")
include("driver.jl")
# include("mcmc.jl")
end
