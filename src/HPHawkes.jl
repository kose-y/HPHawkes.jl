module HPHawkes
using LinearAlgebra, OpenCL
# Write your package code here.
include("kernels.jl")
include("struct.jl")
include("driver.jl")
end
