struct HawkesStorage{T}
    m::Int32
    likContribs_reduced::Vector{T}
    locations_buff::cl.Buffer{T}
    times_buff::cl.Buffer{T}
    likContribs_buff::cl.Buffer{T}
    likContribs_reduced_buff::cl.Buffer{T}
    device::cl.Device
    ctx::cl.Context
    queue::cl.CmdQueue
    k_sum::cl.Kernel
    k_loglik::cl.Kernel
end

function HawkesStorage{T}(ctx::cl.Context, locations::Array{T}, times::Array{T}; 
    device=first(devices(ctx)),
    queue=CmdQueue(ctx)) where T
    m = length(times)
    @assert m % 8 == 0 "number of samples should be a multiple of 8 for now."
    k_sum, k_loglik = get_kernels(ctx, T)
    return HawkesStorage{T}(m,
        Vector{T}(undef, 8),
        cl.Buffer(T, ctx, (:r, :copy), hostbuf=locations),
        cl.Buffer(T, ctx, (:r, :copy), hostbuf=times),
        cl.Buffer(T, ctx, :w, m),
        cl.Buffer(T, ctx, :w, 8),
        device,
        ctx,
        queue,
        k_sum,
        k_loglik)
end
