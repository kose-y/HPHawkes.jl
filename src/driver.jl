function loglik(d::HawkesStorage{T}, 
    sigmaXprec::T, tauXprec::T, tauTprec::T, omega::T, theta::T, mu0::T, dimX) where T
    d.queue(d.k_loglik, d.m * TPB, TPB, d.locations_buff, d.times_buff, d.likContribs_buff, 
        sigmaXprec, tauXprec, tauTprec, omega, theta, mu0, 
        Int32(dimX), UInt32(d.m))
    d.queue(d.k_sum, d.m ÷ 8, TPB, d.likContribs_buff, d.likContribs_reduced_buff, UInt32(d.m ÷ 8)) # what is done if m not multiple of 8?
    cl.copy!(d.queue, d.likContribs_reduced, d.likContribs_reduced_buff)
    sum(d.likContribs_reduced)
end
