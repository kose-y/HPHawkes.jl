function loglik(d::HawkesStorage{T},
    sigmaXprec::T, tauXprec::T, tauTprec::T, omega::T, theta::T, mu0::T, dimX;
    first_idx = 0,
    last_idx = d.m
    ) where T
    d.queue(d.k_loglik, (last_idx - first_idx) * TPB, TPB, d.locations_buff, d.times_buff, d.likContribs_buff, 
        sigmaXprec, tauXprec, tauTprec, omega, theta, mu0, 
        Int32(dimX), UInt32(first_idx), UInt32(last_idx), UInt32(d.m)) |> cl.wait
    cl.copy!(d.queue, d.likContribs, d.likContribs_buff) |> cl.wait
    d.queue(d.k_sum, TPB, TPB, d.likContribs_buff, d.likContribs_reduced_buff, UInt32((last_idx - first_idx) ÷ 8)) |> cl.wait# what is done if m not multiple of 8?
    cl.copy!(d.queue, d.likContribs_reduced, d.likContribs_reduced_buff) |> cl.wait
    sum(d.likContribs_reduced)
end
