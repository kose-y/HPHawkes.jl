const defs_Float64 = """
static double pdf(double);

static double pdf(double value) {
    return 0.5 * M_SQRT1_2 * M_2_SQRTPI * exp( - pow(value,2.0) * 0.5);
}
static double cdf(double);

static double cdf(double value) {
    return 0.5 * erfc(-value * M_SQRT1_2);
}
static double safe_exp(double);

static double safe_exp(double value) {
    return exp(value);
}
"""
const defs_Float32 = """
static float pdf(float);
static float pdf(float value) {
    const float rSqrt2f = 0.70710678118655f;
    const float rSqrtPif = 0.56418958354775f;
    return rSqrt2f * rSqrtPif * exp( - pow(value,2.0f) * 0.5f);
}
static float cdf(float);
static float cdf(float value) {
    const float rSqrt2f = 0.70710678118655f;
    return 0.5f * erfc(-value * rSqrt2f);
}
static float safe_exp(float);
static float safe_exp(float value) {
    if (value < -103.0f) {
        return 0.0f;
    } else if (value > 88.0f) {
        return MAXFLOAT;
    } else {
        return exp(value);
    }
}
"""
const TPB=128
for (RealType, RealCType, CastCType, RealCVectorType, ZERO, defs) in [
    (Float32, "float", "int", "float8", "0.0f", defs_Float32),
    (Float64, "double", "long", "double8", "0.0", defs_Float64)
]
@eval begin
    function compute_sum_kernel(::Val{$RealType})
        RealCType = $RealCType
        CastCType = $CastCType
        RealCVectorType = $RealCVectorType
        ZERO = $ZERO
        return """
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
        __kernel void computeSum(
            __global const $RealCVectorType *summand,
            __global $RealCVectorType *outputSum,
            const uint locationCount) {
            const uint lid = get_local_id(0);
            uint j = get_local_id(0);
            __local $RealCVectorType scratch[$TPB];
            $RealCVectorType sum = $ZERO;
            while (j < locationCount) {
                sum += summand[j];
                j += $TPB;
            } 
            scratch[lid] = sum;

            for(int k = 1; k < $TPB; k <<= 1) {
                barrier(CLK_LOCAL_MEM_FENCE);
                uint mask = (k << 1) - 1;
                if ((lid & mask) == 0) {
                    scratch[lid] += scratch[lid + k];
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            if (lid == 0) {
                outputSum[0] = scratch[0];
            }
        }
        """
    end
    function lik_contribs_kernel(::Val{$RealType})
        RealCType = $RealCType
        CastCType = $CastCType
        RealCVectorType = $RealCVectorType
        ZERO = $ZERO
        return $defs * """
            #pragma OPENCL EXTENSION cl_khr_fp64 : enable
            __kernel void computeLikContribs(__global const $RealCVectorType *locations,
                                            __global const $RealCType *times,
                                            __global $RealCType *likContribs,
                                            const $RealCType sigmaXprec,
                                            const $RealCType tauXprec,
                                            const $RealCType tauTprec,
                                            const $RealCType omega,
                                            const $RealCType theta,
                                            const $RealCType mu0,
                                            const int dimX,
                                            const uint locationCount) {
                const uint i = get_group_id(0);
                const uint lid = get_local_id(0);
                uint j = get_local_id(0);
                __local $RealCType scratch[$TPB];
                const $RealCVectorType vectorI = locations[i];
                const $RealCType timeI = times[i];
                $RealCType        sum = $ZERO;
                $RealCType mu0TauXprecDTauTprec = mu0 * pow(tauXprec,dimX) * tauTprec;
                $RealCType thetaSigmaXprecDOmega = theta * pow(sigmaXprec,dimX) * omega;
                while (j < locationCount) {
                    const $RealCType timDiff = timeI - times[j]; // timDiffs[i * locationCount + j];
                    const $RealCVectorType vectorJ = locations[j];
                    const $RealCVectorType difference = vectorI - vectorJ;



                    const $RealCType distance = sqrt(
                        dot(difference.lo, difference.lo) +
                        dot(difference.hi, difference.hi)
                    );

                    const $RealCType innerContrib = mu0TauXprecDTauTprec *
                                            pdf(distance * tauXprec) * 
                                            select($ZERO, pdf(timDiff*tauTprec), ($CastCType)isnotequal(timDiff,$ZERO)) +
                                            thetaSigmaXprecDOmega *
                                            select($ZERO, exp(-omega * timDiff), ($CastCType)isgreater(timDiff,$ZERO)) * pdf(distance * sigmaXprec);
                    sum += innerContrib;
                    j += $TPB;
                }
                scratch[lid] = sum;

                for(int k = 1; k < $TPB; k <<= 1) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                    uint mask = (k << 1) - 1;
                    if ((lid & mask) == 0) {
                        scratch[lid] += scratch[lid + k];
                    }
                }

                barrier(CLK_LOCAL_MEM_FENCE);
                if (lid == 0) {
                    likContribs[i] = log(scratch[0]) + theta *
                    ( exp(-omega*(times[locationCount-1]-times[i]))-1 ) - 
                    mu0 * ( cdf((times[locationCount-1]-times[i])*tauTprec)-
                    cdf(-times[i]*tauTprec) )   ;
                }
            }
            """
        end
    end

end
function get_kernels(ctx::cl.Context, T::Type{<:AbstractFloat})
    p_sum = cl.Program(ctx, source=compute_sum_kernel(Val(T))) |> cl.build!
    k_sum = cl.Kernel(p_sum, "computeSum")

    p_loglik = cl.Program(ctx, source=lik_contribs_kernel(Val(T))) |> cl.build!
    k_loglik = cl.Kernel(p_loglik, "computeLikContribs")
    return k_sum, k_loglik
end
