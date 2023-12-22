const defs_Float64 = """
static double pdf(double);

static double pdf(double value) {
    return 0.5 * M_SQRT1_2 * M_2_SQRTPI * exp( - value * value * 0.5);
}
static double pdfpdf(double value1, double value2) {
    return 0.5 * 0.5 * M_SQRT1_2 * M_SQRT1_2 * M_2_SQRTPI * M_2_SQRTPI * exp( - (value1 * value1 + value2 * value2) * 0.5);
}
static double pdfpdf_1sq(double value1sq, double value2) {
    return 0.5 * 0.5 * M_SQRT1_2 * M_SQRT1_2 * M_2_SQRTPI * M_2_SQRTPI * exp( - (value1sq + value2 * value2) * 0.5);
}
static double exppdf(double value1, double value2) {
    return 0.5 * M_SQRT1_2 * M_2_SQRTPI * exp(- value1 - value2 * value2 * 0.5);
}
static double exppdf_2sq(double value1, double value2sq) {
    return 0.5 * M_SQRT1_2 * M_2_SQRTPI * exp(- value1 - value2sq * 0.5);
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
    return rSqrt2f * rSqrtPif * exp( - value * value * 0.5f);
}
static float pdfpdf(float value1, float value2) {
    const float rSqrt2f = 0.70710678118655f;
    const float rSqrtPif = 0.56418958354775f;
    return rSqrt2f * rSqrt2f * rSqrtPif * rSqrtPif * exp( - (value1 * value1 + value2 * value2) * 0.5f);
}
static float pdfpdf_1sq(float value1sq, float value2) {
    const float rSqrt2f = 0.70710678118655f;
    const float rSqrtPif = 0.56418958354775f;
    return rSqrt2f * rSqrt2f * rSqrtPif * rSqrtPif * exp( - (value1sq + value2 * value2) * 0.5f);
}
static float exppdf(float value1, float value2) {
    const float rSqrt2f = 0.70710678118655f;
    const float rSqrtPif = 0.56418958354775f;
    return rSqrt2f * rSqrtPif * exp(-value1 -value2 * value2 * 0.5f);
}
static float exppdf_2sq(float value1, float value2sq) {
    const float rSqrt2f = 0.70710678118655f;
    const float rSqrtPif = 0.56418958354775f;
    return rSqrt2f * rSqrtPif * exp(-value1 -value2sq * 0.5f);
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
const TPB=256
for (RealType, RealCType, IntCType, RealCVectorType8, RealCVectorType2, ZERO, defs) in [
    (Float32, "float", "int", "float8", "float2", "0.0f", defs_Float32),
    (Float64, "double", "long", "double8", "double2", "0.0", defs_Float64)
]
@eval begin
    function compute_sum_kernel(::Val{$RealType})
        RealCType = $RealCType
        IntCType = $IntCType
        RealCVectorType8 = $RealCVectorType8
        ZERO = $ZERO
        return """
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
        __kernel void computeSum(
            __global const $RealCVectorType8 *summand,
            __global $RealCVectorType8 *outputSum,
            const uint locationCount) {
            const uint lid = get_local_id(0);
            uint j = get_local_id(0);
            __local $RealCVectorType8 scratch[$TPB];
            $RealCVectorType8 sum = $ZERO;
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
        IntCType = $IntCType
        RealCVectorType2 = $RealCVectorType2
        ZERO = $ZERO
        return $defs * """
            #pragma OPENCL EXTENSION cl_khr_fp64 : enable
            __kernel void computeLikContribs(__global const $RealCVectorType2 *locations,
                                            __global const $RealCType *times,
                                            __global $RealCType *likContribs,
                                            const $RealCType sigmaXprec,
                                            const $RealCType tauXprec,
                                            const $RealCType tauTprec,
                                            const $RealCType omega,
                                            const $RealCType theta,
                                            const $RealCType mu0,
                                            const int dimX,
                                            const uint locationStart,
                                            const uint locationEnd,
                                            const uint locationCount) {
                const uint i = locationStart + get_group_id(0);
                const uint gid = get_group_id(0);
                const uint lid = get_local_id(0);
                uint j = get_local_id(0);
                __local $RealCType scratch[$TPB];
                __local $RealCType scratch2[$TPB];
                const $RealCVectorType2 vectorI = locations[i];
                const $RealCType timeI = times[i];
                $RealCType        sum = $ZERO;
                $RealCType mu0TauXprecDTauTprec = mu0 * pow(tauXprec,dimX) * tauTprec;
                $RealCType thetaSigmaXprecDOmega = theta * pow(sigmaXprec,dimX) * omega;
                while (j < locationCount) {
                    const $RealCType timDiff = timeI - times[j];
                    const $RealCVectorType2 vectorJ = locations[j];
                    const $RealCVectorType2 difference = vectorI - vectorJ;

                    const $RealCType distancesq = dot(difference.lo, difference.lo) + dot(difference.hi, difference.hi);

                    const $RealCType innerContrib = mu0TauXprecDTauTprec *
                                            ((timDiff != 0) ? pdfpdf_1sq(distancesq * tauXprec * tauXprec, timDiff * tauTprec) : $ZERO) +
                                            //pdf(distance * tauXprec) * pdf(timDiff*tauTprec) : $ZERO) +
                                            thetaSigmaXprecDOmega *
                                            ((timDiff > 0) ? exppdf_2sq(omega * timDiff, distancesq * sigmaXprec * sigmaXprec) : $ZERO);
                                            //exp(-omega * timDiff) * pdf(distance * sigmaXprec) : $ZERO);
                    sum += innerContrib;
                    j += $TPB;
                }

                scratch[lid] = sum;
                scratch2[lid] = sum;

                barrier(CLK_LOCAL_MEM_FENCE);
                for(int k = 1; k < $TPB; k <<= 1) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                    uint mask = (k << 1) - 1;
                    if ((lid & mask) == 0) {
                        if (scratch2[lid] < scratch2[lid + k])
                            scratch2[lid] = scratch2[lid + k];
                    }
                }

                barrier(CLK_LOCAL_MEM_FENCE);
                $RealCType maximum = max(scratch2[0], ($RealCType) 1e-40);
                scratch[lid] = scratch[lid] / maximum;
                barrier(CLK_LOCAL_MEM_FENCE);
                for(int k = 1; k < $TPB; k <<= 1) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                    uint mask = (k << 1) - 1;
                    if ((lid & mask) == 0) {
                        scratch[lid] += scratch[lid + k];
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                scratch[0] = max(scratch[0], ($RealCType) 1e-40);

                barrier(CLK_LOCAL_MEM_FENCE);
                if (lid == 0) {
                    likContribs[gid] = log(maximum) + log(scratch[0]) + theta *
                    ( exp(-omega*(times[locationCount-1]-times[i]))-1 ) - 
                    mu0 * ( cdf((times[locationCount-1]-times[i])*tauTprec)-
                    cdf(-times[i]*tauTprec) ) ;
                }
            }
            """
        end
    end

end
function get_kernels(ctx::cl.Context, T::Type{<:AbstractFloat})
    p_sum = @pipe cl.Program(ctx, source=compute_sum_kernel(Val(T))) |> cl.build!(_; options= "-cl-fast-relaxed-math")
    k_sum = cl.Kernel(p_sum, "computeSum")
03
    p_loglik = @pipe cl.Program(ctx, source=lik_contribs_kernel(Val(T))) |> cl.build!(_; options="-cl-fast-relaxed-math")
    k_loglik = cl.Kernel(p_loglik, "computeLikContribs")
    return k_sum, k_loglik
end
