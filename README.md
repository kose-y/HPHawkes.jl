# HPHawkes

This software package is an open-source Julia implementation of HPHawkes, high-performance spatiotemporal Hawkes process. It supports acceleartion through graphic processing units (GPUs) with OpenCL. This package needs the following steps to install.

```julia
using Pkg
pkg"add https://github.com/kose-y/HPHawkes.jl"
```

## Citation

The methods and applications of this software package are detailed in the following publication:

*Ko S, Suchard MA, Holbrook AJ (2024). Scaling Hawkes processes to one million COVID-19 cases. [Arxiv 2407.11349](https://arxiv.org/abs/2407.11349).*


## Acknowledgments

This project has been supported by the awards NIH K25 AI153816, NSF DMS 2152774, NSF DMS 2236854 (Holbrook), NIH U19 AI135995, NIH R01 AI153044, and NIH R01 AI162611 (Suchard). We gratefully acknowledge generous support of Advanced Micro Devices, Inc., including the donation of parallel computing resources used for this research.
