# wofry (Wave Optics FRamwork in pYthon)


WOFRY [1] is the OASYS https://oasys-kit.github.io/ framework for waveoptics calculations. It contains a threefold functionality: 
- it provides a generalization (or abstraction) of a software tool for wave optics, combining the component definitions from SYNED https://github.com/oasys-kit/syned with the abstract declaration of wavefronts and wave propagators in free space.
- it defines a mechanism for interfacing a wave optics code (e.g., SRW, WISE etc.) in it, a first step for becoming interfaced in OASYS
- Moreover, WOFRY is complemented by WOFRYIMPL https://github.com/oasys-kit/wofryimpl with the implementation of the physical models, and by OASYS-WOFRY https://github.com/oasys-kit/OASYS-WOFRY that contain the OASYS widgets. All of them are described in [2].

## References

[1] https://doi.org/10.1117/12.2274232

[2] https://doi.org/10.48550/arXiv.2410.01338
