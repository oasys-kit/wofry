# wofry (Wave Optics FRamework in pYthon)

WOFRY [1] is the [OASYS](https://oasys-kit.github.io/) framework for wave optics
calculations. It provides threefold functionality:

- A generalization (abstraction) of a software tool for wave optics, combining
  the component definitions from [SYNED](https://github.com/oasys-kit/syned) with
  the abstract declaration of wavefronts and wave propagators in free space.
- A mechanism for interfacing a wave optics code (e.g. SRW, WISE, etc.),
  as a first step for integration into OASYS.
- WOFRY is complemented by [WOFRYIMPL](https://github.com/oasys-kit/wofryimpl)
  (physical model implementations) and [OASYS-WOFRY](https://github.com/oasys-kit/OASYS-WOFRY)
  (OASYS widgets), both described in [2].

## Documentation

https://wofry.readthedocs.io/

## Source repository

https://github.com/oasys-kit/wofry

## Quick installation

```console
$ pip install wofry
```

## References

[1] L. Rebuffi, M. Sanchez del Rio,
"Interoperability and complementarity of simulation tools for beamline design in the OASYS environment",
Proc. SPIE 10388, Advances in Computational Methods for X-Ray Optics IV, 1038808 (23 August 2017)
https://doi.org/10.1117/12.2274232

[2] Manuel Sanchez del Rio, Juan Reyes-Herrera, Rafael Celestre, Luca Rebuffi,
"WOFRY: a package for partially coherent beamline simulations in fourth-generation storage rings"
https://doi.org/10.48550/arXiv.2410.01338