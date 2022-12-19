# f64ad

[![Crates.io](https://img.shields.io/crates/v/f64ad.svg)](https://crates.io/crates/f64ad)

| [Documentation](https://djrakita.github.io/f64ad/) | [API](https://docs.rs/f64ad/0.0.5/) |

## Introduction

This crate brings easy to use, efficient, and highly flexible automatic differentiation to the
Rust programming language.  Utilizing Rust's extensive operator overloading and expressive Enum
features, f64ad can be thought of as a drop-in replacement for f64 that affords forward mode
or backwards mode automatic differentiation on any downstream computation in Rust.

## Key features
- f64ad supports reverse mode or forward mode automatic differentiation
- f64ad supports not just first derivatives, but also any higher order derivatives on any functions.  
- f64ad uses polymorphism such that any `f64ad` object can either be considered a derivative 
tracking variable or a standard f64 with very little overhead depending on your current use case.
Thus, it is reasonable to replace almost all uses of f64 with f64ad, and in return, you'll be able  
to "turn on" derivatives with respect to these values whenever you need them.    
- The f64ad Enum type implements several useful traits that allow it to operate almost exactly as a
standard f64.  For example, it even implements the `RealField` and `ComplexField` traits,
meaning it can be used in any `nalgebra` or `ndarray` computations.
- Certain functions can be pre-computed and locked to boost performance at run-time.

## Crate structure
This crate is a cargo workspace with two member crates: (1) `f64ad_core`; and (2) `f64ad_core_derive`.
All core implementations for f64ad can be found in `f64ad_core`.  The `f64ad_core_derive` is
currently a placeholder and will be used for procedural macro implementations.

## Citing f64ad 

If you use any part of the f64ad library in your research, please cite the software as follows:

```string
 @misc{rakita_2022, url={https://djrakita.github.io/f64ad/}, 
 author={Rakita, Daniel}, 
 title={f64ad: Efficient and Flexible Automatic Differentiation in Rust}
 year={2022}} 
```
