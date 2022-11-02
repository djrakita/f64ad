# f64ad

# Introduction

This crate brings easy to use, efficient, and highly flexible automatic differentiation to the
Rust programming language.  Utilizing Rust's extensive operator overloading and expressive Enum
features, f64ad can be thought of as a drop-in replacement for f64 that affords forward mode
or backwards mode automatic differentiation on any downstream computation in Rust.

## Key features
- f64ad supports reverse mode or forward mode automatic differentiation
- The f64ad Enum type implements several useful traits that allow it to operate almost exactly as a
standard f64.  For example, it even implements the `RealField` and `ComplexField` traits,
meaning it can be used in any `nalgebra` or `ndarray` computations.
- Certain functions can be pre-computed and locked to boost performance at run-time.

## Example 1: Univariate Autodiff
```
extern crate f64ad;
use f64ad_core::ComplexField;
use f64ad_core::f64ad::{ComputationGraph, ComputationGraphMode};

fn main() {
    // Create a standard computation graph.
    let mut computation_graph = ComputationGraph::new(ComputationGraphMode::Standard, None);

    // Spawn an f64ad variable with a value of 2.
    let v = computation_graph.spawn_f64ad_var(2.0);

    // You can now use an f64ad exactly the same as you would use a standard f64.  In this example,
    // we are just using the `powi` function to take v to the third power.
    let result = v.powi(3);
    println!("Result of v.powi(3): {:?}", result);

    // We can now find the derivative of our just computed function with respect to our input variable,
    // `v`.

    // We can do this in one of two ways.  First, we can use backwards mode autodiff, meaning we
    // call `backwards_mode_grad` on our output result wrt our input variable, `v`:
    let backwards_mode_derivative = result.backwards_mode_grad(false).wrt(&v);

    // Alternatively, we can use forward mode autodiff, meaning we call `forward_mode_grad` on
    // our input variable `v` wrt to our output variable, `result`.
    let forward_mode_derivative = v.forward_mode_grad(false).wrt(&result);

    // Both methods will output the same derivative.
    println!("Backwards mode derivative: {:?}", backwards_mode_derivative);
    println!("Forward mode derivative: {:?}", forward_mode_derivative);
}
```

## Example 2: Backwards Mode Multivariate Autodiff
```
use f64ad_core::ComplexField;
use f64ad_core::f64ad::{ComputationGraph, ComputationGraphMode};

fn main() {
    // Create a standard computation graph.
    let mut computation_graph = ComputationGraph::new(ComputationGraphMode::Standard, None);

    // Spawn an f64ad variables from computation graph.
    let v0 = computation_graph.spawn_f64ad_var(2.0);
    let v1 = computation_graph.spawn_f64ad_var(4.0);
    let v2 = computation_graph.spawn_f64ad_var(6.0);
    let v3 = computation_graph.spawn_f64ad_var(8.0);

    // compute some result using our variables
    let result = v0.sin() * v1 + 5.0 * v2.log(v3);
    println!("Result: {:?}", result);

    // compute derivatives in backwards direction from result.  Using backwards mode automatic
    // differentiation makes sense in this case because our number of outputs (1) is less than
    // our number of input variables (4).
    let derivatives = result.backwards_mode_grad(false);

    // access derivatives for each input variable from our `derivatives` object.
    let d_result_d_v0 = derivatives.wrt(&v0);
    let d_result_d_v1 = derivatives.wrt(&v1);
    let d_result_d_v2 = derivatives.wrt(&v2);
    let d_result_d_v3 = derivatives.wrt(&v3);

    // print results
    println!("d_result_d_v0: {:?}", d_result_d_v0);
    println!("d_result_d_v1: {:?}", d_result_d_v1);
    println!("d_result_d_v2: {:?}", d_result_d_v2);
    println!("d_result_d_v3: {:?}", d_result_d_v3);
}
```
## Example 3: Forward Mode Multivariate Autodiff

```
use f64ad_core::ComplexField;
use f64ad_core::f64ad::{ComputationGraph, ComputationGraphMode};

fn main() {
    // Create a standard computation graph.
    let mut computation_graph = ComputationGraph::new(ComputationGraphMode::Standard, None);

    // Spawn an f64ad variable with a value of 2.
    let v = computation_graph.spawn_f64ad_var(2.0);

    // compute some results using our variable
    let result0 = v.sin();
    let result1 = v.cos();
    let result2 = v.tan();
    println!("Result0: {:?}", result0);
    println!("Result1: {:?}", result1);
    println!("Result2: {:?}", result2);

    // compute derivatives in forward direction from v.  Using forward mode automatic
    // differentiation makes sense in this case because our number of outputs (3) is greater than
    // our number of input variables (1).
    let derivatives = v.forward_mode_grad(false);

    // access derivatives for each input variable from our `derivatives` object.
    let d_result0_d_v = derivatives.wrt(&result0);
    let d_result1_d_v = derivatives.wrt(&result1);
    let d_result2_d_v = derivatives.wrt(&result2);

    // print results
    println!("d_result0_d_v: {:?}", d_result0_d_v);
    println!("d_result1_d_v: {:?}", d_result1_d_v);
    println!("d_result2_d_v: {:?}", d_result2_d_v);
}
```

## Crate structure
This crate is a cargo workspace with two member crates: (1) `f64ad_core`; and (2) `f64ad_core_derive`.
All core implementations for f64ad can be found in `f64ad_core`.  The `f64ad_core_derive` is
currently a placeholder and will be used for procedural macro implementations.

## Implementation note on unsafe code
This crate uses `unsafe` implementations under the hood.  I tried to avoid using `unsafe`, but
I deemed it necessary in this case.  Without using unsafe code, I had two other options:
(1) `f64ad_var` could've used a smart pointer that cannot implement the `Copy` trait and, thus,
`f64ad` could not be `Copy` either.  This would mean that f64ad could not be an easy drop-in
replacement for f64 as annoying .clone() functions would have to be littered everywhere; or (2)
`f64ad_var` could've used a reference to some smart pointer, such as `&RefCell`.  However, certain
libraries, such as `nalgebra`, require their generic inputs to be static,
meaning the reference would've had to be `&'static RefCell` in `f64ad_var`.  In turn,
`ComputationGraph` would also have to be static, meaning it would essentially have to be a
mutable global static variable that would involve unsafe code anyway.  In the end, I viewed
an `unsafe` raw pointer to a `ComputationGraph` as the "best" among three sub-optimal
options.  

I was very careful with my internal implementations given the unsafe nature of the computations;
however, PLEASE be aware that the computation graph should NEVER GO OUT OF SCOPE IF ANY OF
ITS VARIABLES ARE STILL IN USE.