# f64ad

[![Crates.io](https://img.shields.io/crates/v/f64ad.svg)](https://crates.io/crates/f64ad)

| [Documentation](https://djrakita.github.io/f64ad/) | [API](https://docs.rs/f64ad/0.0.2/) |

## Introduction

This crate brings easy to use, efficient, and highly flexible automatic differentiation to the
Rust programming language.  Utilizing Rust's extensive operator overloading and expressive Enum
features, f64ad can be thought of as a drop-in replacement for f64 that affords forward mode
or backwards mode automatic differentiation on any downstream computation in Rust.

## Key features
- f64ad supports reverse mode or forward mode automatic differentiation
- f64ad uses polymorphism such that any `f64ad` object can either be considered a derivative 
tracking variable or a standard f64 with very little overhead depending on your current use case.
Thus, it is reasonable to replace almost all uses of f64 with f64ad, and in return, you'll be able  
to "turn on" derivatives with respect to these values whenever you need them.    
- The f64ad Enum type implements several useful traits that allow it to operate almost exactly as a
standard f64.  For example, it even implements the `RealField` and `ComplexField` traits,
meaning it can be used in any `nalgebra` or `ndarray` computations.
- Certain functions can be pre-computed and locked to boost performance at run-time.

## Citing f64ad 

If you use any part of the f64ad library in your research, please cite the software as follows:

```string
 @misc{rakita_2022, url={https://djrakita.github.io/f64ad/}, 
 author={Rakita, Daniel}, 
 title={f64ad: Efficient and Flexible Automatic Differentiation in Rust}
 year={2022}} 
```

## Example 1: Univariate Autodiff
```rust
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

Output
```text
Result of v.powi(3): f64ad_var(f64ad_var{ value: 8.0, node_idx: 1 })
Backwards mode derivative: f64(12.0)
Forward mode derivative: f64(12.0)
```

## Example 2: Backwards Mode Multivariate Autodiff
```rust
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

Output
```text
Result: f64ad_var(f64ad_var{ value: 7.9454605418379876, node_idx: 8 })
d_result_d_v0: f64(-1.6645873461885696)
d_result_d_v1: f64(0.9092974268256817)
d_result_d_v2: f64(0.40074862246915655)
d_result_d_v3: f64(-0.25898004032460736)
```

## Example 3: Forward Mode Multivariate Autodiff

```rust
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
Output
```text
Result0: f64ad_var(f64ad_var{ value: 0.9092974268256817, node_idx: 1 })
Result1: f64ad_var(f64ad_var{ value: -0.4161468365471424, node_idx: 2 })
Result2: f64ad_var(f64ad_var{ value: -2.185039863261519, node_idx: 3 })
d_result0_d_v: f64(-0.4161468365471424)
d_result1_d_v: f64(-0.9092974268256817)
d_result2_d_v: f64(5.774399204041917)
```

## Example 4: Polymorphism
```rust
extern crate f64ad as f64ad_crate;
use f64ad_core::f64ad::{ComputationGraph, ComputationGraphMode, f64ad};

// f64ad is an enum here that is a drop-in replacement for f64.  It can track derivative information
// for both, either, or neither of the variables, you can select what you want depending on your
// application at the time.
fn f64ad_test(a: f64ad, b: f64ad) -> f64ad {
    return a + b;
}

fn main() {
    let mut computation_graph = ComputationGraph::new(ComputationGraphMode::Standard, None);
    let a = computation_graph.spawn_f64ad_var(1.0);
    let b = computation_graph.spawn_f64ad_var(2.0);

    // Compute result using two f64ad variables that track derivative information for both `a` and `b'.
    let result1 = f64ad_test(a, b);
    println!("result 1: {:?}", result1.value());

    ////////////////////////////////////////////////////////////////////////////////////////////////

    let mut computation_graph = ComputationGraph::new(ComputationGraphMode::Standard, None);
    let a = computation_graph.spawn_f64ad_var(1.0);

    // Compute result using one f64ad variables that only tracks derivative information for `a'.
    let result2 = f64ad_test(a, f64ad::f64(2.0));
    println!("result 2: {:?}", result2.value());

    ////////////////////////////////////////////////////////////////////////////////////////////////

    // Compute result using zero f64ad variables.  This operation will not keep track of derivative information
    // for any variable and will essentially run as normal f64 floats with almost no overhead.
    let result3 = f64ad_test(f64ad::f64(1.0), f64ad::f64(2.0));
    println!("result 3: {:?}", result3.value());
}
```
Output
```text
result 1: 3.0
result 2: 3.0
result 3: 3.0
```

## Example 5: Univariate Higher Order Derivatives
```rust
use f64ad_core::ComplexField;
use f64ad_core::f64ad::{ComputationGraph, ComputationGraphMode};

fn main() {
    // Create a standard computation graph.
    let mut computation_graph = ComputationGraph::new(ComputationGraphMode::Standard, None);

    // Spawn an f64ad variables from computation graph.
    let v = computation_graph.spawn_f64ad_var(2.0);

    let result = v.powi(5);
    println!("Result: {:?}", result);
    println!("////////////////////////////////////////////////////////////////////////////////////");

    // first derivative computations...
    // we must set the parameter `add_to_computation_graph` to `true`
    let derivatives = result.backwards_mode_grad(true);
    let d_result_d_v = derivatives.wrt(&v);
    println!("d_result_d_v: {:?}", d_result_d_v);
    println!("////////////////////////////////////////////////////////////////////////////////////");

    // second derivative computations...
    // we must set the parameter `add_to_computation_graph` to `true`
    let derivatives = d_result_d_v.backwards_mode_grad(true);
    let d2_result_d_v2 = derivatives.wrt(&v);
    println!("d2_result_d_v2: {:?}", d2_result_d_v2);
    println!("////////////////////////////////////////////////////////////////////////////////////");

    // third derivative computations...
    // we must set the parameter `add_to_computation_graph` to `true`
    let derivatives = d2_result_d_v2.backwards_mode_grad(true);
    let d3_result_d_v3 = derivatives.wrt(&v);
    println!("d3_result_d_v3: {:?}", d3_result_d_v3);
    println!("////////////////////////////////////////////////////////////////////////////////////");

    // fourth derivative computations...
    // we must set the parameter `add_to_computation_graph` to `true`
    let derivatives = d3_result_d_v3.backwards_mode_grad(true);
    let d4_result_d_v4 = derivatives.wrt(&v);
    println!("d4_result_d_v4: {:?}", d4_result_d_v4);
    println!("////////////////////////////////////////////////////////////////////////////////////");

    // fifth derivative computations...
    // we must set the parameter `add_to_computation_graph` to `true`
    let derivatives = d4_result_d_v4.backwards_mode_grad(true);
    let d5_result_d_v5 = derivatives.wrt(&v);
    println!("d5_result_d_v5: {:?}", d5_result_d_v5);
    println!("////////////////////////////////////////////////////////////////////////////////////");

    // sixth derivative computations...
    // we must set the parameter `add_to_computation_graph` to `true`
    let derivatives = d5_result_d_v5.backwards_mode_grad(true);
    let d6_result_d_v6 = derivatives.wrt(&v);
    println!("d6_result_d_v6: {:?}", d6_result_d_v6);
    println!("////////////////////////////////////////////////////////////////////////////////////");
}
```
Output
```text
Result: f64ad_var(f64ad_var{ value: 32.0, node_idx: 1 })
////////////////////////////////////////////////////////////////////////////////////
d_result_d_v: f64ad_var(f64ad_var{ value: 80.0, node_idx: 6 })
////////////////////////////////////////////////////////////////////////////////////
d2_result_d_v2: f64ad_var(f64ad_var{ value: 160.0, node_idx: 23 })
////////////////////////////////////////////////////////////////////////////////////
d3_result_d_v3: f64ad_var(f64ad_var{ value: 240.0, node_idx: 80 })
////////////////////////////////////////////////////////////////////////////////////
d4_result_d_v4: f64ad_var(f64ad_var{ value: 240.0, node_idx: 249 })
////////////////////////////////////////////////////////////////////////////////////
d5_result_d_v5: f64ad_var(f64ad_var{ value: 120.0, node_idx: 706 })
////////////////////////////////////////////////////////////////////////////////////
d6_result_d_v6: f64ad_var(f64ad_var{ value: 0.0, node_idx: 1888 })
////////////////////////////////////////////////////////////////////////////////////
```

## Example 6: Multivariate higher order derivatives
```rust
use f64ad_core::ComplexField;
use f64ad_core::f64ad::{ComputationGraph, ComputationGraphMode};

fn main() {
    // Create a standard computation graph.
    let mut computation_graph = ComputationGraph::new(ComputationGraphMode::Standard, None);

    // Spawn an f64ad variables from computation graph.
    let v0 = computation_graph.spawn_f64ad_var(2.0);
    let v1 = computation_graph.spawn_f64ad_var(4.0);

    let result = v0.powf(v1);
    println!("Result: {:?}", result);
    println!("////////////////////////////////////////////////////////////////////////////////////");

    // first derivative computations...
    let derivatives = result.backwards_mode_grad(true);

    let d_result_d_v0 = derivatives.wrt(&v0);
    let d_result_d_v1 = derivatives.wrt(&v1);
    println!("d_result_d_v0: {:?}", d_result_d_v0);
    println!("d_result_d_v0: {:?}", d_result_d_v1);
    println!("////////////////////////////////////////////////////////////////////////////////////");

    // second derivative computations...
    let derivatives2_d_result_d_v0 = d_result_d_v0.backwards_mode_grad(true);
    let derivatives2_d_result_d_v1 = d_result_d_v1.backwards_mode_grad(true);

    let d2_result_dv0v0 = derivatives2_d_result_d_v0.wrt(&v0);
    let d2_result_dv0v1 = derivatives2_d_result_d_v0.wrt(&v1);
    let d2_result_dv1v0 = derivatives2_d_result_d_v1.wrt(&v0);
    let d2_result_dv1v1 = derivatives2_d_result_d_v1.wrt(&v1);
    println!("d2_result_dv0v0: {:?}", d2_result_dv0v0);
    println!("d2_result_dv0v1: {:?}", d2_result_dv0v1);
    println!("d2_result_dv1v0: {:?}", d2_result_dv1v0);
    println!("d2_result_dv1v1: {:?}", d2_result_dv1v1);
    println!("////////////////////////////////////////////////////////////////////////////////////");

    // etc...
}
```
Output
```text
Result: f64ad_var(f64ad_var{ value: 16.0, node_idx: 2 })
////////////////////////////////////////////////////////////////////////////////////
d_result_d_v0: f64ad_var(f64ad_var{ value: 32.0, node_idx: 11 })
d_result_d_v0: f64ad_var(f64ad_var{ value: 11.090354888959125, node_idx: 13 })
////////////////////////////////////////////////////////////////////////////////////
d2_result_dv0v0: f64ad_var(f64ad_var{ value: 48.0, node_idx: 66 })
d2_result_dv0v1: f64ad_var(f64ad_var{ value: 30.18070977791825, node_idx: 68 })
d2_result_dv1v0: f64ad_var(f64ad_var{ value: 30.18070977791825, node_idx: 290 })
d2_result_dv1v1: f64ad_var(f64ad_var{ value: 7.687248222691222, node_idx: 292 })
////////////////////////////////////////////////////////////////////////////////////
```

## Example 7: Locked Computation Graphs
```rust
use f64ad_core::ComplexField;
use f64ad_core::f64ad::{ComputationGraph, ComputationGraphMode};

fn main() {
    // Create a computation graph with mode `Lock`.  This signals that all computations that happen
    // on this graph will eventually be locked and, thus, only certain programs without conditional
    // branching will be compatible.  Any incompatible programs will panic.
    let mut computation_graph = ComputationGraph::new(ComputationGraphMode::Lock, None);
    let v = computation_graph.spawn_f64ad_var(3.0);

    let result = v.cos();

    // This locks the computation graph.  The `result` variable is taken as input here and is stored
    // by the locked graph.
    let mut function_locked_computation_graph = computation_graph.lock(None, result);

    // We can now replace the value of `v` here and use the `push_forward_compute` function to
    // recompute all downstream values on the locked function.
    function_locked_computation_graph.set_value(0, 0.0);
    function_locked_computation_graph.push_forward_compute();
    let new_output = function_locked_computation_graph.get_value(function_locked_computation_graph.template_output().node_idx());
    println!("v.cos() at v = 0.0: {:?}", new_output);

    // Here is another example where `v` is set with a value of 1.0.
    function_locked_computation_graph.set_value(0, 1.0);
    function_locked_computation_graph.push_forward_compute();
    let new_output = function_locked_computation_graph.get_value(function_locked_computation_graph.template_output().node_idx());
    println!("v.cos() at v = 1.0: {:?}", new_output);
    println!("////////////////////////////////////////////////////////////////////////////////////");

    // Because derivative/ gradient computations do not ever require conditional branching,
    // any derivatives over any compatible lockable functions can be locked as well.
    let derivatives = result.backwards_mode_grad(true);
    let derivative = derivatives.wrt(&v);
    let mut derivative_locked_computation_graph = computation_graph.lock(None, derivative);

    // Here, we are pushing forward the computation on the derivative of v.cos() at v = 0.0
    derivative_locked_computation_graph.set_value(0, 0.0);
    derivative_locked_computation_graph.push_forward_compute();
    let new_output = derivative_locked_computation_graph.get_value(derivative_locked_computation_graph.template_output().node_idx());
    println!("derivative of v.cos() at v = 0.0: {:?}", new_output);

    // Here, we are pushing forward the computation on the derivative of v.cos() at v = 1.0
    derivative_locked_computation_graph.set_value(0, 1.0);
    derivative_locked_computation_graph.push_forward_compute();
    let new_output = derivative_locked_computation_graph.get_value(derivative_locked_computation_graph.template_output().node_idx());
    println!("derivative of v.cos() at v = 1.0: {:?}", new_output);

    // Locked computation graphs can also spawn `locked_vars`.  These are also variants of the f64ad
    // Enum, thus they can also operate in any function.  However, after spawning variables, all downstream
    // computations must be EXACTLY THE SAME as the functions used on the original ComputationGraph
    // prior to locking (if functions are different, an error will be thrown).  Thus, in this example,
    // we call v.cos() on the locked_var because it is the same as the original computation above.
    // The locked_computation_graph then automatically updates its internal data and correctly
    // computes the derivative after the push_forward_compute function.
    let v = derivative_locked_computation_graph.spawn_locked_var(2.0);
    v.cos();
    derivative_locked_computation_graph.push_forward_compute();
    let new_output = derivative_locked_computation_graph.get_value(derivative_locked_computation_graph.template_output().node_idx());
    println!("derivative of v.cos() at v = 2.0: {:?}", new_output);
}
```
Output
```text
v.cos() at v = 0.0: 1.0
v.cos() at v = 1.0: 0.5403023058681398
////////////////////////////////////////////////////////////////////////////////////
derivative of v.cos() at v = 0.0: 0.0
derivative of v.cos() at v = 1.0: -0.8414709848078965
derivative of v.cos() at v = 2.0: -0.9092974268256817
```

## Crate structure
This crate is a cargo workspace with two member crates: (1) `f64ad_core`; and (2) `f64ad_core_derive`.
All core implementations for f64ad can be found in `f64ad_core`.  The `f64ad_core_derive` is
currently a placeholder and will be used for procedural macro implementations.

## Implementation note on unsafe code
This crate uses `unsafe` implementations under the hood.  I tried to avoid using unsafe, but
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