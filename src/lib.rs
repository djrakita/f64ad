//! # Introduction
//!
//! This crate brings easy to use, efficient, and highly flexible automatic differentiation to the
//! Rust programming language.  Utilizing Rust's extensive operator overloading and expressive Enum
//! features, f64ad can be thought of as a drop-in replacement for f64 that affords forward mode
//! or backwards mode automatic differentiation on any downstream computation in Rust.
//!
//! The f64ad type implements several useful traits that allow it to operate almost exactly as a
//! standard f64 would.  For example, it even implements the `RealField` and `ComplexField` traits,
//! meaning it can be used in any `nalgebra` or `ndarray` computations.
//!
//! This crate is a cargo workspace with two member crates: (1) `f64ad_core`; and (2) `f64ad_core_derive`.
//! All core implementations for f64ad can be found in `f64ad_core`.  The `f64ad_core_derive` is
//! currently a placeholder and will be used for procedural macro implementations.
//!
//! ## Example 1
//! ```
//! extern crate f64ad;
//! use f64ad_core::ComplexField;
//! use f64ad_core::f64ad::{ComputationGraph, ComputationGraphMode};
//!
//! fn main() {
//!     // Create a standard computation graph.
//!     let mut computation_graph = ComputationGraph::new(ComputationGraphMode::Standard, None);
//!
//!     // Spawn an f64ad variable with a value of 2.
//!     let v = computation_graph.spawn_f64ad_var(2.0);
//!
//!     // You can now use an f64ad exactly the same as you would use a standard f64.  In this example,
//!     // we are just using the `powi` function to take v to the third power.
//!     let result = v.powi(3);
//!     println!("Result of v.powi(3): {:?}", result);
//!
//!     // We can now find the derivative of our just computed function with respect to our input variable,
//!     // `v`.
//!
//!     // We can do this in one of two ways.  First, we can use backwards mode autodiff, meaning we
//!     // call `backwards_mode_grad` on our output result wrt our input variable, `v`:
//!     let backwards_mode_derivative = result.backwards_mode_grad(false).wrt(&v);
//!
//!     // Alternatively, we can use forward mode autodiff, meaning we call `forward_mode_grad` on
//!     // our input variable `v` wrt to our output variable, `result`.
//!     let forward_mode_derivative = v.forward_mode_grad(false).wrt(&result);
//!
//!     // Both methods will output the same derivative.
//!     println!("Backwards mode derivative: {:?}", backwards_mode_derivative);
//!     println!("Forward mode derivative: {:?}", forward_mode_derivative);
//! }
//! ```

extern crate f64ad_core;
extern crate f64ad_core_derive;

pub use f64ad_core::*;
pub use f64ad_core_derive::*;