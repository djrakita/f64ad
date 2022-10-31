//! This is a member crate of the [f64ad](https://crates.io/crates/f64ad) cargo workspace.  For full description and documentation,
//! refer to the f64ad crate.

pub mod f64ad;

pub use num_traits::*;
pub use nalgebra::{ComplexField, RealField};
pub use approx::*;
pub use simba::scalar::*;