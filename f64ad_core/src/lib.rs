//! This is a member crate of the [f64ad_](https://crates.io/crates/f64ad) cargo workspace.  For full description and documentation,
//! refer to the f64ad_ crate.

extern crate core;

pub mod f64ad_;
pub mod f64ad;

pub use num_traits::*;
pub use nalgebra::{ComplexField, RealField};
pub use approx::*;
pub use simba::scalar::*;

// #[global_allocator]
// static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// #[global_allocator]
// static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;