use std::time::Instant;
use nalgebra::{ComplexField, RealField};
use f64ad_core::f64ad2::{ComputationGraph, ComputationGraphMode, f64ad};
use num_traits::Signed;

fn main() {
    let mut c = ComputationGraph::new(ComputationGraphMode::Lock, Some("test"));
    let v1 = c.spawn_f64ad_var(4.0);
    let v2 = c.spawn_f64ad_var(-0.2);
    let v3 = c.spawn_f64ad_var(0.0);

    println!(" >>> {:?}", 4.0 % 0.1);
    let res = v1 % v2;
    println!("{:?}", res);
    let res = res.backwards_mode_grad(true).wrt(&v2);
    println!("{:?}", res);
    let res = res.backwards_mode_grad(true).wrt(&v2);
    println!("{:?}", res);
    let res = res.backwards_mode_grad(true).wrt(&v2);
    println!("{:?}", res);
    let res = res.backwards_mode_grad(true).wrt(&v2);
    println!("{:?}", res);
}