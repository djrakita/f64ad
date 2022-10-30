use std::time::Instant;
use f64ad_core::f64ad2::{ComputationGraph, ComputationGraphMode};
use num_traits::Signed;

fn main() {
    let mut c = ComputationGraph::new(ComputationGraphMode::Lock, Some("test"));
    let v1 = c.spawn_f64ad_var(4.0);
    let v2 = c.spawn_f64ad_var(0.0);
    let v3 = c.spawn_f64ad_var(8.0);

    let res = v2.abs();

    let res = res.backwards_mode_grad(true).wrt(&v2);
    println!("{:?}", res);
    let res = res.backwards_mode_grad(true).wrt(&v2);
    println!("{:?}", res);
    let res = res.backwards_mode_grad(true).wrt(&v2);
    println!("{:?}", res);
}