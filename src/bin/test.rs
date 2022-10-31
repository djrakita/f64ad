use nalgebra::DVector;
use f64ad_core::f64ad::{ComputationGraph, ComputationGraphMode};


fn main() {
    let mut c = ComputationGraph::new(ComputationGraphMode::Standard, None);
    let v1 = c.spawn_f64ad_var(1.0);
    let v2 = c.spawn_f64ad_var(2.0);
    let v3 = c.spawn_f64ad_var(3.0);
    let v4 = c.spawn_f64ad_var(4.0);
    let v5 = c.spawn_f64ad_var(5.0);

    let d1 = DVector::from_column_slice(&vec![v1, v2, v3, v4, v5]);

    let res = d1.svd(true, true);
    println!("{:?}", res);
}