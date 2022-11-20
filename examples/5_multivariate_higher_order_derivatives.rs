use f64ad_core::ComplexField;
use f64ad_core::f64ad::{ComputationGraphMode, GlobalComputationGraphs};

fn main() {
    // Create a computation graph.
    let computation_graph = GlobalComputationGraphs::get_with_reset(None, None, ComputationGraphMode::Standard);

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