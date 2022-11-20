use f64ad_core::ComplexField;
use f64ad_core::f64ad::{ComputationGraphMode, GlobalComputationGraphs};

fn main() {
    // Create a computation graph.
    let computation_graph = GlobalComputationGraphs::get_with_reset(None, None, ComputationGraphMode::Standard);

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