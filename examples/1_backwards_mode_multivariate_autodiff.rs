use f64ad_core::ComplexField;
use f64ad_core::f64ad::{ComputationGraphMode, GlobalComputationGraphs};

fn main() {
    // Create a computation graph.
    let computation_graph = GlobalComputationGraphs::get_with_reset(None, None, ComputationGraphMode::Standard);

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