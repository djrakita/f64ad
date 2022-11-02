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