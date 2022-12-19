use f64ad_core::ComplexField;
use f64ad_core::f64ad::{GlobalComputationGraphs};

fn main() {
    // Create a computation graph.
    let computation_graph = GlobalComputationGraphs::get(None, None);

    // Spawn an f64ad_ variable with a value of 2.
    let v = computation_graph.spawn_variable(2.0);

    // You can now use an f64ad_ exactly the same as you would use a standard f64.  In this example,
    // we are just using the `powi` function to take v to the third power.
    let result = v.powi(3);
    println!("Result of v.powi(3): {:?}", result);

    // We can now find the derivative of our just computed function with respect to our input variable,
    // `v`.

    // We can do this in one of two ways.  First, we can use backwards mode autodiff, meaning we
    // call `backwards_mode_grad` on our output result wrt our input variable, `v`:
    let backwards_mode_derivative = result.backwards_mode_grad(false).wrt(&v);

    // Alternatively, we can use forward mode autodiff, meaning we call `forward_mode_grad` on
    // our input variable `v` wrt to our output variable, `result`.
    let forward_mode_derivative = v.forward_mode_grad(false).wrt(&result);

    // Both methods will output the same derivative.
    println!("Backwards mode derivative: {:?}", backwards_mode_derivative);
    println!("Forward mode derivative: {:?}", forward_mode_derivative);
}