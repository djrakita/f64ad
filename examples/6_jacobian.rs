use f64ad_core::ComplexField;
use f64ad_core::f64ad::{f64ad_jacobian, GlobalComputationGraphs};

fn main() {
    // Create a computation graph.
    let computation_graph = GlobalComputationGraphs::get(None, None);

    // Spawn an f64ad_ variables from computation graph.
    let v0 = computation_graph.spawn_variable(2.0);
    let v1 = computation_graph.spawn_variable(4.0);

    let result1 = v0.powf(v1);
    let result2 = v1.log(v0);
    println!("Result1: {:?}", result1);
    println!("Result2: {:?}", result2);
    println!("////////////////////////////////////////////////////////////////////////////////////");

    // Computes the first order jacobian and prints the summary
    let first_order_jacobian = f64ad_jacobian(&[v0, v1], &[result1, result2], 1);
    first_order_jacobian.print_summary();

    println!("////////////////////////////////////////////////////////////////////////////////////");

    // Computes the fourth order jacobian and prints the summary
    let higher_order_jacobian = f64ad_jacobian(&[v0, v1], &[result1, result2], 4);
    higher_order_jacobian.print_summary();
}