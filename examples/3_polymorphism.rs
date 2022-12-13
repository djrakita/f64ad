use f64ad_core::f64ad::{ComputationGraphType, GlobalComputationGraphs, f64ad};

// f64ad_ is an enum here that is a drop-in replacement for f64.  It can track derivative information
// for both, either, or neither of the variables, you can select what you want depending on your
// application at the time.
fn f64ad_test(a: f64ad, b: f64ad) -> f64ad {
    return a + b;
}

fn main() {
    let computation_graph = GlobalComputationGraphs::get(None, None, ComputationGraphType::ComputationGraph1);
    let a = computation_graph.spawn_variable(1.0);
    let b = computation_graph.spawn_variable(2.0);

    // Compute result using two f64ad_ variables that track derivative information for both `a` and `b'.
    let result1 = f64ad_test(a, b);
    println!("result 1: {:?}", result1);

    ////////////////////////////////////////////////////////////////////////////////////////////////

    computation_graph.reset();
    let a = computation_graph.spawn_variable(1.0);

    // Compute result using one f64ad_ variables that only tracks derivative information for `a'.
    let result2 = f64ad_test(a, f64ad::f64(2.0));
    println!("result 2: {:?}", result2);

    ////////////////////////////////////////////////////////////////////////////////////////////////

    // Compute result using zero f64ad_ variables.  This operation will not keep track of derivative information
    // for any variable and will essentially run as normal f64 floats with almost no overhead.
    let result3 = f64ad_test(f64ad::f64(1.0), f64ad::f64(2.0));
    println!("result 3: {:?}", result3);
}