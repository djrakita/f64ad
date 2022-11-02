use f64ad_core::ComplexField;
use f64ad_core::f64ad::{ComputationGraph, ComputationGraphMode};

fn main() {
    // Create a computation graph with mode `Lock`.  This signals that all computations that happen
    // on this graph will eventually be locked and, thus, only certain programs without conditional
    // branching will be compatible.  Any incompatible programs will panic.
    let mut computation_graph = ComputationGraph::new(ComputationGraphMode::Lock, None);
    let v = computation_graph.spawn_f64ad_var(3.0);

    let result = v.cos();

    // This locks the computation graph.  The `result` variable is taken as input here and is stored
    // by the locked graph.
    let mut function_locked_computation_graph = computation_graph.lock(None, result);

    // We can now replace the value of `v` here and use the `push_forward_compute` function to
    // recompute all downstream values on the locked function.
    function_locked_computation_graph.set_value(0, 0.0);
    function_locked_computation_graph.push_forward_compute();
    let new_output = function_locked_computation_graph.get_value(function_locked_computation_graph.template_output().node_idx());
    println!("v.cos() at v = 0.0: {:?}", new_output);

    // Here is another example where `v` is set with a value of 1.0.
    function_locked_computation_graph.set_value(0, 1.0);
    function_locked_computation_graph.push_forward_compute();
    let new_output = function_locked_computation_graph.get_value(function_locked_computation_graph.template_output().node_idx());
    println!("v.cos() at v = 1.0: {:?}", new_output);
    println!("////////////////////////////////////////////////////////////////////////////////////");

    // Because derivative/ gradient computations do not ever require conditional branching,
    // any derivatives over any compatible lockable functions can be locked as well.
    let derivatives = result.backwards_mode_grad(true);
    let derivative = derivatives.wrt(&v);
    let mut derivative_locked_computation_graph = computation_graph.lock(None, derivative);

    // Here, we are pushing forward the computation on the derivative of v.cos() at v = 0.0
    derivative_locked_computation_graph.set_value(0, 0.0);
    derivative_locked_computation_graph.push_forward_compute();
    let new_output = derivative_locked_computation_graph.get_value(derivative_locked_computation_graph.template_output().node_idx());
    println!("derivative of v.cos() at v = 0.0: {:?}", new_output);

    // Here, we are pushing forward the computation on the derivative of v.cos() at v = 1.0
    derivative_locked_computation_graph.set_value(0, 1.0);
    derivative_locked_computation_graph.push_forward_compute();
    let new_output = derivative_locked_computation_graph.get_value(derivative_locked_computation_graph.template_output().node_idx());
    println!("derivative of v.cos() at v = 1.0: {:?}", new_output);

    // Locked computation graphs can also spawn `locked_vars`.  These are also variants of the f64ad
    // Enum, thus they can also operate in any function.  However, after spawning variables, all downstream
    // computations must be EXACTLY THE SAME as the functions used on the original ComputationGraph
    // prior to locking (if functions are different, an error will be thrown).  Thus, in this example,
    // we call v.cos() on the locked_var because it is the same as the original computation above.
    // The locked_computation_graph then automatically updates its internal data and correctly
    // computes the derivative after the push_forward_compute function.
    let v = derivative_locked_computation_graph.spawn_locked_var(2.0);
    v.cos();
    derivative_locked_computation_graph.push_forward_compute();
    let new_output = derivative_locked_computation_graph.get_value(derivative_locked_computation_graph.template_output().node_idx());
    println!("derivative of v.cos() at v = 2.0: {:?}", new_output);
}