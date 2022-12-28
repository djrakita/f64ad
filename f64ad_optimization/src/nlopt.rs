use nlopt::{Algorithm, Nlopt, Target};
use f64ad_core::f64ad::{f64ad, GlobalComputationGraphs};

pub fn f64ad_optimize_nlopt(objective: Box<dyn Fn(&[f64ad]) -> f64ad>,
                            inequality_constraints: Vec<Box<dyn Fn(&[f64ad]) -> f64ad>>,
                            equality_constraints: Vec<Box<dyn Fn(&[f64ad]) -> f64ad>>,
                            algorithm: Algorithm,
                            n_dims: usize,
                            target: Target,
                            init_condition: &[f64]) -> (Vec<f64>, f64) {

    let o = get_nlopt_function(objective);

    let mut nlopt = Nlopt::new(algorithm, n_dims, o, target, ());

    for inequality_constraint in inequality_constraints {
        nlopt.add_inequality_constraint(get_nlopt_function(inequality_constraint), (), 0.000001).expect("error");
    }
    for equality_constraint in equality_constraints {
        nlopt.add_equality_constraint(get_nlopt_function(equality_constraint), (), 0.000001).expect("error");
    }

    nlopt.set_ftol_rel(0.00001).expect("error");
    nlopt.set_ftol_abs(0.00001).expect("error");
    nlopt.set_xtol_rel(0.00001).expect("error");

    let mut x = init_condition.to_vec();
    let result = nlopt.optimize(&mut x);

    return match result {
        Ok((_success_state, x_val)) => { (x, x_val) }
        Err((_fail_state, x_val)) => { (x, x_val) }
    };
}

fn get_nlopt_function (f: Box<dyn Fn(&[f64ad]) -> f64ad + 'static>) -> Box<dyn Fn(&[f64], Option<&mut [f64]>, &mut ()) -> f64> {
    let o = move |x: &[f64], grad: Option<&mut [f64]>, _: &mut ()| -> f64 {
        return match grad {
            None => {
                let mut f64ad_inputs = vec![];
                for input in x { f64ad_inputs.push(f64ad::f64(*input)); }

                let result: f64ad = f(&f64ad_inputs);

                result.value()
            }
            Some(grad) => {
                let g = GlobalComputationGraphs::get(None, None);
                g.reset();

                let mut f64ad_inputs = vec![];
                for input in x { f64ad_inputs.push(g.spawn_variable(*input)); }

                let result: f64ad = f(&f64ad_inputs);
                let backwards_mode_grad = result.backwards_mode_grad(false);
                for (idx, f64ad_input) in f64ad_inputs.iter().enumerate() {
                    grad[idx] = backwards_mode_grad.wrt(f64ad_input).value();
                }

                result.value()
            }
        };
    };

    return Box::new(o);
}