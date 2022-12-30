use optimization_engine::{Optimizer, Problem, SolverError};
use optimization_engine::constraints::{Rectangle};
use optimization_engine::core::SolverStatus;
use optimization_engine::panoc::{PANOCCache, PANOCOptimizer};
use f64ad_core::f64ad::{f64ad, GlobalComputationGraphs};

pub(crate) fn f64ad_optimize_open_panoc(objective: Box<dyn Fn(&[f64ad]) -> f64ad>,
                                        n_dims: usize,
                                        init_condition: &[f64],
                                        lower_bounds: &[f64],
                                        upper_bounds: &[f64]) -> (Vec<f64>, SolverStatus) {
    let mut panoc_cache = PANOCCache::new(n_dims, 1e-5, 3);

    let f = |u: &[f64], c: &mut f64| -> Result<(), SolverError> {
        let mut f64ad_inputs = vec![];
        for input in u { f64ad_inputs.push(f64ad::f64(*input)); }

        let result: f64ad = objective(&f64ad_inputs);
        *c = result.value();

        Ok(())
    };
    let df = |u: &[f64], grad: &mut [f64]| -> Result<(), SolverError> {
        let c = GlobalComputationGraphs::get(None, None);
        c.reset();
        let mut f64ad_inputs = vec![];
        for input in u { f64ad_inputs.push(c.spawn_variable(*input)); }

        let result: f64ad = objective(&f64ad_inputs);

        let backwards_mode_grad = result.backwards_mode_grad(false);
        for (idx, f64ad_input) in f64ad_inputs.iter().enumerate() {
            grad[idx] = backwards_mode_grad.wrt(f64ad_input).value();
        }

        Ok(())
    };

    let constraints = Rectangle::new(Some(lower_bounds), Some(upper_bounds));
    let problem = Problem::new(&constraints, df, f);

    let mut panoc = PANOCOptimizer::new(problem, &mut panoc_cache);

    let mut u = init_condition.to_vec();
    let result = panoc.solve(&mut u);

    match result {
        Ok(solver_status) => {
            return (u, solver_status);
        }
        Err(_) => { panic!("err") }
    }
}

/*
pub fn f64ad_optimize_open_alm(objective: Box<dyn Fn(&[f64ad]) -> f64ad>,
                               inequality_constraints: Vec<Box<dyn Fn(&[f64ad]) -> f64ad>>,
                               equality_constraints: Vec<Box<dyn Fn(&[f64ad]) -> f64ad>>,
                               n_dims: usize,
                               init_condition: &[f64],
                               lower_bounds: &[f64],
                               upper_bounds: &[f64]) -> (Vec<f64>, f64) {
    let panoc_cache = PANOCCache::new(n_dims, 1e-5, 3);

    /*
    let mut alm_cache = AlmCache::new(panoc_cache, 0, match self.constraint_function {
        None => { 0 }
        Some(_) => { self.problem_size }
    });
    */

    todo!()
}
*/
