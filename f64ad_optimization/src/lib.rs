use ::nlopt::{Algorithm, Target};
use optimization_engine::core::SolverStatus;
use f64ad_core::f64ad::f64ad;
use crate::nlopt::f64ad_optimize_nlopt;
use crate::open::f64ad_optimize_open_panoc;

#[cfg(feature = "nlopt_optimization")]
pub mod nlopt;

#[cfg(feature = "open_optimization")]
pub mod open;


pub fn f64ad_optimize(input: OptimizeInput) -> OptimizeOutput {
    return match input {
        #[cfg(feature = "nlopt_optimization")]
        OptimizeInput::NLopt { objective, inequality_constraints, equality_constraints, algorithm, n_dims, target, init_condition, lower_bounds, upper_bounds } => {
            let output = f64ad_optimize_nlopt(objective, inequality_constraints, equality_constraints, algorithm, n_dims, target, init_condition, lower_bounds, upper_bounds);
            OptimizeOutput::NLopt { x: output.0, x_val: output.1 }
        }
        #[cfg(feature = "open_optimization")]
        OptimizeInput::OpEnPANOC { objective, n_dims, init_condition, lower_bounds, upper_bounds } => {
            let output = f64ad_optimize_open_panoc(objective, n_dims, init_condition, lower_bounds, upper_bounds);
            OptimizeOutput::OpEnPANOC { x: output.0, solver_status: output.1 }
        }
    }
}

pub enum OptimizeInput<'a> {
    #[cfg(feature = "nlopt_optimization")]
    NLopt {
        objective: Box<dyn Fn(&[f64ad]) -> f64ad>,
        inequality_constraints: Vec<Box<dyn Fn(&[f64ad]) -> f64ad>>,
        equality_constraints: Vec<Box<dyn Fn(&[f64ad]) -> f64ad>>,
        algorithm: Algorithm,
        n_dims: usize,
        target: Target,
        init_condition: &'a [f64],
        lower_bounds: Option<&'a [f64]>,
        upper_bounds: Option<&'a [f64]>,
    },
    #[cfg(feature = "open_optimization")]
    OpEnPANOC {
        objective: Box<dyn Fn(&[f64ad]) -> f64ad>,
        n_dims: usize,
        init_condition: &'a [f64],
        lower_bounds: &'a [f64],
        upper_bounds: &'a [f64],
    },
}

#[derive(Clone, Debug)]
pub enum OptimizeOutput {
    #[cfg(feature = "nlopt_optimization")]
    NLopt { x: Vec<f64>, x_val: f64 },
    #[cfg(feature = "open_optimization")]
    OpEnPANOC { x: Vec<f64>, solver_status: SolverStatus }
}