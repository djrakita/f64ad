use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};
use std::sync::Mutex;
use tinyvec::{tiny_vec, TinyVec};
use once_cell::sync::OnceCell;
use nalgebra::ComplexField;
use rand::{Rng, thread_rng};
use crate::f64ad::f64ad_var_1_mod::*;
use crate::f64ad::f64ad_var_f_mod::{ComputationGraphF, f64ad_var_f};
use crate::f64ad::f64ad_var_l_mod::{ComputationGraphL, f64ad_var_l, F64ADNodeL};
use crate::f64ad::f64ad_var_t_mod::{ComputationGraphT, f64ad_var_t};

pub mod trait_impls;
pub mod f64ad_var_1_mod;
pub mod f64ad_var_f_mod;
pub mod f64ad_var_l_mod;
pub mod f64ad_var_t_mod;
pub mod manual_derivative_functions;

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
pub enum f64ad {
    f64(f64),
    f64ad_var_1(f64ad_var_1),
    f64ad_var_f(f64ad_var_f),
    f64ad_var_t(f64ad_var_t),
    f64ad_var_l(f64ad_var_l)
}
impl Default for f64ad {
    fn default() -> Self {
        Self::f64(0.0)
    }
}
impl f64ad {
    #[inline(always)]
    pub fn value(&self) -> f64 {
        match self {
            f64ad::f64(v) => { *v }
            f64ad::f64ad_var_1(v) => { v.value() }
            f64ad::f64ad_var_f(v) => { v.value() }
            f64ad::f64ad_var_t(v) => { v.value() }
            f64ad::f64ad_var_l(v) => { v.value() }
        }
    }
    #[inline(always)]
    pub fn node_idx(&self) -> usize {
        match self {
            f64ad::f64(_) => { panic!("no node_idx on f64.") }
            f64ad::f64ad_var_1(v) => { v.node_idx() }
            f64ad::f64ad_var_f(v) => { v.node_idx() }
            f64ad::f64ad_var_t(v) => { v.node_idx() }
            f64ad::f64ad_var_l(v) => { v.node_idx() }
        }
    }
    #[inline(always)]
    pub fn map_to_type(&self) -> F64adType {
        match self {
            f64ad::f64(_) => { F64adType::F64 }
            f64ad::f64ad_var_1(_) => { F64adType::Var1 }
            f64ad::f64ad_var_f(_) => { F64adType::VarF }
            f64ad::f64ad_var_t(_) => { F64adType::VarT }
            f64ad::f64ad_var_l(_) => { F64adType::VarL }
        }
    }
    pub fn forward_mode_grad(&self, add_to_computation_graph: bool) -> ForwardModeGradOutput {
        return match self {
            f64ad::f64(_) => { panic!("cannot compute gradient on f64.") }
            f64ad::f64ad_var_1(_) => {
                assert_eq!(add_to_computation_graph, false);
                f64ad_universal_forward_mode_grad(self.clone(), false)
            }
            f64ad::f64ad_var_f(_) => {
                f64ad_universal_forward_mode_grad(self.clone(), add_to_computation_graph)
            }
            f64ad::f64ad_var_t(_) => { panic!("cannot compute gradient on f64ad_var_t") }
            f64ad::f64ad_var_l(_) => {
                assert_eq!(add_to_computation_graph, false);
                f64ad_universal_forward_mode_grad(self.clone(), false)
            }
        };
    }
    pub fn backwards_mode_grad(&self, add_to_computation_graph: bool) -> BackwardsModeGradOutput {
        return match self {
            f64ad::f64(_) => { panic!("cannot compute gradient on f64.") }
            f64ad::f64ad_var_1(_) => {
                assert_eq!(add_to_computation_graph, false);
                f64ad_universal_backwards_mode_grad(self.clone(), false)
            }
            f64ad::f64ad_var_f(_) => {
                f64ad_universal_backwards_mode_grad(self.clone(), add_to_computation_graph)
            }
            f64ad::f64ad_var_t(_) => { panic!("cannot compute gradient on f64ad_var_t") }
            f64ad::f64ad_var_l(_) => {
                assert_eq!(add_to_computation_graph, false);
                f64ad_universal_backwards_mode_grad(self.clone(), add_to_computation_graph)
            }
        };
    }
    #[inline(always)]
    pub fn computation_graph(&self) -> &'static ComputationGraph {
        match self {
            f64ad::f64(_) => { panic!("no computation graph on f64.") }
            f64ad::f64ad_var_1(v) => { v.computation_graph() }
            f64ad::f64ad_var_f(v) => { v.computation_graph() }
            f64ad::f64ad_var_t(v) => { v.computation_graph() }
            f64ad::f64ad_var_l(v) => { v.computation_graph() }
        }
    }

    pub fn to_bits(&self) -> u64 {
        self.value().to_bits()
    }
    pub fn from_bits(v: u64) -> Self {
        f64ad::f64(f64::from_bits(v))
    }
    pub const MAX: Self = f64ad::f64(f64::MAX);
    pub const MIN: Self = f64ad::f64(f64::MIN);
    pub const EPSILON: Self = f64ad::f64(f64::EPSILON);
}

#[derive(Clone, Debug, PartialEq, Eq, Copy)]
pub enum F64adType {
    F64,
    Var1,
    VarF,
    VarT,
    VarL
}
pub enum ComputationGraph {
    ComputationGraph1(RefCell<ComputationGraph1>),
    ComputationGraphF(RefCell<ComputationGraphF>),
    ComputationGraphT(RefCell<ComputationGraphT>),
    ComputationGraphL(RefCell<ComputationGraphL>)
}
impl ComputationGraph {
    pub (crate) fn new(computation_graph_type: ComputationGraphType) -> Self {
        match computation_graph_type {
            ComputationGraphType::ComputationGraph1 => {
                Self::ComputationGraph1(RefCell::new(ComputationGraph1::new()))
            }
            ComputationGraphType::ComputationGraphF => {
                Self::ComputationGraphF(RefCell::new(ComputationGraphF::new()))
            }
            ComputationGraphType::ComputationGraphT => {
                Self::ComputationGraphT(RefCell::new(ComputationGraphT::new()))
            }
            ComputationGraphType::ComputationGraphL => {
                panic!("Cannot initialize a locked computation graph in this way.  Must be done through through global structure.")
            }
        }
    }
    #[inline(always)]
    pub(crate) fn add_node(&'static self, value: f64, node_type_class: NodeTypeClass, node_operands_mode: NodeOperandsMode, parent_0: Option<f64ad>, parent_1: Option<f64ad>) -> f64ad {
        return match self {
            ComputationGraph::ComputationGraph1(c) => {
                if c.borrow().paused() {
                    f64ad::f64(value)
                } else {
                    c.borrow().add_node(value, node_type_class, node_operands_mode, parent_0, parent_1, self)
                }
            }
            ComputationGraph::ComputationGraphF(c) => {
                c.borrow().add_node(value, node_type_class, node_operands_mode, parent_0, parent_1, self)
            }
            ComputationGraph::ComputationGraphT(c) => {
                c.borrow().add_node(value, node_type_class, node_operands_mode, parent_0, parent_1, self)
            }
            ComputationGraph::ComputationGraphL(c) => {
                c.borrow().add_node(value, node_type_class, node_operands_mode, parent_0, parent_1, self)
            }
        };
    }
    pub(crate) fn spawn_variable(&'static self, value: f64) -> f64ad {
        match self {
            ComputationGraph::ComputationGraph1(c) => {
                c.borrow().add_node(value, NodeTypeClass::InputVariable, NodeOperandsMode::NoParents, None, None, self)
            }
            ComputationGraph::ComputationGraphF(c) => {
                c.borrow().add_node(value, NodeTypeClass::InputVariable, NodeOperandsMode::NoParents, None, None, self)
            }
            ComputationGraph::ComputationGraphT(c) => {
                c.borrow().add_node(value, NodeTypeClass::InputVariable, NodeOperandsMode::NoParents, None, None, self)
            }
            ComputationGraph::ComputationGraphL(c) => {
                c.borrow().add_node(value, NodeTypeClass::InputVariable, NodeOperandsMode::NoParents, None, None, self)
            }
        }
    }
    #[inline(always)]
    pub(crate) fn get_node_value(&self, node_idx: usize) -> f64 {
        match self {
            ComputationGraph::ComputationGraph1(c) => {
                c.borrow().computation_graph().borrow().item(node_idx).value()
            }
            ComputationGraph::ComputationGraphF(c) => {
                c.borrow().computation_graph().borrow().item(node_idx).value()
            }
            ComputationGraph::ComputationGraphT(c) => {
                c.borrow().computation_graph().borrow()[node_idx].value()
            }
            ComputationGraph::ComputationGraphL(c) => {
                c.borrow().locked_nodes().borrow()[node_idx].value()
            }
        }
    }
    #[inline(always)]
    /// Returns the node parents, node type class, and node operands mode
    pub(crate) fn get_node_bundle(&self, node_idx: usize) -> ([Option<f64ad>; 2], NodeTypeClass, NodeOperandsMode) {
        return match self {
            ComputationGraph::ComputationGraph1(c) => {
                let binding0 = c.borrow();
                let binding1 = binding0.computation_graph().borrow();
                let node = binding1.item(node_idx);
                ([node.parent_0().clone(), node.parent_1().clone()], node.node_type_class(), node.node_operands_mode())
            }
            ComputationGraph::ComputationGraphF(c) => {
                let binding0 = c.borrow();
                let binding1 = binding0.computation_graph().borrow();
                let node = binding1.item(node_idx);
                ([node.parent_0().clone(), node.parent_1().clone()], node.node_type_class(), node.node_operands_mode())
            }
            ComputationGraph::ComputationGraphT(_) => {
                unreachable!()
            }
            ComputationGraph::ComputationGraphL(c) => {
                let binding0 = c.borrow();
                let binding1 = binding0.locked_nodes().borrow();
                let node = &binding1[node_idx];
                ([node.parent_0().clone(), node.parent_1().clone()], node.node_type_class(), node.node_operands_mode())
            }
        };
    }
    #[inline(always)]
    pub fn num_nodes(&self) -> usize {
        match self {
            ComputationGraph::ComputationGraph1(c) => { c.borrow().num_nodes() }
            ComputationGraph::ComputationGraphF(c) => { c.borrow().num_nodes() }
            ComputationGraph::ComputationGraphT(c) => { c.borrow().num_nodes() }
            ComputationGraph::ComputationGraphL(c) => { c.borrow().num_nodes() }
        }
    }
    #[inline(always)]
    pub(crate) fn map_to_type(&self) -> ComputationGraphType {
        match self {
            ComputationGraph::ComputationGraph1(_) => { ComputationGraphType::ComputationGraph1 }
            ComputationGraph::ComputationGraphF(_) => { ComputationGraphType::ComputationGraphF }
            ComputationGraph::ComputationGraphT(_) => { ComputationGraphType::ComputationGraphT }
            ComputationGraph::ComputationGraphL(_) => { ComputationGraphType::ComputationGraphL }
        }
    }
    #[inline(always)]
    pub (crate) fn computation_graph_id(&self) -> usize {
        match self {
            ComputationGraph::ComputationGraph1(c) => { c.borrow().computation_graph_id() }
            ComputationGraph::ComputationGraphF(c) => { c.borrow().computation_graph_id() }
            ComputationGraph::ComputationGraphT(c) => { c.borrow().computation_graph_id() }
            ComputationGraph::ComputationGraphL(c) => { c.borrow().computation_graph_id() }
        }
    }
    pub (crate) fn reset(&self) {
        match self {
            ComputationGraph::ComputationGraph1(c) => { c.borrow_mut().soft_reset(); }
            ComputationGraph::ComputationGraphF(c) => { c.borrow_mut().soft_reset(); }
            ComputationGraph::ComputationGraphT(c) => { c.borrow_mut().reset()  }
            ComputationGraph::ComputationGraphL(c) => { c.borrow_mut().reset() }
        }
    }
    #[inline(always)]
    #[allow(dead_code)]
    pub (crate) fn pause(&self) {
        match self {
            ComputationGraph::ComputationGraph1(c) => {
                c.borrow_mut().paused = true;
            }
            _ => { panic!("can only pause a ComputationGraph1") }
        }
    }
    #[inline(always)]
    #[allow(dead_code)]
    pub (crate) fn unpause(&self) {
        match self {
            ComputationGraph::ComputationGraph1(c) => {
                c.borrow_mut().paused = false;
            }
            _ => { panic!("can only pause a ComputationGraph1") }
        }
    }
}

unsafe impl Sync for ComputationGraph {}
unsafe impl Send for ComputationGraph {}

#[derive(Clone, Debug, PartialEq, Eq, Copy, Hash)]
pub enum ComputationGraphType {
    ComputationGraph1,
    ComputationGraphF,
    ComputationGraphT,
    ComputationGraphL
}

#[derive(Clone)]
pub struct GlobalComputationGraph(*const ComputationGraph);
impl GlobalComputationGraph {
    #[inline(always)]
    pub fn spawn_variable(&self, value: f64) -> f64ad {
        return unsafe { (*self.0).spawn_variable(value) };
    }
    #[inline(always)]
    pub fn add_node(&self, value: f64, node_type_class: NodeTypeClass, node_operands_mode: NodeOperandsMode, parent_0: Option<f64ad>, parent_1: Option<f64ad>) -> f64ad {
        return unsafe { (*self.0).add_node(value, node_type_class, node_operands_mode, parent_0, parent_1) };
    }
    #[inline(always)]
    pub fn get_value(&self, node_idx: usize) -> f64 {
        return unsafe { (*self.0).get_node_value(node_idx) };
    }
    #[inline(always)]
    pub fn map_to_type(&self) -> ComputationGraphType {
        return unsafe { (*self.0).map_to_type() };
    }
    #[inline(always)]
    pub (crate) fn computation_graph(&self) -> &'static ComputationGraph {
        return unsafe { &(*self.0) }
    }
    #[inline(always)]
    pub fn computation_graph_id(&self) -> usize {
        return unsafe { (*self.0).computation_graph_id() };
    }
    pub fn lock(&self, name: Option<&str>, idx: Option<usize>) {
        let c = self.computation_graph();
        match c {
            ComputationGraph::ComputationGraphT(c) => {
                let binding0 = c.borrow();
                let binding1 = binding0.computation_graph().borrow();

                let mut rng = thread_rng();
                let id: usize = rng.gen();

                let locked_computation_graph = ComputationGraphL {
                    computation_graph_id: id,
                    locked_nodes: RefCell::new(vec![]),
                    count: RefCell::new(0)
                };

                for node in binding1.iter() {
                    locked_computation_graph.locked_nodes.borrow_mut().push(F64ADNodeL {
                        node_idx: node.node_idx(),
                        node_type_class: node.node_type_class(),
                        node_operands_mode: node.node_operands_mode(),
                        value: node.value(),
                        parent_0: node.parent_0().clone(),
                        parent_1: node.parent_1().clone()
                    });
                }

                let computation_graph = ComputationGraph::ComputationGraphL(RefCell::new(locked_computation_graph));

                let hashmap = unsafe { _GLOBAL_COMPUTATION_GRAPHS.get_or_init(|| Mutex::new(HashMap::new())) };

                let name = match name {
            None => { "".to_string() }
            Some(name) => { name.to_string() }
        };
                let idx = match idx {
            None => { 0 }
            Some(idx) => { idx }
        };

                let mut binding = hashmap.lock().unwrap();

                binding.insert((name, idx, ComputationGraphType::ComputationGraphL), computation_graph);
            }
            _ => { panic!("can only lock a tracer graph") }
        }
    }
    pub fn reset(&self) {
        return unsafe { (*self.0).reset() };
    }
}

pub struct GenericComputationGraph<T> {
    nodes: Vec<T>,
    curr_idx: usize,
    curr_len: usize,
}
impl<T> GenericComputationGraph<T> {
    pub fn new() -> Self {
        Self {
            nodes: Vec::with_capacity(1_000_000),
            curr_idx: 0,
            curr_len: 0,
        }
    }
    #[inline(always)]
    pub fn push(&mut self, item: T) {
        if self.curr_idx == self.curr_len {
            self.nodes.push(item);
            self.curr_len += 1;
        } else {
            self.nodes[self.curr_idx] = item;
        }

        self.curr_idx += 1;
    }
    pub fn reset(&mut self) {
        if self.curr_len > 10_000_000 {
            self.nodes = Vec::with_capacity(1_000_000);
            self.curr_idx = 0;
            self.curr_len = 0;
        } else {
            self.curr_idx = 0;
        }
    }
    #[inline(always)]
    pub fn curr_idx(&self) -> usize {
        self.curr_idx
    }
    #[inline(always)]
    pub fn item(&self, idx: usize) -> &T {
        assert!(idx < self.curr_idx, "idx: {}, self_idx: {}", idx, self.curr_idx);
        return &self.nodes[idx];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static mut _GLOBAL_COMPUTATION_GRAPHS: OnceCell<Mutex<HashMap<(String, usize, ComputationGraphType), ComputationGraph>>> = OnceCell::new();

pub struct GlobalComputationGraphs;
impl GlobalComputationGraphs {
    /// This function should be used to access a `GlobalComputationGraph`.  The `name` and `idx`
    /// inputs here can be used to access different graphs.  This even works across threads
    /// allowing for nice multithreaded automatic differentiation.  NOTE: `GlobalComputationGraph`
    /// objects should be reset whenever the previous computation is fully complete, this will
    /// help avoid an excessive use of memory.
    pub fn get(name: Option<&str>, idx: Option<usize>) -> GlobalComputationGraph {
        return Self::get_internal(name, idx, ComputationGraphType::ComputationGraphF);
    }
    fn get_internal(name: Option<&str>, idx: Option<usize>, computation_graph_type: ComputationGraphType) -> GlobalComputationGraph {
        let hashmap = unsafe { _GLOBAL_COMPUTATION_GRAPHS.get_or_init(|| Mutex::new(HashMap::new())) };

        let name = match name {
            None => { "".to_string() }
            Some(name) => { name.to_string() }
        };
        let idx = match idx {
            None => { 0 }
            Some(idx) => { idx }
        };

        let mut binding = hashmap.lock().unwrap();

        let res = binding.get(&(name.clone(), idx, computation_graph_type));
        return match res {
            None => {
                binding.insert((name.clone(), idx, computation_graph_type), ComputationGraph::new(computation_graph_type));
                drop(binding);
                Self::get_internal(Some(&name), Some(idx), computation_graph_type)
            }
            Some(computation_graph) => {
                let r: *const ComputationGraph = computation_graph;
                return GlobalComputationGraph(r);
            }
        };
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

fn f64ad_universal_backwards_mode_grad(v: f64ad, add_to_computation_graph: bool) -> BackwardsModeGradOutput {
    let l = v.computation_graph().num_nodes();
    let mut derivs = vec![f64ad::f64(0.0); l];
    derivs[v.node_idx()] = f64ad::f64(1.0);

    let computation_graph = v.computation_graph();
    'l: for node_idx in (0..l).rev() {
        let (parents, node_type_class, operands_mode) = computation_graph.get_node_bundle(node_idx);
        if node_type_class == NodeTypeClass::InputVariable { continue 'l; }
        let derivatives = compute_derivatives(parents[0].unwrap(), parents[1], node_type_class, operands_mode, add_to_computation_graph);
        match operands_mode {
            NodeOperandsMode::TwoParents => {
                let parent0 = parents[0].unwrap();
                let parent1 = parents[1].unwrap();
                let curr_deriv = derivs[node_idx];
                derivs[parent0.node_idx()] += curr_deriv * derivatives[0];
                derivs[parent1.node_idx()] += curr_deriv * derivatives[1];
            }
            NodeOperandsMode::OneParentLHS => {
                let parent0 = parents[0].unwrap();
                let curr_deriv = derivs[node_idx];
                derivs[parent0.node_idx()] += curr_deriv * derivatives[0];
            }
            NodeOperandsMode::OneParentRHS => {
                let parent1 = parents[1].unwrap();
                let curr_deriv = derivs[node_idx];
                derivs[parent1.node_idx()] += curr_deriv * derivatives[0];
            }
            NodeOperandsMode::NoParents => { }
        }
    }

    return BackwardsModeGradOutput { derivs };
}

fn f64ad_universal_forward_mode_grad(v: f64ad, add_to_computation_graph: bool) -> ForwardModeGradOutput {
    let l = v.computation_graph().num_nodes();
    let mut derivs = vec![f64ad::f64(0.0); l];
    derivs[v.node_idx()] = f64ad::f64(1.0);

    let computation_graph = v.computation_graph();
    'l: for node_idx in 0..l {
        let (parents, node_type_class, operands_mode) = computation_graph.get_node_bundle(node_idx);
        if node_type_class == NodeTypeClass::InputVariable { continue 'l; }
        let derivatives = compute_derivatives(parents[0].unwrap(), parents[1], node_type_class, operands_mode, add_to_computation_graph);
        match operands_mode {
            NodeOperandsMode::TwoParents => {
                let parent0 = parents[0].unwrap();
                let parent1 = parents[1].unwrap();
                let d0 = derivs[parent0.node_idx()];
                let d1 = derivs[parent1.node_idx()];
                derivs[node_idx] += d0 * derivatives[0];
                derivs[node_idx] += d1 * derivatives[1];
            }
            NodeOperandsMode::OneParentLHS => {
                let parent0 = parents[0].unwrap();
                let d0 = derivs[parent0.node_idx()];
                derivs[node_idx] += d0 * derivatives[0];
            }
            NodeOperandsMode::OneParentRHS => {
                let parent1 = parents[1].unwrap();
                let d1 = derivs[parent1.node_idx()];
                derivs[node_idx] += d1 * derivatives[0];
            }
            NodeOperandsMode::NoParents => {}
        }
    }

    return ForwardModeGradOutput { derivs };
}

fn convert_to_f64_if_not_add_to_computation_graph(v: f64ad, add_to_computation_graph: bool) -> f64ad {
    return if !add_to_computation_graph { f64ad::f64(v.value()) } else { v };
}

#[derive(Clone, Debug)]
pub struct ForwardModeGradOutput {
    derivs: Vec<f64ad>,
}
impl ForwardModeGradOutput {
    pub fn wrt(&self, output: &f64ad) -> f64ad {
        return self.derivs[output.node_idx() as usize];
    }
}

#[derive(Clone, Debug)]
pub struct BackwardsModeGradOutput {
    derivs: Vec<f64ad>,
}
impl BackwardsModeGradOutput {
    pub fn wrt(&self, input: &f64ad) -> f64ad {
        return self.derivs[input.node_idx() as usize];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug, Copy, PartialEq)]
pub enum NodeTypeClass {
    InputVariable,
    Add,
    Mul,
    Sub,
    Div,
    Neg,
    Abs,
    Signum,
    Max,
    Min,
    Atan2,
    Floor,
    Ceil,
    Round,
    Trunc,
    Fract,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
    Log,
    Sqrt,
    Exp,
    Powf,
    Manual { value: f64, derivative: f64 }
}

#[derive(Clone, Debug, Copy, PartialEq)]
pub enum NodeOperandsMode {
    TwoParents,
    OneParentLHS,
    OneParentRHS,
    NoParents,
}

#[inline(always)]
pub (crate) fn f64ad_universal_function(lhs: f64ad, rhs: Option<f64ad>, node_type_class: NodeTypeClass) -> f64ad {
    let t = lhs.map_to_type();
    if t == F64adType::VarL {

    }

    match rhs {
        None => { f64ad_universal_function_1_operand(lhs, node_type_class) }
        Some(rhs) => { f64ad_universal_function_2_operands(lhs, rhs, node_type_class) }
    }
}

#[inline(always)]
fn f64ad_universal_function_2_operands(lhs: f64ad, rhs: f64ad, node_type_class: NodeTypeClass) -> f64ad {
    let t0 = lhs.map_to_type();
    let t1 = rhs.map_to_type();

    let t0_is_f64 = t0 == F64adType::F64;
    let t1_is_f64 = t1 == F64adType::F64;

    let operands_mode = if t0_is_f64 && t1_is_f64 {
        NodeOperandsMode::NoParents
    } else if t0_is_f64 {
        NodeOperandsMode::OneParentRHS
    } else if t1_is_f64 {
        NodeOperandsMode::OneParentLHS
    } else {
        assert_eq!(t0, t1);
        assert_eq!(lhs.computation_graph().computation_graph_id(), rhs.computation_graph().computation_graph_id());

        NodeOperandsMode::TwoParents
    };

    compute_value_f64ad(lhs, Some(rhs), node_type_class, operands_mode)
}

#[inline(always)]
fn f64ad_universal_function_1_operand(lhs: f64ad, node_type_class: NodeTypeClass) -> f64ad {
    let t0 = lhs.map_to_type();

    let operands_mode = if t0 == F64adType::F64 {
        NodeOperandsMode::NoParents
    } else {
        NodeOperandsMode::OneParentLHS
    };

    compute_value_f64ad(lhs, None, node_type_class, operands_mode)
}

#[inline(always)]
/// Will panic if variants are not valid
pub (crate) fn f64ad_function_2_operands_valid_variants(lhs: f64ad, rhs: f64ad) {
    let t0 = lhs.map_to_type();
    let t1 = rhs.map_to_type();

    let t0_is_f64 = t0 == F64adType::F64;
    let t1_is_f64 = t1 == F64adType::F64;

    if t0_is_f64 || t1_is_f64 { return; }

    if t0 != t1 { panic!("variants are not valid: {:?}, {:?}", t0, t1) }
}

#[inline(always)]
fn compute_value_f64(lhs: f64ad, rhs: Option<f64ad>, node_type_class: NodeTypeClass) -> f64 {
    match node_type_class {
        NodeTypeClass::InputVariable => { panic!("input variable cannot compute value.") }
        NodeTypeClass::Add => { lhs.value() + rhs.unwrap().value() }
        NodeTypeClass::Mul => { lhs.value() * rhs.unwrap().value() }
        NodeTypeClass::Sub => { lhs.value() - rhs.unwrap().value() }
        NodeTypeClass::Div => { lhs.value() / rhs.unwrap().value() }
        NodeTypeClass::Neg => { -lhs.value() }
        NodeTypeClass::Abs => { lhs.value().abs() }
        NodeTypeClass::Signum => { lhs.value().signum() }
        NodeTypeClass::Max => { lhs.value().max(rhs.unwrap().value()) }
        NodeTypeClass::Min => { lhs.value().min(rhs.unwrap().value()) }
        NodeTypeClass::Atan2 => { lhs.value().atan2(rhs.unwrap().value()) }
        NodeTypeClass::Floor => { lhs.value().floor() }
        NodeTypeClass::Ceil => { lhs.value().ceil() }
        NodeTypeClass::Round => { lhs.value().round() }
        NodeTypeClass::Trunc => { lhs.value().trunc() }
        NodeTypeClass::Fract => { lhs.value().fract() }
        NodeTypeClass::Sin => { lhs.value().sin() }
        NodeTypeClass::Cos => { lhs.value().cos() }
        NodeTypeClass::Tan => { lhs.value().tan() }
        NodeTypeClass::Asin => { lhs.value().asin() }
        NodeTypeClass::Acos => { lhs.value().acos() }
        NodeTypeClass::Atan => { lhs.value().atan() }
        NodeTypeClass::Sinh => { lhs.value().sinh() }
        NodeTypeClass::Cosh => { lhs.value().cosh() }
        NodeTypeClass::Tanh => { lhs.value().tanh() }
        NodeTypeClass::Asinh => { lhs.value().asinh() }
        NodeTypeClass::Acosh => { lhs.value().acosh() }
        NodeTypeClass::Atanh => { lhs.value().atanh() }
        NodeTypeClass::Log => { lhs.value().log(rhs.unwrap().value()) }
        NodeTypeClass::Sqrt => { lhs.value().sqrt() }
        NodeTypeClass::Exp => { lhs.value().exp() }
        NodeTypeClass::Powf => { lhs.value().powf(rhs.unwrap().value()) }
        NodeTypeClass::Manual { value, .. } => { value }
    }
}

#[inline(always)]
fn compute_value_f64ad(lhs: f64ad, rhs: Option<f64ad>, node_type_class: NodeTypeClass, operands_mode: NodeOperandsMode) -> f64ad {
    let value = compute_value_f64(lhs, rhs, node_type_class);
    return match operands_mode {
        NodeOperandsMode::TwoParents => {
            lhs.computation_graph().add_node(value, node_type_class, operands_mode, Some(lhs), rhs)
        }
        NodeOperandsMode::OneParentLHS => {
            lhs.computation_graph().add_node(value, node_type_class, operands_mode, Some(lhs), rhs)
        }
        NodeOperandsMode::OneParentRHS => {
            rhs.unwrap().computation_graph().add_node(value, node_type_class, operands_mode, Some(lhs), rhs)
        }
        NodeOperandsMode::NoParents => {
            f64ad::f64(value)
        }
    };
}

#[inline(always)]
fn compute_derivatives(lhs: f64ad, rhs: Option<f64ad>, node_type_class: NodeTypeClass, operands_mode: NodeOperandsMode, add_to_computation_graph: bool) -> TinyVec<[f64ad; 2]> {
    let lhs = convert_to_f64_if_not_add_to_computation_graph(lhs, add_to_computation_graph);
    let rhs = match rhs {
        None => { None }
        Some(rhs) => { Some(convert_to_f64_if_not_add_to_computation_graph(rhs, add_to_computation_graph)) }
    };

    match node_type_class {
        NodeTypeClass::InputVariable => { tiny_vec!([f64ad; 2]) }
        NodeTypeClass::Add => {
            match operands_mode {
                NodeOperandsMode::TwoParents => { tiny_vec!([f64ad; 2] => f64ad::f64(1.0), f64ad::f64(1.0)) }
                NodeOperandsMode::OneParentLHS => { tiny_vec!([f64ad; 2] => f64ad::f64(1.0)) }
                NodeOperandsMode::OneParentRHS => { tiny_vec!([f64ad; 2] => f64ad::f64(1.0)) }
                NodeOperandsMode::NoParents => { tiny_vec!([f64ad; 2]) }
            }
        }
        NodeTypeClass::Mul => {
            match operands_mode {
                NodeOperandsMode::TwoParents => { tiny_vec!([f64ad; 2] => rhs.unwrap(), lhs) }
                NodeOperandsMode::OneParentLHS => { tiny_vec!([f64ad; 2] => rhs.unwrap()) }
                NodeOperandsMode::OneParentRHS => { tiny_vec!([f64ad; 2] => lhs) }
                NodeOperandsMode::NoParents => { tiny_vec!([f64ad; 2]) }
            }
        }
        NodeTypeClass::Sub => {
            match operands_mode {
                NodeOperandsMode::TwoParents => { tiny_vec!([f64ad; 2] => f64ad::f64(1.0), f64ad::f64(-1.0)) }
                NodeOperandsMode::OneParentLHS => { tiny_vec!([f64ad; 2] => f64ad::f64(1.0)) }
                NodeOperandsMode::OneParentRHS => { tiny_vec!([f64ad; 2] => f64ad::f64(-1.0)) }
                NodeOperandsMode::NoParents => { tiny_vec!([f64ad; 2]) }
            }
        }
        NodeTypeClass::Div => {
            match operands_mode {
                NodeOperandsMode::TwoParents => {
                    let rhs = rhs.unwrap();
                    tiny_vec!([f64ad; 2] => 1.0/rhs, -lhs / (rhs*rhs))
                }
                NodeOperandsMode::OneParentLHS => {
                    let rhs = rhs.unwrap();
                    tiny_vec!([f64ad; 2] => 1.0/rhs)
                }
                NodeOperandsMode::OneParentRHS => {
                    let rhs = rhs.unwrap();
                    tiny_vec!([f64ad; 2] => -lhs / (rhs*rhs))
                }
                NodeOperandsMode::NoParents => { tiny_vec!([f64ad; 2]) }
            }
        }
        NodeTypeClass::Neg => {
            match operands_mode {
                NodeOperandsMode::TwoParents => { unreachable!() }
                NodeOperandsMode::OneParentLHS => { tiny_vec!([f64ad; 2] => f64ad::f64(-1.0)) }
                NodeOperandsMode::OneParentRHS => { unreachable!() }
                NodeOperandsMode::NoParents => { tiny_vec!([f64ad; 2]) }
            }
        }
        NodeTypeClass::Abs => {
            match operands_mode {
                NodeOperandsMode::TwoParents => { unreachable!() }
                NodeOperandsMode::OneParentLHS => {
                    let val = lhs.value();
                    if val >= 0.0 { return tiny_vec!([f64ad; 2] => f64ad::f64(1.0)); } else { return tiny_vec!([f64ad; 2] => f64ad::f64(-1.0)); }
                }
                NodeOperandsMode::OneParentRHS => { unreachable!() }
                NodeOperandsMode::NoParents => { tiny_vec!([f64ad; 2]) }
            }
        }
        NodeTypeClass::Signum => {
            match operands_mode {
                NodeOperandsMode::TwoParents => { unreachable!() }
                NodeOperandsMode::OneParentLHS => { tiny_vec!([f64ad; 2] => f64ad::f64(0.0)) }
                NodeOperandsMode::OneParentRHS => { unreachable!() }
                NodeOperandsMode::NoParents => { tiny_vec!([f64ad; 2]) }
            }
        }
        NodeTypeClass::Max => {
            match operands_mode {
                NodeOperandsMode::TwoParents => {
                    if lhs >= rhs.unwrap() {
                        tiny_vec!([f64ad; 2] => f64ad::f64(1.0), f64ad::f64(0.0))
                    } else {
                        tiny_vec!([f64ad; 2] => f64ad::f64(0.0), f64ad::f64(1.0))
                    }
                }
                NodeOperandsMode::OneParentLHS => {
                    if lhs >= rhs.unwrap() {
                        tiny_vec!([f64ad; 2] => f64ad::f64(1.0))
                    } else {
                        tiny_vec!([f64ad; 2] => f64ad::f64(0.0))
                    }
                }
                NodeOperandsMode::OneParentRHS => {
                    if lhs >= rhs.unwrap() {
                        tiny_vec!([f64ad; 2] => f64ad::f64(0.0))
                    } else {
                        tiny_vec!([f64ad; 2] => f64ad::f64(1.0))
                    }
                }
                NodeOperandsMode::NoParents => {
                    tiny_vec!([f64ad; 2])
                }
            }
        }
        NodeTypeClass::Min => {
            match operands_mode {
                NodeOperandsMode::TwoParents => {
                    if lhs <= rhs.unwrap() {
                        tiny_vec!([f64ad; 2] => f64ad::f64(1.0), f64ad::f64(0.0))
                    } else {
                        tiny_vec!([f64ad; 2] => f64ad::f64(0.0), f64ad::f64(1.0))
                    }
                }
                NodeOperandsMode::OneParentLHS => {
                    if lhs <= rhs.unwrap() {
                        tiny_vec!([f64ad; 2] => f64ad::f64(1.0))
                    } else {
                        tiny_vec!([f64ad; 2] => f64ad::f64(0.0))
                    }
                }
                NodeOperandsMode::OneParentRHS => {
                    if lhs <= rhs.unwrap() {
                        tiny_vec!([f64ad; 2] => f64ad::f64(0.0))
                    } else {
                        tiny_vec!([f64ad; 2] => f64ad::f64(1.0))
                    }
                }
                NodeOperandsMode::NoParents => {
                    tiny_vec!([f64ad; 2])
                }
            }
        }
        NodeTypeClass::Atan2 => {
            match operands_mode {
                NodeOperandsMode::TwoParents => {
                    let rhs = rhs.unwrap();
                    tiny_vec!([f64ad; 2] => rhs/(lhs*lhs + rhs*rhs), -lhs/(lhs*lhs + rhs*rhs))
                }
                NodeOperandsMode::OneParentLHS => {
                    let rhs = rhs.unwrap();
                    tiny_vec!([f64ad; 2] => rhs/(lhs*lhs + rhs*rhs))
                }
                NodeOperandsMode::OneParentRHS => {
                    let rhs = rhs.unwrap();
                    tiny_vec!([f64ad; 2] => -lhs/(lhs*lhs + rhs*rhs))
                }
                NodeOperandsMode::NoParents => {
                    tiny_vec!([f64ad; 2])
                }
            }
        }
        NodeTypeClass::Floor => {
            match operands_mode {
                NodeOperandsMode::TwoParents => { unreachable!() }
                NodeOperandsMode::OneParentLHS => { tiny_vec!([f64ad; 2] => f64ad::f64(0.0)) }
                NodeOperandsMode::OneParentRHS => { unreachable!() }
                NodeOperandsMode::NoParents => { tiny_vec!([f64ad; 2]) }
            }
        }
        NodeTypeClass::Ceil => {
            match operands_mode {
                NodeOperandsMode::TwoParents => { unreachable!() }
                NodeOperandsMode::OneParentLHS => { tiny_vec!([f64ad; 2] => f64ad::f64(0.0)) }
                NodeOperandsMode::OneParentRHS => { unreachable!() }
                NodeOperandsMode::NoParents => { tiny_vec!([f64ad; 2]) }
            }
        }
        NodeTypeClass::Round => {
            match operands_mode {
                NodeOperandsMode::TwoParents => { unreachable!() }
                NodeOperandsMode::OneParentLHS => { tiny_vec!([f64ad; 2] => f64ad::f64(0.0)) }
                NodeOperandsMode::OneParentRHS => { unreachable!() }
                NodeOperandsMode::NoParents => { tiny_vec!([f64ad; 2]) }
            }
        }
        NodeTypeClass::Trunc => {
            match operands_mode {
                NodeOperandsMode::TwoParents => { unreachable!() }
                NodeOperandsMode::OneParentLHS => { tiny_vec!([f64ad; 2] => f64ad::f64(0.0)) }
                NodeOperandsMode::OneParentRHS => { unreachable!() }
                NodeOperandsMode::NoParents => { tiny_vec!([f64ad; 2]) }
            }
        }
        NodeTypeClass::Fract => {
            match operands_mode {
                NodeOperandsMode::TwoParents => { unreachable!() }
                NodeOperandsMode::OneParentLHS => { tiny_vec!([f64ad; 2] => f64ad::f64(0.0)) }
                NodeOperandsMode::OneParentRHS => { unreachable!() }
                NodeOperandsMode::NoParents => { tiny_vec!([f64ad; 2]) }
            }
        }
        NodeTypeClass::Sin => {
            match operands_mode {
                NodeOperandsMode::TwoParents => { unreachable!() }
                NodeOperandsMode::OneParentLHS => { tiny_vec!([f64ad; 2] => lhs.cos()) }
                NodeOperandsMode::OneParentRHS => { unreachable!() }
                NodeOperandsMode::NoParents => { tiny_vec!([f64ad; 2]) }
            }
        }
        NodeTypeClass::Cos => {
            match operands_mode {
                NodeOperandsMode::TwoParents => { unreachable!() }
                NodeOperandsMode::OneParentLHS => { tiny_vec!([f64ad; 2] => -lhs.sin()) }
                NodeOperandsMode::OneParentRHS => { unreachable!() }
                NodeOperandsMode::NoParents => { tiny_vec!([f64ad; 2]) }
            }
        }
        NodeTypeClass::Tan => {
            match operands_mode {
                NodeOperandsMode::TwoParents => { unreachable!() }
                NodeOperandsMode::OneParentLHS => {
                    let c = lhs.cos();
                    tiny_vec!([f64ad; 2] => 1.0/(c*c))
                }
                NodeOperandsMode::OneParentRHS => { unreachable!() }
                NodeOperandsMode::NoParents => { tiny_vec!([f64ad; 2]) }
            }
        }
        NodeTypeClass::Asin => {
            match operands_mode {
                NodeOperandsMode::TwoParents => { unreachable!() }
                NodeOperandsMode::OneParentLHS => { tiny_vec!([f64ad; 2] => 1.0 / (1.0 - lhs * lhs).sqrt()) }
                NodeOperandsMode::OneParentRHS => { unreachable!() }
                NodeOperandsMode::NoParents => { tiny_vec!([f64ad; 2]) }
            }
        }
        NodeTypeClass::Acos => {
            match operands_mode {
                NodeOperandsMode::TwoParents => { unreachable!() }
                NodeOperandsMode::OneParentLHS => { tiny_vec!([f64ad; 2] => -1.0/(1.0 - lhs * lhs).sqrt()) }
                NodeOperandsMode::OneParentRHS => { unreachable!() }
                NodeOperandsMode::NoParents => { tiny_vec!([f64ad; 2]) }
            }
        }
        NodeTypeClass::Atan => {
            match operands_mode {
                NodeOperandsMode::TwoParents => { unreachable!() }
                NodeOperandsMode::OneParentLHS => { tiny_vec!([f64ad; 2] => 1.0/(lhs*lhs + 1.0)) }
                NodeOperandsMode::OneParentRHS => { unreachable!() }
                NodeOperandsMode::NoParents => { tiny_vec!([f64ad; 2]) }
            }
        }
        NodeTypeClass::Sinh => {
            match operands_mode {
                NodeOperandsMode::TwoParents => { unreachable!() }
                NodeOperandsMode::OneParentLHS => { tiny_vec!([f64ad; 2] => lhs.cosh()) }
                NodeOperandsMode::OneParentRHS => { unreachable!() }
                NodeOperandsMode::NoParents => { tiny_vec!([f64ad; 2]) }
            }
        }
        NodeTypeClass::Cosh => {
            match operands_mode {
                NodeOperandsMode::TwoParents => { unreachable!() }
                NodeOperandsMode::OneParentLHS => { tiny_vec!([f64ad; 2] => lhs.sinh()) }
                NodeOperandsMode::OneParentRHS => { unreachable!() }
                NodeOperandsMode::NoParents => { tiny_vec!([f64ad; 2]) }
            }
        }
        NodeTypeClass::Tanh => {
            match operands_mode {
                NodeOperandsMode::TwoParents => { unreachable!() }
                NodeOperandsMode::OneParentLHS => {
                    let c = lhs.cosh();
                    tiny_vec!([f64ad; 2] => 1.0 / (c*c))
                }
                NodeOperandsMode::OneParentRHS => { unreachable!() }
                NodeOperandsMode::NoParents => { tiny_vec!([f64ad; 2]) }
            }
        }
        NodeTypeClass::Asinh => {
            match operands_mode {
                NodeOperandsMode::TwoParents => { unreachable!() }
                NodeOperandsMode::OneParentLHS => { tiny_vec!([f64ad; 2] => 1.0/(lhs*lhs + 1.0).sqrt()) }
                NodeOperandsMode::OneParentRHS => { unreachable!() }
                NodeOperandsMode::NoParents => { tiny_vec!([f64ad; 2]) }
            }
        }
        NodeTypeClass::Acosh => {
            match operands_mode {
                NodeOperandsMode::TwoParents => { unreachable!() }
                NodeOperandsMode::OneParentLHS => { tiny_vec!([f64ad; 2] => 1.0/((lhs - 1.0).sqrt()*(lhs + 1.0).sqrt()) ) }
                NodeOperandsMode::OneParentRHS => { unreachable!() }
                NodeOperandsMode::NoParents => { tiny_vec!([f64ad; 2]) }
            }
        }
        NodeTypeClass::Atanh => {
            match operands_mode {
                NodeOperandsMode::TwoParents => { unreachable!() }
                NodeOperandsMode::OneParentLHS => { tiny_vec!([f64ad; 2] => 1.0/(1.0 - lhs*lhs)) }
                NodeOperandsMode::OneParentRHS => { unreachable!() }
                NodeOperandsMode::NoParents => { tiny_vec!([f64ad; 2]) }
            }
        }
        NodeTypeClass::Log => {
            match operands_mode {
                NodeOperandsMode::TwoParents => {
                    let rhs = rhs.unwrap();
                    let ln_rhs = rhs.ln();
                    let ln_lhs = lhs.ln();
                    tiny_vec!([f64ad; 2] => 1.0/(lhs * ln_rhs), -ln_lhs / (rhs * ln_rhs * ln_rhs))
                }
                NodeOperandsMode::OneParentLHS => {
                    let rhs = rhs.unwrap();
                    let ln_rhs = rhs.ln();
                    tiny_vec!([f64ad; 2] => 1.0/(lhs * ln_rhs))
                }
                NodeOperandsMode::OneParentRHS => {
                    let rhs = rhs.unwrap();
                    let ln_rhs = rhs.ln();
                    let ln_lhs = lhs.ln();
                    tiny_vec!([f64ad; 2] => -ln_lhs / (rhs * ln_rhs * ln_rhs))
                }
                NodeOperandsMode::NoParents => {
                    tiny_vec!([f64ad; 2])
                }
            }
        }
        NodeTypeClass::Sqrt => {
            match operands_mode {
                NodeOperandsMode::TwoParents => { unreachable!() }
                NodeOperandsMode::OneParentLHS => { tiny_vec!([f64ad; 2] => 1.0/(2.0*lhs.sqrt())) }
                NodeOperandsMode::OneParentRHS => { unreachable!() }
                NodeOperandsMode::NoParents => { tiny_vec!([f64ad; 2]) }
            }
        }
        NodeTypeClass::Exp => {
            match operands_mode {
                NodeOperandsMode::TwoParents => { unreachable!() }
                NodeOperandsMode::OneParentLHS => { tiny_vec!([f64ad; 2] => lhs.exp()) }
                NodeOperandsMode::OneParentRHS => { unreachable!() }
                NodeOperandsMode::NoParents => { tiny_vec!([f64ad; 2]) }
            }
        }
        NodeTypeClass::Powf => {
            match operands_mode {
                NodeOperandsMode::TwoParents => {
                    let rhs = rhs.unwrap();
                    tiny_vec!([f64ad; 2] => rhs * lhs.powf(rhs - 1.0), lhs.powf(rhs) * lhs.ln())
                }
                NodeOperandsMode::OneParentLHS => {
                    let rhs = rhs.unwrap();
                    tiny_vec!([f64ad; 2] => rhs * lhs.powf(rhs - 1.0))
                }
                NodeOperandsMode::OneParentRHS => {
                    let rhs = rhs.unwrap();
                    tiny_vec!([f64ad; 2] => lhs.powf(rhs) * lhs.ln())
                }
                NodeOperandsMode::NoParents => {
                    tiny_vec!([f64ad; 2])
                }
            }
        }
        NodeTypeClass::Manual { derivative, .. } => {
            tiny_vec!([f64ad; 2] => f64ad::f64(derivative))
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

pub fn f64ad_jacobian(inputs: &[f64ad], outputs: &[f64ad], order: usize) -> JacobianOutput {
    let mut out = JacobianOutput::new();
    for (output_idx, output) in outputs.iter().enumerate() {
        out.push_entry(vec![], output_idx, output.clone());
    }

    let mut curr_order = 0;
    while curr_order < order {
        let add_to_computation_graph = !(curr_order == order - 1);
        out = f64ad_jacobian_internal(inputs, &out, add_to_computation_graph);
        curr_order += 1;
    }

    out.sort();
    out
}

fn f64ad_jacobian_internal(inputs: &[f64ad], outputs: &JacobianOutput, add_to_computation_graph: bool) -> JacobianOutput {
    f64ad_jacobian_internal_nonparallel(inputs, outputs, add_to_computation_graph)
    /*
    return if parallel && !add_to_computation_graph { f64ad_jacobian_internal_parallel(inputs, outputs, add_to_computation_graph) }
    else { f64ad_jacobian_internal_nonparallel(inputs, outputs, add_to_computation_graph) }
    */
}

fn f64ad_jacobian_internal_nonparallel(inputs: &[f64ad], outputs: &JacobianOutput, add_to_computation_graph: bool) -> JacobianOutput {
    let mut out = JacobianOutput::new();

    let num_inputs = inputs.len();
    let num_outputs = outputs.entries.len();

    // forward mode
    if num_inputs <= num_outputs {
        for (input_idx, input) in inputs.iter().enumerate() {
            let grad = input.forward_mode_grad(add_to_computation_graph);

            for output in outputs.entries.iter() {
                let mut new_jacobian_entry_signature = output.signature.clone();
                new_jacobian_entry_signature.add_input_wrt(input_idx);
                let new_value = grad.wrt(&output.value);
                let new_jacobian_entry = JacobianEntry {
                    signature: new_jacobian_entry_signature,
                    value: new_value,
                };
                out.entries.push(new_jacobian_entry);
            }
        }
    }
    // backwards mode
    else {
        for output in outputs.entries.iter() {
            let grad = output.value.backwards_mode_grad(add_to_computation_graph);

            for (input_idx, input) in inputs.iter().enumerate() {
                let mut new_jacobian_entry_signature = output.signature.clone();
                new_jacobian_entry_signature.add_input_wrt(input_idx);
                let new_value = grad.wrt(input);
                let new_jacobian_entry = JacobianEntry {
                    signature: new_jacobian_entry_signature,
                    value: new_value,
                };
                out.entries.push(new_jacobian_entry);
            }
        }
    }

    out
}

#[derive(Clone, Debug)]
pub struct JacobianOutput {
    entries: Vec<JacobianEntry>,
}
impl JacobianOutput {
    pub fn new() -> Self {
        Self {
            entries: vec![]
        }
    }
    fn push_entry(&mut self, inputs_wrt: Vec<usize>, output: usize, value: f64ad) {
        let entry_to_add = JacobianEntry {
            signature: JacobianEntrySignature::new(output, inputs_wrt),
            value,
        };
        self.entries.push(entry_to_add);
    }
    fn sort(&mut self) {
        self.entries.sort_by(|x, y| x.signature.partial_cmp(&y.signature).unwrap());
    }
    #[inline(always)]
    pub fn get_entry(&self, inputs_wrt: Vec<usize>, output: usize) -> Option<&JacobianEntry> {
        let signature = JacobianEntrySignature::new(output, inputs_wrt);
        let binary_search_res = self.entries.binary_search_by(|x| x.signature.partial_cmp(&signature).unwrap());
        return match binary_search_res {
            Ok(i) => { Some(&self.entries[i]) }
            Err(_) => { None }
        };
    }
    pub fn print_summary(&self) {
        for (i, entry) in self.entries.iter().enumerate() {
            println!("{:?} --- output: {:?}, inputs wrt: {:?}, value: {:?}", i, entry.signature.output, entry.signature.inputs_wrt, entry.value.value());
        }
    }
}

#[derive(Clone, Debug)]
pub struct JacobianEntry {
    signature: JacobianEntrySignature,
    value: f64ad,
}
impl JacobianEntry {
    #[inline(always)]
    pub fn signature(&self) -> &JacobianEntrySignature {
        &self.signature
    }
    #[inline(always)]
    pub fn value(&self) -> f64ad {
        self.value
    }
}

#[derive(Clone, Debug, PartialOrd, PartialEq)]
pub struct JacobianEntrySignature {
    output: usize,
    inputs_wrt: Vec<usize>,
}
impl JacobianEntrySignature {
    pub fn new(output: usize, inputs_wrt: Vec<usize>) -> Self {
        Self {
            output,
            inputs_wrt,
        }
    }
    pub fn add_input_wrt(&mut self, input_wrt: usize) {
        self.inputs_wrt.push(input_wrt);
    }
    pub fn inputs_wrt(&self) -> &Vec<usize> {
        &self.inputs_wrt
    }
    #[inline(always)]
    pub fn output(&self) -> usize {
        self.output
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// TODO: These functions are not working yet.  Maybe fix them for ComputationGraph1?
/*
// Todo: this is currently incorrect.  Outputs can have more than one parent, and right now this
// Todo: is only set up for one parent.
pub fn bind_manual_derivative_between_values(i: &f64ad, o: &mut f64ad, derivative: f64) {
    assert_eq!(i.map_to_type(), F64adType::Var1);
    let computation_graph = i.computation_graph();
    computation_graph.unpause();
    let new_value = computation_graph.add_node(o.value(), NodeTypeClass::Manual { value: o.value(), derivative }, NodeOperandsMode::OneParentLHS, Some(i.clone()), None);
    *o = new_value;
    computation_graph.pause();
}

pub fn manual_derivative_function<I: Debug, O: Debug, F0, F1, F2>(
    input: &I,
    f64ad_sampler_function: F0,
    mut function: F1,
    d_f_d_o: F2
) -> O where F0: Fn(&I) -> f64ad,
        F1: FnMut(&I) -> O,
        F2: Fn(&I, &mut O) {

    let sample = f64ad_sampler_function(input);
    return match sample {
        f64ad::f64ad_var_1(v) => {
            v.computation_graph().pause();
            let mut output = function(input);
            d_f_d_o(input, &mut output);
            v.computation_graph().unpause();
            output
        }
        _ => {
            function(input)
        }
    }

}
*/

////////////////////////////////////////////////////////////////////////////////////////////////////

impl Add<f64ad> for f64ad {
    type Output = f64ad;

    fn add(self, rhs: f64ad) -> Self::Output {
        f64ad_universal_function(self, Some(rhs), NodeTypeClass::Add)
    }
}
impl Add<f64> for f64ad {
    type Output = f64ad;

    fn add(self, rhs: f64) -> Self::Output {
        return self + f64ad::f64(rhs);
    }
}
impl Add<f64ad> for f64 {
    type Output = f64ad;

    fn add(self, rhs: f64ad) -> Self::Output {
        return f64ad::f64(self) + rhs;
    }
}
impl AddAssign for f64ad {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl AddAssign<f64> for f64ad {
    fn add_assign(&mut self, rhs: f64) {
        *self = *self + rhs;
    }
}
impl AddAssign<f64ad> for f64 {
    fn add_assign(&mut self, rhs: f64ad) {
        *self += rhs.value();
    }
}

impl Mul<f64ad> for f64ad {
    type Output = f64ad;

    fn mul(self, rhs: f64ad) -> Self::Output {
        f64ad_universal_function(self, Some(rhs), NodeTypeClass::Mul)
    }
}
impl Mul<f64> for f64ad {
    type Output = f64ad;

    fn mul(self, rhs: f64) -> Self::Output {
        return self * f64ad::f64(rhs);
    }
}
impl Mul<f64ad> for f64 {
    type Output = f64ad;

    fn mul(self, rhs: f64ad) -> Self::Output {
        return f64ad::f64(self) * rhs;
    }
}
impl MulAssign for f64ad {

    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl MulAssign<f64> for f64ad {

    fn mul_assign(&mut self, rhs: f64) {
        *self = *self * rhs;
    }
}
impl MulAssign<f64ad> for f64 {

    fn mul_assign(&mut self, rhs: f64ad) {
        *self *= rhs.value();
    }
}

impl Sub<f64ad> for f64ad {
    type Output = f64ad;

    fn sub(self, rhs: f64ad) -> Self::Output {
        f64ad_universal_function(self, Some(rhs), NodeTypeClass::Sub)
    }
}
impl Sub<f64> for f64ad {
    type Output = f64ad;

    fn sub(self, rhs: f64) -> Self::Output {
        return self - f64ad::f64(rhs);
    }
}
impl Sub<f64ad> for f64 {
    type Output = f64ad;

    fn sub(self, rhs: f64ad) -> Self::Output {
        return f64ad::f64(self) - rhs;
    }
}
impl SubAssign for f64ad {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl SubAssign<f64> for f64ad {
    fn sub_assign(&mut self, rhs: f64) {
        *self = *self - rhs;
    }
}
impl SubAssign<f64ad> for f64 {
    fn sub_assign(&mut self, rhs: f64ad) {
        *self -= rhs.value();
    }
}

impl Div<f64ad> for f64ad {
    type Output = f64ad;

    fn div(self, rhs: f64ad) -> Self::Output {
        f64ad_universal_function(self, Some(rhs), NodeTypeClass::Div)
    }
}
impl Div<f64> for f64ad {
    type Output = f64ad;

    fn div(self, rhs: f64) -> Self::Output {
        return self / f64ad::f64(rhs);
    }
}
impl Div<f64ad> for f64 {
    type Output = f64ad;

    fn div(self, rhs: f64ad) -> Self::Output {
        return f64ad::f64(self) / rhs;
    }
}
impl DivAssign for f64ad {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}
impl DivAssign<f64> for f64ad {

    fn div_assign(&mut self, rhs: f64) {
        *self = *self / rhs;
    }
}
impl DivAssign<f64ad> for f64 {

    fn div_assign(&mut self, rhs: f64ad) {
        *self /= rhs.value();
    }
}

impl Rem<f64ad> for f64ad {
    type Output = f64ad;

    fn rem(self, rhs: f64ad) -> Self::Output {
        self - (self / rhs).floor() * rhs
    }
}
impl Rem<f64> for f64ad {
    type Output = f64ad;

    fn rem(self, rhs: f64) -> Self::Output {
        self % f64ad::f64(rhs)
    }
}
impl Rem<f64ad> for f64 {
    type Output = f64ad;

    fn rem(self, rhs: f64ad) -> Self::Output {
        f64ad::f64(self) % rhs
    }
}
impl RemAssign for f64ad {
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}
impl RemAssign<f64> for f64ad {
    fn rem_assign(&mut self, rhs: f64) {
        *self = *self % rhs;
    }
}
impl RemAssign<f64ad> for f64 {
    fn rem_assign(&mut self, rhs: f64ad) {
        *self %= rhs.value();
    }
}

impl Neg for f64ad {
    type Output = f64ad;

    fn neg(self) -> Self::Output {
        f64ad_universal_function(self, None, NodeTypeClass::Neg)
    }
}

impl PartialEq for f64ad {
    fn eq(&self, other: &Self) -> bool {
        assert_ne!(self.map_to_type(), F64adType::VarT, "cannot equate tracer variables.");
        assert_ne!(other.map_to_type(), F64adType::VarT, "cannot equate tracer variables.");

        f64ad_function_2_operands_valid_variants(*self, *other);

        return self.value() == other.value();
    }
}
impl PartialEq<f64> for f64ad {
    fn eq(&self, other: &f64) -> bool {
        return self.eq(&f64ad::f64(*other));
    }
}
impl PartialEq<f64ad> for f64 {
    fn eq(&self, other: &f64ad) -> bool {
        return f64ad::f64(*self).eq(other);
    }
}

impl PartialOrd for f64ad {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        assert_ne!(self.map_to_type(), F64adType::VarT, "cannot order tracer variables.");
        assert_ne!(other.map_to_type(), F64adType::VarT, "cannot order tracer variables.");

        f64ad_function_2_operands_valid_variants(*self, *other);

        return self.value().partial_cmp(&other.value());
    }
}
impl PartialOrd<f64> for f64ad {
    fn partial_cmp(&self, other: &f64) -> Option<Ordering> {
        return self.partial_cmp(&f64ad::f64(*other));
    }
}
impl PartialOrd<f64ad> for f64 {
    fn partial_cmp(&self, other: &f64ad) -> Option<Ordering> {
        return f64ad::f64(*self).partial_cmp(other);
    }
}

impl Display for f64ad {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        f.write_str(&format!("{:?}", self)).expect("error");
        Ok(())
    }
}

impl From<f64> for f64ad {
    fn from(a: f64) -> Self {
        return f64ad::f64(a);
    }
}
impl Into<f64> for f64ad {
    fn into(self) -> f64 {
        self.value()
    }
}
impl From<f32> for f64ad {
    fn from(a: f32) -> Self {
        return f64ad::f64(a as f64);
    }
}
impl Into<f32> for f64ad {
    fn into(self) -> f32 {
        self.value() as f32
    }
}