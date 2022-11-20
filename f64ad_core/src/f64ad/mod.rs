#![allow(dead_code)]

pub mod trait_impls;

use std::cell::{RefCell};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};
use std::sync::{Mutex};
use tinyvec::{tiny_vec, TinyVec};
use rand::prelude::*;
use once_cell::sync::OnceCell;
use simba::scalar::ComplexField;

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy)]
pub enum f64ad {
    f64ad_var(f64ad_var),
    f64ad_locked_var(f64ad_locked_var),
    f64(f64),
}
impl f64ad {
    pub fn value(&self) -> f64 {
        return match self {
            f64ad::f64ad_var(f) => { f.value() }
            f64ad::f64ad_locked_var(f) => { f.value() }
            f64ad::f64(f) => { *f }
        };
    }
    pub fn node_idx(&self) -> usize {
        match self {
            f64ad::f64ad_var(a) => { a.node_idx as usize }
            f64ad::f64ad_locked_var(a) => { a.node_idx as usize }
            f64ad::f64(_) => { panic!("no node idx on f64.") }
        }
    }
    pub fn forward_mode_grad(&self, add_to_computation_graph: bool) -> ForwardModeGradOutput {
        let v = self.value();
        assert!(!v.is_nan() && v.is_finite());
        match self {
            f64ad::f64ad_var(f) => { f.forward_mode_grad(add_to_computation_graph) }
            _ => { panic!("cannot compute grad on f64.") }
        }
    }
    pub fn backwards_mode_grad(&self, add_to_computation_graph: bool) -> BackwardsModeGradOutput {
        let v = self.value();
        assert!(!v.is_nan() && v.is_finite());
        match self {
            f64ad::f64ad_var(f) => { f.backwards_mode_grad(add_to_computation_graph) }
            _ => { panic!("cannot compute grad on f64.") }
        }
    }
    fn transfer_to_other_global_computation_graph(&mut self, name: Option<&str>, idx: Option<usize>) {
        match self {
            f64ad::f64ad_var(v) => {
                v.transfer_to_other_global_computation_graph(name, idx);
            }
            _ => { panic!("cannot transfer an f64ad that is not an f64ad_var.") }
        }
    }
    fn clone_computation_graph_to_other_global_computation_graph(&self, name: Option<&str>, idx: Option<usize>) {
        match self {
            f64ad::f64ad_var(v) => {
                v.clone_computation_graph_to_other_global_computation_graph(name, idx);
            }
            _ => { panic!("cannot clone computation graph of an f64ad that is not an f64ad_var.") }
        }
    }
}
impl Default for f64ad {
    fn default() -> Self {
        Self::f64(0.0)
    }
}
unsafe impl Sync for f64ad {}
unsafe impl Send for f64ad {}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub struct f64ad_var {
    computation_graph_id: usize,
    node_idx: u32,
    mode: ComputationGraphMode,
    computation_graph: &'static ComputationGraph,
}
impl f64ad_var {
    pub(crate) fn value(&self) -> f64 {
        self.check_if_out_of_sync();
        self.computation_graph.0.borrow().nodes[self.node_idx as usize].value
    }
    pub(crate) fn new(computation_graph_id: usize, node_idx: u32, mode: ComputationGraphMode, computation_graph: &'static ComputationGraph) -> Self {
        Self {
            computation_graph_id,
            node_idx,
            mode,
            computation_graph,
        }
    }
    fn transfer_to_other_global_computation_graph(&mut self, name: Option<&str>, idx: Option<usize>) {
        let global_computation_graph = GlobalComputationGraphs::get(name, idx);
        unsafe { self.computation_graph = &*global_computation_graph.0 };
    }
    fn clone_computation_graph_to_other_global_computation_graph(&self, name: Option<&str>, idx: Option<usize>) {
        self.computation_graph.clone_to_global_computation_graph(name, idx);
    }
    fn forward_mode_grad(&self, add_to_computation_graph: bool) -> ForwardModeGradOutput {
        self.check_if_out_of_sync();
        let l = self.computation_graph.0.borrow().nodes.len();
        let mut derivs = vec![f64ad::f64(0.0); l];
        if add_to_computation_graph {
            let v = self.computation_graph.spawn_f64ad_var(1.0);
            derivs[self.node_idx as usize] = v;
        } else {
            derivs[self.node_idx as usize] = f64ad::f64(1.0);
        }

        let computation_graph = self.computation_graph;
        for node_idx in 0..l {
            let node = computation_graph.0.borrow().nodes[node_idx].clone();
            let derivatives_of_value_wrt_parent_values = node.node_type.compute_derivatives_of_value_wrt_parent_values(&node.parent_nodes, &node.constant_operands, computation_graph, add_to_computation_graph);
            for (i, d) in derivatives_of_value_wrt_parent_values.iter().enumerate() {
                let parent_idx = node.parent_nodes[i];
                let parent_deriv = derivs[parent_idx as usize];
                derivs[node_idx] += parent_deriv * *d;
            }
        }

        return ForwardModeGradOutput { derivs };
    }
    fn backwards_mode_grad(&self, add_to_computation_graph: bool) -> BackwardsModeGradOutput {
        self.check_if_out_of_sync();
        let l = self.computation_graph.0.borrow().nodes.len();
        let mut derivs = vec![f64ad::f64(0.0); l];
        if add_to_computation_graph {
            let v = self.computation_graph.spawn_f64ad_var(1.0);
            derivs[self.node_idx as usize] = v;
        } else {
            derivs[self.node_idx as usize] = f64ad::f64(1.0);
        }

        let computation_graph = self.computation_graph;
        for node_idx in (0..l).rev() {
            let curr_deriv = derivs[node_idx];
            let node = computation_graph.0.borrow().nodes[node_idx].clone();
            let derivatives_of_value_wrt_parent_values = node.node_type.compute_derivatives_of_value_wrt_parent_values(&node.parent_nodes, &node.constant_operands, computation_graph, add_to_computation_graph);
            for (i, d) in derivatives_of_value_wrt_parent_values.iter().enumerate() {
                let parent_idx = node.parent_nodes[i];
                derivs[parent_idx as usize] += curr_deriv * *d;
            }
        }

        return BackwardsModeGradOutput { derivs }
    }
    pub (crate) fn check_if_out_of_sync(&self) {
        assert_eq!(self.computation_graph_id, self.computation_graph.0.borrow().id, "this variable is not in sync with its computation graph, the computation graph must have been reset.");
    }
}
impl Debug for f64ad_var {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        f.write_str("f64ad_var{ ").expect("error");
        f.write_str(&format!("value: {:?}, ", self.value())).expect("error");
        f.write_str(&format!("node_idx: {:?}", self.node_idx)).expect("error");
        f.write_str(" }").expect("error");

        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ForwardModeGradOutput {
    derivs: Vec<f64ad>
}
impl ForwardModeGradOutput {
    pub fn wrt(&self, output: &f64ad) -> f64ad {
        return self.derivs[output.node_idx() as usize];
    }
}

#[derive(Clone, Debug)]
pub struct BackwardsModeGradOutput {
    derivs: Vec<f64ad>
}
impl BackwardsModeGradOutput {
    pub fn wrt(&self, input: &f64ad) -> f64ad {
        return self.derivs[input.node_idx() as usize];
    }
}

#[derive(Clone, Debug)]
pub struct ComputationGraph_ {
    id: usize,
    nodes: Vec<F64ADNode>,
    mode: ComputationGraphMode,
}
impl ComputationGraph_ {
    fn new(mode: ComputationGraphMode) -> Self {
        let mut rng = rand::thread_rng();
        let id: usize = rng.gen();
        Self {
            id,
            nodes: Vec::with_capacity(1_000_000),
            mode,
        }
    }
    pub fn reset(&mut self, mode: ComputationGraphMode) {
        *self = Self::new(mode);
    }
}

#[derive(Debug)]
pub struct ComputationGraph(RefCell<ComputationGraph_>);
impl ComputationGraph {
    pub (crate) fn add_node(&'static self, node_type: NodeType, parent_nodes: TinyVec<[u32; 2]>, constant_operands: TinyVec<[f64; 1]>) -> f64ad_var {
        let value = node_type.compute_value(&parent_nodes, &constant_operands, self);
        let node_idx = self.0.borrow().nodes.len() as u32;
        for parent_node in &parent_nodes { self.0.borrow_mut().nodes[*parent_node as usize].child_nodes.push(node_idx); }
        let node = F64ADNode::new(node_idx, value, node_type, constant_operands, parent_nodes);
        self.0.borrow_mut().nodes.push(node);

        f64ad_var {
            computation_graph_id: self.0.borrow().id,
            node_idx,
            mode: self.0.borrow().mode.clone(),
            computation_graph: self,
        }
    }
    fn new(c: ComputationGraph_) -> Self {
        Self(RefCell::new(c))
    }
    fn reset(&self, mode: ComputationGraphMode) {
        self.0.borrow_mut().reset(mode);
    }
    fn spawn_f64ad_var(&'static self, value: f64) -> f64ad {
        let node_idx = self.0.borrow().nodes.len();

        let f = f64ad_var {
            computation_graph_id: self.0.borrow().id,
            node_idx: node_idx as u32,
            mode: self.0.borrow().mode.clone(),
            computation_graph: self,
        };

        let n = F64ADNode {
            node_idx: node_idx as u32,
            value,
            node_type: NodeType::InputVar,
            constant_operands: tiny_vec!([f64; 1]),
            parent_nodes: tiny_vec!([u32; 2]),
            child_nodes: tiny_vec!([u32; 5]),
        };

        self.0.borrow_mut().nodes.push(n);

        return f64ad::f64ad_var(f);
    }
    fn clone_to_global_computation_graph(&self, name: Option<&str>, idx: Option<usize>) {
        let computation_graph = self.clone();
        GlobalComputationGraphs::set(name, idx, computation_graph);
    }
    fn lock(&self) -> LockedComputationGraph {
        return LockedComputationGraph::new(self.clone());
    }
}
impl Clone for ComputationGraph {
    fn clone(&self) -> Self {
        let computation_graph = self.0.borrow().clone();
        Self(RefCell::new(computation_graph))
    }
}

#[derive(Clone)]
pub struct GlobalComputationGraph(*const ComputationGraph);
impl GlobalComputationGraph {
    fn reset(&self, mode: ComputationGraphMode) {
        unsafe { (*self.0).reset(mode) }
    }
    pub fn spawn_f64ad_var(&self, value: f64) -> f64ad {
        return unsafe { (*self.0).spawn_f64ad_var(value) }
    }
    fn lock(&self) -> LockedComputationGraph {
        return unsafe { (*self.0).lock() }
    }
}
impl Debug for GlobalComputationGraph {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        unsafe { (*self.0).fmt(f) }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ComputationGraphMode {
    Standard,
    Lock,
}

#[derive(Clone, Debug)]
pub struct F64ADNode {
    node_idx: u32,
    value: f64,
    node_type: NodeType,
    constant_operands: TinyVec<[f64; 1]>,
    parent_nodes: TinyVec<[u32; 2]>,
    child_nodes: TinyVec<[u32; 5]>,
}
impl F64ADNode {
    pub fn new(node_idx: u32, value: f64, node_type: NodeType, constant_operands: TinyVec<[f64; 1]>, parent_nodes: TinyVec<[u32; 2]>) -> Self {
        Self {
            node_idx,
            value,
            node_type,
            constant_operands,
            parent_nodes,
            child_nodes: tiny_vec!([u32; 5]),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NodeType {
    None,
    InputVar,
    AdditionOneParent,
    AdditionTwoParents,
    MultiplicationOneParent,
    MultiplicationTwoParents,
    SubtractionOneParentLeft,
    SubtractionOneParentRight,
    SubtractionTwoParents,
    DivisionOneParentDenominator,
    DivisionOneParentNumerator,
    DivisionTwoParents,
    Neg,
    Abs,
    Signum,
    MaxOneParent,
    MaxTwoParents,
    MinOneParent,
    MinTwoParents,
    Atan2OneParentLeft,
    Atan2OneParentRight,
    Atan2TwoParents,
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
    LogOneParentArgument,
    LogOneParentBase,
    LogTwoParents,
    Sqrt,
    Exp,
    PowOneParentArgument,
    PowOneParentExponent,
    PowTwoParents,
}
impl NodeType {
    pub fn compute_value(&self, parent_nodes: &TinyVec<[u32; 2]>, constant_operands: &TinyVec<[f64; 1]>, computation_graph: &ComputationGraph) -> f64 {
        match self {
            NodeType::None => { panic!("Cannot compute value on node type None.") }
            NodeType::InputVar => { panic!("Cannot compute value on node type InputVar.") }
            NodeType::AdditionOneParent => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return parent_value + constant_operands[0];
            }
            NodeType::AdditionTwoParents => {
                let parent_value_0 = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                let parent_value_1 = computation_graph.0.borrow().nodes[parent_nodes[1] as usize].value;
                return parent_value_0 + parent_value_1;
            }
            NodeType::MultiplicationOneParent => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return parent_value * constant_operands[0];
            }
            NodeType::MultiplicationTwoParents => {
                let parent_value_0 = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                let parent_value_1 = computation_graph.0.borrow().nodes[parent_nodes[1] as usize].value;
                return parent_value_0 * parent_value_1;
            }
            NodeType::SubtractionOneParentLeft => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return parent_value - constant_operands[0];
            }
            NodeType::SubtractionOneParentRight => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return constant_operands[0] - parent_value;
            }
            NodeType::SubtractionTwoParents => {
                let parent_value_0 = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                let parent_value_1 = computation_graph.0.borrow().nodes[parent_nodes[1] as usize].value;
                return parent_value_0 - parent_value_1;
            }
            NodeType::DivisionOneParentDenominator => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return constant_operands[0] / parent_value;
            }
            NodeType::DivisionOneParentNumerator => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return parent_value / constant_operands[0];
            }
            NodeType::DivisionTwoParents => {
                let parent_value_0 = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                let parent_value_1 = computation_graph.0.borrow().nodes[parent_nodes[1] as usize].value;
                return parent_value_0 / parent_value_1;
            }
            NodeType::Neg => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return -parent_value;
            }
            NodeType::Abs => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return parent_value.abs();
            }
            NodeType::Signum => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return parent_value.signum();
            }
            NodeType::MaxOneParent => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return parent_value.max(constant_operands[0]);
            }
            NodeType::MaxTwoParents => {
                let parent_value_0 = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                let parent_value_1 = computation_graph.0.borrow().nodes[parent_nodes[1] as usize].value;
                return parent_value_0.max(parent_value_1);
            }
            NodeType::MinOneParent => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return parent_value.min(constant_operands[0]);
            }
            NodeType::MinTwoParents => {
                let parent_value_0 = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                let parent_value_1 = computation_graph.0.borrow().nodes[parent_nodes[1] as usize].value;
                return parent_value_0.min(parent_value_1);
            }
            NodeType::Atan2OneParentLeft => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return parent_value.atan2(constant_operands[0]);
            }
            NodeType::Atan2OneParentRight => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return constant_operands[0].atan2(parent_value);
            }
            NodeType::Atan2TwoParents => {
                let parent_value_0 = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                let parent_value_1 = computation_graph.0.borrow().nodes[parent_nodes[1] as usize].value;
                return parent_value_0.atan2(parent_value_1);
            }
            NodeType::Floor => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return parent_value.floor();
            }
            NodeType::Ceil => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return parent_value.ceil();
            }
            NodeType::Round => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return parent_value.round();
            }
            NodeType::Trunc => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return parent_value.trunc();
            }
            NodeType::Fract => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return parent_value.fract();
            }
            NodeType::Sin => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return parent_value.sin();
            }
            NodeType::Cos => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return parent_value.cos();
            }
            NodeType::Tan => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return parent_value.tan();
            }
            NodeType::Asin => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return parent_value.asin();
            }
            NodeType::Acos => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return parent_value.acos();
            }
            NodeType::Atan => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return parent_value.atan();
            }
            NodeType::Sinh => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return parent_value.sinh();
            }
            NodeType::Cosh => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return parent_value.cosh();
            }
            NodeType::Tanh => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return parent_value.tanh();
            }
            NodeType::Asinh => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return parent_value.asinh();
            }
            NodeType::Acosh => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return parent_value.acosh();
            }
            NodeType::Atanh => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return parent_value.atanh();
            }
            NodeType::LogOneParentArgument => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return parent_value.log(constant_operands[0]);
            }
            NodeType::LogOneParentBase => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return constant_operands[0].log(parent_value);
            }
            NodeType::LogTwoParents => {
                let parent_value_0 = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                let parent_value_1 = computation_graph.0.borrow().nodes[parent_nodes[1] as usize].value;
                return parent_value_0.log(parent_value_1);
            }
            NodeType::Sqrt => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return parent_value.sqrt();
            }
            NodeType::Exp => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return parent_value.exp();
            }
            NodeType::PowOneParentArgument => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return parent_value.powf(constant_operands[0]);
            }
            NodeType::PowOneParentExponent => {
                let parent_value = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                return constant_operands[0].powf(parent_value);
            }
            NodeType::PowTwoParents => {
                let parent_value_0 = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                let parent_value_1 = computation_graph.0.borrow().nodes[parent_nodes[1] as usize].value;
                return parent_value_0.powf(parent_value_1);
            }
        }
    }
    pub fn compute_derivatives_of_value_wrt_parent_values(&self, parent_nodes: &TinyVec<[u32; 2]>, constant_operands: &TinyVec<[f64; 1]>, computation_graph: &'static ComputationGraph, add_to_computation_graph: bool) -> TinyVec<[f64ad; 2]> {
        match self {
            NodeType::None => { panic!("Cannot compute derivatives on node type None.") }
            NodeType::InputVar => { return tiny_vec!([f64ad; 2]); }
            NodeType::AdditionOneParent => { return tiny_vec!([f64ad; 2] => f64ad::f64(1.0)); }
            NodeType::AdditionTwoParents => { return tiny_vec!([f64ad; 2] => f64ad::f64(1.0), f64ad::f64(1.0)); }
            NodeType::MultiplicationOneParent => { return tiny_vec!([f64ad; 2] => f64ad::f64(constant_operands[0])); }
            NodeType::MultiplicationTwoParents => {
                return if add_to_computation_graph {
                    let parent_node_0 = &computation_graph.0.borrow().nodes[parent_nodes[0] as usize];
                    let parent_node_1 = &computation_graph.0.borrow().nodes[parent_nodes[1] as usize];
                    let id = computation_graph.0.borrow().id;
                    let mode = computation_graph.0.borrow().mode.clone();
                    let node_idx_0 = parent_node_0.node_idx;
                    let node_idx_1 = parent_node_1.node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_1, mode, computation_graph));
                    let f1 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph));
                    tiny_vec!([f64ad; 2] => f0, f1)
                } else {
                    let parent_node_0 = &computation_graph.0.borrow().nodes[parent_nodes[0] as usize];
                    let parent_node_1 = &computation_graph.0.borrow().nodes[parent_nodes[1] as usize];
                    tiny_vec!([f64ad; 2] => f64ad::f64(parent_node_1.value), f64ad::f64(parent_node_0.value))
                };
            }
            NodeType::SubtractionOneParentLeft => { return tiny_vec!([f64ad; 2] => f64ad::f64(1.0)); }
            NodeType::SubtractionOneParentRight => { return tiny_vec!([f64ad; 2] => f64ad::f64(-1.0)); }
            NodeType::SubtractionTwoParents => { return tiny_vec!([f64ad; 2] => f64ad::f64(1.0), f64ad::f64(-1.0)); }
            NodeType::DivisionOneParentDenominator => {
                return if add_to_computation_graph {
                    let id = computation_graph.0.borrow().id;
                    let mode = computation_graph.0.borrow().mode.clone();
                    let node_idx_0 = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph));
                    let ret = -(constant_operands[0] / (f0 * f0));
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.0.borrow().nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(-(constant_operands[0] / (v * v))))
                };
            }
            NodeType::DivisionOneParentNumerator => {
                return tiny_vec!([f64ad; 2] => f64ad::f64(1.0/constant_operands[0]));
            }
            NodeType::DivisionTwoParents => {
                return if add_to_computation_graph {
                    let id = computation_graph.0.borrow().id;
                    let mode = computation_graph.0.borrow().mode.clone();
                    let node_idx_0 = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].node_idx;
                    let node_idx_1 = computation_graph.0.borrow().nodes[parent_nodes[1] as usize].node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph));
                    let f1 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_1, mode, computation_graph));
                    tiny_vec!([f64ad; 2] => 1.0/f1, -f0/(f1*f1))
                } else {
                    let parent_node_0 = &computation_graph.0.borrow().nodes[parent_nodes[0] as usize];
                    let parent_node_1 = &computation_graph.0.borrow().nodes[parent_nodes[1] as usize];
                    tiny_vec!([f64ad; 2] => f64ad::f64(1.0/parent_node_1.value), f64ad::f64(-parent_node_0.value/(parent_node_1.value * parent_node_1.value)))
                };
            }
            NodeType::Neg => { return tiny_vec!([f64ad; 2] => f64ad::f64(-1.0)); }
            NodeType::Abs => {
                let parent_node_0 = &computation_graph.0.borrow().nodes[parent_nodes[0] as usize];
                let v = parent_node_0.value;
                if v >= 0.0 {
                    tiny_vec!([f64ad; 2] => f64ad::f64(1.0))
                } else {
                    tiny_vec!([f64ad; 2] => f64ad::f64(-1.0))
                }
            }
            NodeType::Signum => {
                return tiny_vec!([f64ad; 2] => f64ad::f64(0.0));
            }
            NodeType::MaxOneParent => {
                let parent_node_0 = &computation_graph.0.borrow().nodes[parent_nodes[0] as usize];
                let v = parent_node_0.value;
                return if v >= constant_operands[0] { tiny_vec!([f64ad; 2] => f64ad::f64(1.0)) } else { tiny_vec!([f64ad; 2] => f64ad::f64(0.0)) };
            }
            NodeType::MaxTwoParents => {
                let parent_node_0 = &computation_graph.0.borrow().nodes[parent_nodes[0] as usize];
                let parent_node_1 = &computation_graph.0.borrow().nodes[parent_nodes[1] as usize];
                let v0 = parent_node_0.value;
                let v1 = parent_node_1.value;
                return if v0 >= v1 { tiny_vec!([f64ad; 2] => f64ad::f64(1.0), f64ad::f64(0.0)) } else { tiny_vec!([f64ad; 2] => f64ad::f64(0.0), f64ad::f64(1.0)) };
            }
            NodeType::MinOneParent => {
                let parent_node_0 = &computation_graph.0.borrow().nodes[parent_nodes[0] as usize];
                let v = parent_node_0.value;
                return if v <= constant_operands[0] { tiny_vec!([f64ad; 2] => f64ad::f64(1.0)) } else { tiny_vec!([f64ad; 2] => f64ad::f64(0.0)) };
            }
            NodeType::MinTwoParents => {
                let parent_node_0 = &computation_graph.0.borrow().nodes[parent_nodes[0] as usize];
                let parent_node_1 = &computation_graph.0.borrow().nodes[parent_nodes[1] as usize];
                let v0 = parent_node_0.value;
                let v1 = parent_node_1.value;
                return if v0 <= v1 { tiny_vec!([f64ad; 2] => f64ad::f64(1.0), f64ad::f64(0.0)) } else { tiny_vec!([f64ad; 2] => f64ad::f64(0.0), f64ad::f64(1.0)) };
            }
            NodeType::Atan2OneParentLeft => {
                return if add_to_computation_graph {
                    let id = computation_graph.0.borrow().id;
                    let mode = computation_graph.0.borrow().mode.clone();
                    let node_idx_0 = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph));
                    let ret = constant_operands[0] / (constant_operands[0].powi(2) + (f0 * f0));
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.0.borrow().nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(constant_operands[0]/ (constant_operands[0].powi(2) + v.powi(2))))
                };
            }
            NodeType::Atan2OneParentRight => {
                return if add_to_computation_graph {
                    let id = computation_graph.0.borrow().id;
                    let mode = computation_graph.0.borrow().mode.clone();
                    let node_idx_0 = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph));
                    let ret = -constant_operands[0] / (constant_operands[0].powi(2) + (f0 * f0));
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.0.borrow().nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(-constant_operands[0]/ (constant_operands[0].powi(2) + v.powi(2))))
                };
            }
            NodeType::Atan2TwoParents => {
                return if add_to_computation_graph {
                    let id = computation_graph.0.borrow().id;
                    let mode = computation_graph.0.borrow().mode.clone();
                    let node_idx_0 = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].node_idx;
                    let node_idx_1 = computation_graph.0.borrow().nodes[parent_nodes[1] as usize].node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph));
                    let f1 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_1, mode, computation_graph));
                    tiny_vec!([f64ad; 2] => f1/(f0*f0 + f1*f1), -f0/(f0*f0 + f1*f1))
                } else {
                    let parent_node_0 = &computation_graph.0.borrow().nodes[parent_nodes[0] as usize];
                    let parent_node_1 = &computation_graph.0.borrow().nodes[parent_nodes[1] as usize];
                    let v0 = parent_node_0.value;
                    let v1 = parent_node_1.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(-v0/(v0*v0 + v1*v1)))
                };
            }
            NodeType::Floor => { return tiny_vec!([f64ad; 2] => f64ad::f64(0.0)); }
            NodeType::Ceil => { return tiny_vec!([f64ad; 2] => f64ad::f64(0.0)); }
            NodeType::Round => { return tiny_vec!([f64ad; 2] => f64ad::f64(0.0)); }
            NodeType::Trunc => { return tiny_vec!([f64ad; 2] => f64ad::f64(0.0)); }
            NodeType::Fract => { return tiny_vec!([f64ad; 2] => f64ad::f64(0.0)); }
            NodeType::Sin => {
                return if add_to_computation_graph {
                    let id = computation_graph.0.borrow().id;
                    let mode = computation_graph.0.borrow().mode.clone();
                    let node_idx_0 = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph));
                    let ret = f0.cos();
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let v = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(v.cos()))
                };
            }
            NodeType::Cos => {
                return if add_to_computation_graph {
                    let id = computation_graph.0.borrow().id;
                    let mode = computation_graph.0.borrow().mode.clone();
                    let node_idx_0 = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph));
                    let ret = -f0.sin();
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.0.borrow().nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(-v.sin()))
                };
            }
            NodeType::Tan => {
                return if add_to_computation_graph {
                    let id = computation_graph.0.borrow().id;
                    let mode = computation_graph.0.borrow().mode.clone();
                    let node_idx_0 = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph));
                    let c = f0.cos();
                    let ret = 1.0 / (c * c);
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.0.borrow().nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    let c = v.cos();
                    tiny_vec!([f64ad; 2] => f64ad::f64(1.0/(c*c)))
                };
            }
            NodeType::Asin => {
                return if add_to_computation_graph {
                    let id = computation_graph.0.borrow().id;
                    let mode = computation_graph.0.borrow().mode.clone();
                    let node_idx_0 = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph));
                    let ret = 1.0 / (1.0 - f0 * f0).sqrt();
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.0.borrow().nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(1.0/(1.0 - v*v).sqrt()))
                };
            }
            NodeType::Acos => {
                return if add_to_computation_graph {
                    let id = computation_graph.0.borrow().id;
                    let mode = computation_graph.0.borrow().mode.clone();
                    let node_idx_0 = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph));
                    let ret = -1.0 / (1.0 - f0 * f0).sqrt();
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.0.borrow().nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(-1.0/(1.0 - v*v).sqrt()))
                };
            }
            NodeType::Atan => {
                return if add_to_computation_graph {
                    let id = computation_graph.0.borrow().id;
                    let mode = computation_graph.0.borrow().mode.clone();
                    let node_idx_0 = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph));
                    let ret = 1.0 / (f0 * f0 + 1.0);
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.0.borrow().nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(1.0/(v*v + 1.0)))
                };
            }
            NodeType::Sinh => {
                return if add_to_computation_graph {
                    let id = computation_graph.0.borrow().id;
                    let mode = computation_graph.0.borrow().mode.clone();
                    let node_idx_0 = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph));
                    let ret = f0.cosh();
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.0.borrow().nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(v.cosh()))
                };
            }
            NodeType::Cosh => {
                return if add_to_computation_graph {
                    let id = computation_graph.0.borrow().id;
                    let mode = computation_graph.0.borrow().mode.clone();
                    let node_idx_0 = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph));
                    let ret = f0.sinh();
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.0.borrow().nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(v.sinh()))
                };
            }
            NodeType::Tanh => {
                return if add_to_computation_graph {
                    let id = computation_graph.0.borrow().id;
                    let mode = computation_graph.0.borrow().mode.clone();
                    let node_idx_0 = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph));
                    let c = f0.cosh();
                    let ret = 1.0 / (c * c);
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.0.borrow().nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(1.0/v.cosh().powi(2)))
                };
            }
            NodeType::Asinh => {
                return if add_to_computation_graph {
                    let id = computation_graph.0.borrow().id;
                    let mode = computation_graph.0.borrow().mode.clone();
                    let node_idx_0 = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph));
                    let ret = 1.0 / (f0 * f0 + 1.0).sqrt();
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.0.borrow().nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(1.0 / (v*v + 1.0).sqrt()))
                };
            }
            NodeType::Acosh => {
                return if add_to_computation_graph {
                    let id = computation_graph.0.borrow().id;
                    let mode = computation_graph.0.borrow().mode.clone();
                    let node_idx_0 = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph));
                    let ret = 1.0 / ((f0 - 1.0).sqrt() * (f0 + 1.0).sqrt());
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.0.borrow().nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(1.0 / ((v - 1.0).sqrt()*(v + 1.0).sqrt())))
                };
            }
            NodeType::Atanh => {
                return if add_to_computation_graph {
                    let id = computation_graph.0.borrow().id;
                    let mode = computation_graph.0.borrow().mode.clone();
                    let node_idx_0 = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph));
                    let ret = 1.0 / (1.0 - f0 * f0);
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.0.borrow().nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(1.0 / (1.0 - v*v)))
                };
            }
            NodeType::LogOneParentArgument => {
                return if add_to_computation_graph {
                    let id = computation_graph.0.borrow().id;
                    let mode = computation_graph.0.borrow().mode.clone();
                    let node_idx_0 = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph));
                    let ret = 1.0 / (f0 * constant_operands[0].ln());
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.0.borrow().nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(1.0 / (v*constant_operands[0].ln())))
                };
            }
            NodeType::LogOneParentBase => {
                return if add_to_computation_graph {
                    let id = computation_graph.0.borrow().id;
                    let mode = computation_graph.0.borrow().mode.clone();
                    let node_idx_0 = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph));
                    let ly = constant_operands[0].ln();
                    let lx = f0.ln();
                    println!(" >> {:?}", lx);
                    let ret = -ly / (f0 * (lx * lx));
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.0.borrow().nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    let ly = constant_operands[0].ln();
                    let lx = v.ln();
                    let ret = -ly / (v * (lx * lx));
                    tiny_vec!([f64ad; 2] => f64ad::f64(ret))
                };
            }
            NodeType::LogTwoParents => {
                return if add_to_computation_graph {
                    let id = computation_graph.0.borrow().id;
                    let mode = computation_graph.0.borrow().mode.clone();
                    let node_idx_0 = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].node_idx;
                    let node_idx_1 = computation_graph.0.borrow().nodes[parent_nodes[1] as usize].node_idx;
                    let argument = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph));
                    let base = f64ad::f64ad_var(f64ad_var::new(id, node_idx_1, mode, computation_graph));
                    let ret0 = 1.0 / (argument * base.ln());
                    let lb = base.ln();
                    let ret1 = -argument.ln() / (base * (lb * lb));
                    tiny_vec!([f64ad; 2] => ret0, ret1)
                } else {
                    let parent_node_0 = &computation_graph.0.borrow().nodes[parent_nodes[0] as usize];
                    let parent_node_1 = &computation_graph.0.borrow().nodes[parent_nodes[1] as usize];
                    let argument = parent_node_0.value;
                    let base = parent_node_1.value;
                    let ret0 = 1.0 / (argument * base.ln());
                    let lb = base.ln();
                    let ret1 = -argument.ln() / (base * (lb * lb));
                    tiny_vec!([f64ad; 2] => f64ad::f64(ret0), f64ad::f64(ret1))
                };
            }
            NodeType::Sqrt => {
                return if add_to_computation_graph {
                    let id = computation_graph.0.borrow().id;
                    let mode = computation_graph.0.borrow().mode.clone();
                    let node_idx_0 = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph));
                    let ret = 1.0 / (2.0 * f0.sqrt());
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.0.borrow().nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(1.0/(2.0*v.sqrt())))
                };
            }
            NodeType::Exp => {
                return if add_to_computation_graph {
                    let id = computation_graph.0.borrow().id;
                    let mode = computation_graph.0.borrow().mode.clone();
                    let node_idx_0 = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph));
                    let ret = f0.exp();
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.0.borrow().nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(v.exp()))
                };
            }
            NodeType::PowOneParentArgument => {
                return if add_to_computation_graph {
                    let id = computation_graph.0.borrow().id;
                    let mode = computation_graph.0.borrow().mode.clone();
                    let node_idx_0 = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph));
                    let ret = constant_operands[0] * f0.powf(f64ad::f64(constant_operands[0] - 1.0));
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.0.borrow().nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(constant_operands[0] * v.powf(constant_operands[0] - 1.0)))
                };
            }
            NodeType::PowOneParentExponent => {
                return if add_to_computation_graph {
                    let id = computation_graph.0.borrow().id;
                    let mode = computation_graph.0.borrow().mode.clone();
                    let node_idx_0 = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph));
                    let c = constant_operands[0];
                    let ret = f64ad::f64(c).powf(f0) * c.ln();
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.0.borrow().nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    let c = constant_operands[0];
                    tiny_vec!([f64ad; 2] => f64ad::f64(c.powf(v) * c.ln()))
                };
            }
            NodeType::PowTwoParents => {
                return if add_to_computation_graph {
                    let id = computation_graph.0.borrow().id;
                    let mode = computation_graph.0.borrow().mode.clone();
                    let node_idx_0 = computation_graph.0.borrow().nodes[parent_nodes[0] as usize].node_idx;
                    let node_idx_1 = computation_graph.0.borrow().nodes[parent_nodes[1] as usize].node_idx;
                    let argument = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph));
                    let exponent = f64ad::f64ad_var(f64ad_var::new(id, node_idx_1, mode, computation_graph));
                    let ret0 = exponent * argument.powf(exponent - 1.0);
                    let ret1 = argument.powf(exponent) * argument.ln();
                    tiny_vec!([f64ad; 2] => ret0, ret1)
                } else {
                    let parent_node_0 = &computation_graph.0.borrow().nodes[parent_nodes[0] as usize];
                    let parent_node_1 = &computation_graph.0.borrow().nodes[parent_nodes[1] as usize];
                    let argument = parent_node_0.value;
                    let exponent = parent_node_1.value;
                    let ret0 = exponent * argument.powf(exponent - 1.0);
                    let ret1 = argument.powf(exponent) * argument.ln();
                    tiny_vec!([f64ad; 2] => f64ad::f64(ret0), f64ad::f64(ret1))
                };
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct LockedComputationGraph_ {
    id: usize,
    computation_graph: ComputationGraph,
    curr_count: u32,
    push_forward_compute_start_idx: usize,
}

#[derive(Debug)]
pub struct LockedComputationGraph(RefCell<LockedComputationGraph_>);
impl LockedComputationGraph {
    fn new(computation_graph: ComputationGraph) -> Self {
        assert_eq!(computation_graph.0.borrow().mode, ComputationGraphMode::Lock, "Computation graph mode must be Lock in order to create a LockedComputationGraph.");

        let mut rng = rand::thread_rng();
        let id: usize = rng.gen();

        let locked_computation_graph_ = LockedComputationGraph_ {
            id,
            computation_graph,
            curr_count: 0,
            push_forward_compute_start_idx: 0,
        };

        Self(RefCell::new(locked_computation_graph_))
    }
    pub fn get_value(&self, idx: usize) -> f64 {
        return self.0.borrow().computation_graph.0.borrow().nodes[idx].value;
    }
    pub fn spawn_locked_var(&'static self, value: f64) -> f64ad {
        let node_idx = self.0.borrow().curr_count;
        self.0.borrow_mut().computation_graph.0.borrow_mut().nodes[node_idx as usize].value = value;
        let f = f64ad_locked_var {
            locked_computation_graph_id: self.0.borrow().id,
            node_idx: node_idx as u32,
            locked_computation_graph: &self,
        };
        let curr_count = self.0.borrow().curr_count as usize;
        self.0.borrow_mut().push_forward_compute_start_idx = curr_count;
        self.0.borrow_mut().curr_count += 1;
        return f64ad::f64ad_locked_var(f);
    }
}
impl Clone for LockedComputationGraph {
    fn clone(&self) -> Self {
        let locked_computation_graph = self.0.borrow().clone();
        Self(RefCell::new(locked_computation_graph))
    }
}

#[derive(Clone)]
pub struct GlobalLockedComputationGraph(*const LockedComputationGraph);
impl GlobalLockedComputationGraph {
    pub fn get_value(&self, idx: usize) -> f64 {
        return unsafe { (*self.0).get_value(idx) }
    }
    pub fn spawn_locked_var(&self, value: f64) -> f64ad {
        return unsafe { (*self.0).spawn_locked_var(value) }
    }
}
impl Debug for GlobalLockedComputationGraph {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        unsafe { (*self.0).fmt(f) }
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub struct f64ad_locked_var {
    locked_computation_graph_id: usize,
    node_idx: u32,
    locked_computation_graph: &'static LockedComputationGraph,
}
impl f64ad_locked_var {
    pub(crate) fn value(&self) -> f64 {
        self.locked_computation_graph.0.borrow().computation_graph.0.borrow().nodes[self.node_idx as usize].value
    }
}
impl Debug for f64ad_locked_var {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        f.write_str("f64ad_locked_var{ ").expect("error");
        f.write_str(&format!("value: {:?}, ", self.value())).expect("error");
        f.write_str(&format!("node_idx: {:?}", self.node_idx)).expect("error");
        f.write_str(" }").expect("error");

        Ok(())
    }
}

pub(crate) fn f64ad_locked_var_operation_two_parents(lhs: &f64ad_locked_var, rhs: &f64ad_locked_var, node_type: NodeType) -> f64ad_locked_var {
    assert_eq!(lhs.locked_computation_graph_id, rhs.locked_computation_graph_id);
    let locked_computation_graph = &mut (*lhs.locked_computation_graph.0.borrow_mut());
    let node_idx = locked_computation_graph.curr_count as usize;
    let computation_graph = &mut locked_computation_graph.computation_graph;
    let node = computation_graph.0.borrow().nodes[node_idx].clone();
    assert_eq!(node.node_type, node_type);
    let value = computation_graph.0.borrow().nodes[node_idx].node_type.compute_value(&tiny_vec!([u32; 2] => lhs.node_idx, rhs.node_idx), &tiny_vec!([f64; 1]), computation_graph);
    locked_computation_graph.computation_graph.0.borrow_mut().nodes[node_idx].value = value;
    locked_computation_graph.curr_count += 1;
    locked_computation_graph.push_forward_compute_start_idx += 1;
    return f64ad_locked_var {
        locked_computation_graph_id: lhs.locked_computation_graph_id,
        node_idx: node_idx as u32,
        locked_computation_graph: lhs.locked_computation_graph,
    };
}

pub(crate) fn f64ad_locked_var_operation_one_parent(v: &f64ad_locked_var, constant_operand: Option<f64>, node_type: NodeType) -> f64ad_locked_var {
    let locked_computation_graph = &mut (*v.locked_computation_graph.0.borrow_mut());
    let node_idx = locked_computation_graph.curr_count as usize;
    let computation_graph = &mut locked_computation_graph.computation_graph;
    let node = computation_graph.0.borrow().nodes[node_idx].clone();
    assert_eq!(node.node_type, node_type);
    let value = match constant_operand {
        None => { node.node_type.compute_value(&tiny_vec!([u32; 2] => v.node_idx), &tiny_vec!([f64; 1]), computation_graph) }
        Some(constant_operand) => { node.node_type.compute_value(&tiny_vec!([u32; 2] => v.node_idx), &tiny_vec!([f64; 1] => constant_operand), computation_graph) }
    };
    locked_computation_graph.computation_graph.0.borrow_mut().nodes[node_idx].value = value;
    locked_computation_graph.curr_count += 1;
    locked_computation_graph.push_forward_compute_start_idx += 1;
    return f64ad_locked_var {
        locked_computation_graph_id: v.locked_computation_graph_id,
        node_idx: node_idx as u32,
        locked_computation_graph: v.locked_computation_graph,
    };
}

static mut _GLOBAL_COMPUTATION_GRAPHS: OnceCell<Mutex<HashMap<(String, usize), ComputationGraph>>> = OnceCell::new();
pub struct GlobalComputationGraphs;
impl GlobalComputationGraphs {
    pub fn get(name: Option<&str>, idx: Option<usize>) -> GlobalComputationGraph {
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

        let res = binding.get(&(name.clone(), idx));
        return match res {
            None => {
                binding.insert((name.clone(), idx), ComputationGraph::new(ComputationGraph_::new(ComputationGraphMode::Standard)));
                drop(binding);
                Self::get(Some(&name), Some(idx))
            }
            Some(computation_graph) => {
                let r: *const ComputationGraph = computation_graph;
                return GlobalComputationGraph(r);
            }
        }
    }
    fn set(name: Option<&str>, idx: Option<usize>, computation_graph: ComputationGraph) {
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
        binding.insert((name, idx), computation_graph);
    }
    pub fn get_with_reset(name: Option<&str>, idx: Option<usize>, mode: ComputationGraphMode) -> GlobalComputationGraph {
        Self::reset(name.clone(), idx, mode);
        Self::get(name, idx)
    }
    pub fn reset(name: Option<&str>, idx: Option<usize>, mode: ComputationGraphMode) {
        let c = Self::get(name, idx);
        c.reset(mode);
    }
    pub fn lock(name: Option<&str>, idx: Option<usize>, locked_name: &str) {
        let locked = Self::get(name, idx).lock();
        GlobalLockedComputationGraphs::insert(locked_name, locked);
    }
}

static mut _GLOBAL_LOCKED_COMPUTATION_GRAPHS: OnceCell<Mutex<HashMap<String, LockedComputationGraph>>> = OnceCell::new();
pub struct GlobalLockedComputationGraphs;
impl GlobalLockedComputationGraphs {
    fn insert(name: &str, locked_computation_graph: LockedComputationGraph) {
        let hashmap = unsafe { _GLOBAL_LOCKED_COMPUTATION_GRAPHS.get_mut() };
        match hashmap {
            None => {
                unsafe { _GLOBAL_LOCKED_COMPUTATION_GRAPHS.get_or_init(|| Mutex::new(HashMap::new())) };
                Self::insert(name, locked_computation_graph);
            }
            Some(hashmap) => {
                let mut binding = hashmap.lock().unwrap();
                binding.insert(name.to_string(), locked_computation_graph);
            }
        }
    }
    pub fn get(name: &str) -> Option<GlobalLockedComputationGraph> {
        let hashmap = unsafe { _GLOBAL_LOCKED_COMPUTATION_GRAPHS.get_or_init(|| Mutex::new(HashMap::new())) };

        let binding = hashmap.lock().unwrap();
        let res = binding.get(name);
        return match res {
            None => { None }
            Some(locked_computation_graph) => {
                let r: *const LockedComputationGraph = locked_computation_graph;
                Some(GlobalLockedComputationGraph(r))
            }
        }
    }
}

pub fn f64ad_jacobian(inputs: &[f64ad], outputs: &[f64ad], order: usize) -> JacobianOutput {
    // assert_eq!(parallel, false, "parallel jacobian computation not working at this time, please use single threaded computation for now.");

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
            match input {
                f64ad::f64ad_var(_) => {
                    let grad = input.forward_mode_grad(add_to_computation_graph);

                    for output in outputs.entries.iter() {
                        let mut new_jacobian_entry_signature = output.signature.clone();
                        new_jacobian_entry_signature.add_input_wrt(input_idx);
                        match &output.value {
                            f64ad::f64ad_var(_) => {
                                let new_value = grad.wrt(&output.value);
                                let new_jacobian_entry = JacobianEntry {
                                    signature: new_jacobian_entry_signature,
                                    value: new_value,
                                };
                                out.entries.push(new_jacobian_entry);
                            }
                            _ => {
                                let new_value = f64ad::f64(0.0);
                                let new_jacobian_entry = JacobianEntry {
                                    signature: new_jacobian_entry_signature,
                                    value: new_value
                                };
                                out.entries.push(new_jacobian_entry);
                            }
                        }
                    }
                }
                _ => { }
            }
        }
    }
    // backwards mode
    else {
        for output in outputs.entries.iter() {
            match &output.value {
                f64ad::f64ad_var(_) => {
                    let grad = output.value.backwards_mode_grad(add_to_computation_graph);

                    for (input_idx, input) in inputs.iter().enumerate() {
                        match input {
                            f64ad::f64ad_var(_) => {
                                let mut new_jacobian_entry_signature = output.signature.clone();
                                new_jacobian_entry_signature.add_input_wrt(input_idx);
                                let new_value = grad.wrt(input);
                                let new_jacobian_entry = JacobianEntry {
                                    signature: new_jacobian_entry_signature,
                                    value: new_value,
                                };
                                out.entries.push(new_jacobian_entry);
                            }
                            _ => { }
                        }
                    }
                }
                _ => {
                    for (input_idx, _) in inputs.iter().enumerate() {
                        let mut new_jacobian_entry_signature = output.signature.clone();
                        new_jacobian_entry_signature.add_input_wrt(input_idx);
                        let new_value = f64ad::f64(0.0);
                        let new_jacobian_entry = JacobianEntry {
                            signature: new_jacobian_entry_signature,
                            value: new_value,
                        };
                        out.entries.push(new_jacobian_entry);
                    }
                }
            }
        }
    }

    out
}

/*
fn f64ad_jacobian_internal_parallel(inputs: &[f64ad], outputs: &JacobianOutput, add_to_computation_graph: bool) -> JacobianOutput {
    let out = JacobianOutput::new();
    let out_mutex = Mutex::new(out);

    let num_inputs = inputs.len();
    let num_outputs = outputs.entries.len();

    // forward mode
    if num_inputs <= num_outputs {
        let inputs_idx_vec: Vec<usize> = (0..num_inputs).collect();
        inputs_idx_vec.par_iter().for_each(|input_idx| {
            let input = inputs[*input_idx];
            match input {
                f64ad::f64ad_var(_) => {
                    let grad = input.forward_mode_grad(add_to_computation_graph);

                    for output in outputs.entries.iter() {
                        let mut new_jacobian_entry_signature = output.signature.clone();
                        new_jacobian_entry_signature.inputs_wrt.push(*input_idx);
                        match &output.value {
                            f64ad::f64ad_var(_) => {
                                let new_value = grad.wrt(&output.value);
                                let new_jacobian_entry = JacobianEntry {
                                    signature: new_jacobian_entry_signature,
                                    value: new_value,
                                };
                                out_mutex.lock().unwrap().entries.push(new_jacobian_entry);
                            }
                            _ => {
                                let new_value = f64ad::f64(0.0);
                                let new_jacobian_entry = JacobianEntry {
                                    signature: new_jacobian_entry_signature,
                                    value: new_value,
                                };
                                out_mutex.lock().unwrap().entries.push(new_jacobian_entry);
                            }
                        }
                    }
                }
                _ => {}
            }
        });
    }
    // backwards mode
    else {
        let outputs_idx_vec: Vec<usize> = (0..num_outputs).collect();
        outputs_idx_vec.par_iter().for_each(|output_idx| {
            let output = &outputs.entries[*output_idx];
            match &output.value {
                f64ad::f64ad_var(_) => {
                    let grad = output.value.backwards_mode_grad(add_to_computation_graph);

                    for (input_idx, input) in inputs.iter().enumerate() {
                        match input {
                            f64ad::f64ad_var(_) => {
                                let mut new_jacobian_entry_signature = output.signature.clone();
                                new_jacobian_entry_signature.inputs_wrt.push(input_idx);
                                let new_value = grad.wrt(input);
                                let new_jacobian_entry = JacobianEntry {
                                    signature: new_jacobian_entry_signature,
                                    value: new_value,
                                };
                                out_mutex.lock().unwrap().entries.push(new_jacobian_entry);
                            }
                            _ => {}
                        }
                    }
                }
                _ => {
                    for (input_idx, _) in inputs.iter().enumerate() {
                        let mut new_jacobian_entry_signature = output.signature.clone();
                        new_jacobian_entry_signature.inputs_wrt.push(input_idx);
                        let new_value = f64ad::f64(0.0);
                        let new_jacobian_entry = JacobianEntry {
                            signature: new_jacobian_entry_signature,
                            value: new_value,
                        };
                        out_mutex.lock().unwrap().entries.push(new_jacobian_entry);
                    }
                }
            }
        });
    }

    out_mutex.into_inner().unwrap()
}
*/

#[derive(Clone, Debug)]
pub struct JacobianOutput {
    entries: Vec<JacobianEntry>
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
            value
        };
        self.entries.push(entry_to_add);
    }
    fn sort(&mut self) {
        self.entries.sort_by(|x, y| x.signature.partial_cmp(&y.signature).unwrap() );
    }
    pub fn get_entry(&self, inputs_wrt: Vec<usize>, output: usize) -> Option<&JacobianEntry> {
        let signature = JacobianEntrySignature::new(output, inputs_wrt);
        let binary_search_res = self.entries.binary_search_by(|x| x.signature.partial_cmp(&signature).unwrap() );
        return match binary_search_res {
            Ok(i) => { Some(&self.entries[i]) }
            Err(_) => { None }
        }
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
    value: f64ad
}
impl JacobianEntry {
    pub fn signature(&self) -> &JacobianEntrySignature {
        &self.signature
    }
    pub fn value(&self) -> f64ad {
        self.value
    }
}

#[derive(Clone, Debug, PartialOrd, PartialEq)]
pub struct JacobianEntrySignature {
    output: usize,
    inputs_wrt: Vec<usize>
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
    pub fn output(&self) -> usize {
        self.output
    }
}

// Addition ////////////////////////////////////////////////////////////////////////////////////////

impl Add for f64ad_var {
    type Output = f64ad_var;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.computation_graph_id, rhs.computation_graph_id);
        let res = self.computation_graph.add_node(NodeType::AdditionTwoParents, tiny_vec!([u32; 2] => self.node_idx, rhs.node_idx), tiny_vec!([f64; 1]));
        return res;
    }
}
impl Add<f64> for f64ad_var {
    type Output = f64ad_var;

    fn add(self, rhs: f64) -> Self::Output {
        let res = self.computation_graph.add_node(NodeType::AdditionOneParent, tiny_vec!([u32; 2] => self.node_idx), tiny_vec!([f64; 1] => rhs));
        return res;
    }
}
impl Add<f64ad_var> for f64 {
    type Output = f64ad_var;

    fn add(self, rhs: f64ad_var) -> Self::Output {
        return rhs + self;
    }
}
impl AddAssign for f64ad_var {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl Add for f64ad_locked_var {
    type Output = f64ad_locked_var;

    fn add(self, rhs: Self) -> Self::Output {
        f64ad_locked_var_operation_two_parents(&self, &rhs, NodeType::AdditionTwoParents)
    }
}
impl Add<f64> for f64ad_locked_var {
    type Output = f64ad_locked_var;

    fn add(self, rhs: f64) -> Self::Output {
        f64ad_locked_var_operation_one_parent(&self, Some(rhs), NodeType::AdditionOneParent)
    }
}
impl Add<f64ad_locked_var> for f64 {
    type Output = f64ad_locked_var;

    fn add(self, rhs: f64ad_locked_var) -> Self::Output {
        return rhs + self;
    }
}
impl AddAssign for f64ad_locked_var {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl Add for f64ad {
    type Output = f64ad;

    fn add(self, rhs: Self) -> Self::Output {
        match (&self, &rhs) {
            (f64ad::f64ad_var(v1), f64ad::f64ad_var(v2)) => { f64ad::f64ad_var(*v1 + *v2) }
            (f64ad::f64ad_locked_var(v1), f64ad::f64ad_locked_var(v2)) => { f64ad::f64ad_locked_var(*v1 + *v2) }
            (f64ad::f64(v1), f64ad::f64(v2)) => { f64ad::f64(*v1 + *v2) }
            (f64ad::f64ad_var(_), f64ad::f64ad_locked_var(_)) => { panic!("unsupported.") }
            (f64ad::f64ad_locked_var(_), f64ad::f64ad_var(_)) => { panic!("unsupported.") }
            (f64ad::f64ad_var(v1), f64ad::f64(v2)) => { f64ad::f64ad_var(*v1 + *v2) }
            (f64ad::f64(v1), f64ad::f64ad_var(v2)) => { f64ad::f64ad_var(*v1 + *v2) }
            (f64ad::f64ad_locked_var(v1), f64ad::f64(v2)) => { f64ad::f64ad_locked_var(*v1 + *v2) }
            (f64ad::f64(v1), f64ad::f64ad_locked_var(v2)) => { f64ad::f64ad_locked_var(*v1 + *v2) }
        }
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
        return rhs + f64ad::f64(self);
    }
}
impl AddAssign for f64ad {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Multiplication //////////////////////////////////////////////////////////////////////////////////

impl Mul for f64ad_var {
    type Output = f64ad_var;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.computation_graph_id, rhs.computation_graph_id);
        let res = self.computation_graph.add_node(NodeType::MultiplicationTwoParents, tiny_vec!([u32; 2] => self.node_idx, rhs.node_idx), tiny_vec!([f64; 1]));
        return res;
    }
}
impl Mul<f64> for f64ad_var {
    type Output = f64ad_var;

    fn mul(self, rhs: f64) -> Self::Output {
        let res = self.computation_graph.add_node(NodeType::MultiplicationOneParent, tiny_vec!([u32; 2] => self.node_idx), tiny_vec!([f64; 1] => rhs));
        return res;
    }
}
impl Mul<f64ad_var> for f64 {
    type Output = f64ad_var;

    fn mul(self, rhs: f64ad_var) -> Self::Output {
        return rhs * self;
    }
}
impl MulAssign for f64ad_var {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl Mul for f64ad_locked_var {
    type Output = f64ad_locked_var;

    fn mul(self, rhs: Self) -> Self::Output {
        f64ad_locked_var_operation_two_parents(&self, &rhs, NodeType::MultiplicationTwoParents)
    }
}
impl Mul<f64> for f64ad_locked_var {
    type Output = f64ad_locked_var;

    fn mul(self, rhs: f64) -> Self::Output {
        f64ad_locked_var_operation_one_parent(&self, Some(rhs), NodeType::MultiplicationOneParent)
    }
}
impl Mul<f64ad_locked_var> for f64 {
    type Output = f64ad_locked_var;

    fn mul(self, rhs: f64ad_locked_var) -> Self::Output {
        return rhs * self;
    }
}
impl MulAssign for f64ad_locked_var {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl Mul for f64ad {
    type Output = f64ad;

    fn mul(self, rhs: Self) -> Self::Output {
        match (&self, &rhs) {
            (f64ad::f64ad_var(v1), f64ad::f64ad_var(v2)) => { f64ad::f64ad_var(*v1 * *v2) }
            (f64ad::f64ad_locked_var(v1), f64ad::f64ad_locked_var(v2)) => { f64ad::f64ad_locked_var(*v1 * *v2) }
            (f64ad::f64(v1), f64ad::f64(v2)) => { f64ad::f64(*v1 * *v2) }
            (f64ad::f64ad_var(_), f64ad::f64ad_locked_var(_)) => { panic!("unsupported.") }
            (f64ad::f64ad_locked_var(_), f64ad::f64ad_var(_)) => { panic!("unsupported.") }
            (f64ad::f64ad_var(v1), f64ad::f64(v2)) => { f64ad::f64ad_var(*v1 * *v2) }
            (f64ad::f64(v1), f64ad::f64ad_var(v2)) => { f64ad::f64ad_var(*v1 * *v2) }
            (f64ad::f64ad_locked_var(v1), f64ad::f64(v2)) => { f64ad::f64ad_locked_var(*v1 * *v2) }
            (f64ad::f64(v1), f64ad::f64ad_locked_var(v2)) => { f64ad::f64ad_locked_var(*v1 * *v2) }
        }
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
        return rhs * self;
    }
}
impl MulAssign for f64ad {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Subtraction /////////////////////////////////////////////////////////////////////////////////////

impl Sub for f64ad_var {
    type Output = f64ad_var;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.computation_graph_id, rhs.computation_graph_id);
        let res = self.computation_graph.add_node(NodeType::SubtractionTwoParents, tiny_vec!([u32; 2] => self.node_idx, rhs.node_idx), tiny_vec!([f64; 1]));
        return res;
    }
}
impl Sub<f64> for f64ad_var {
    type Output = f64ad_var;

    fn sub(self, rhs: f64) -> Self::Output {
        let res = self.computation_graph.add_node(NodeType::SubtractionOneParentLeft, tiny_vec!([u32; 2] => self.node_idx), tiny_vec!([f64; 1] => rhs));
        return res;
    }
}
impl Sub<f64ad_var> for f64 {
    type Output = f64ad_var;

    fn sub(self, rhs: f64ad_var) -> Self::Output {
        let res = rhs.computation_graph.add_node(NodeType::SubtractionOneParentRight, tiny_vec!([u32; 2] => rhs.node_idx), tiny_vec!([f64; 1] => self));
        return res;
    }
}
impl SubAssign for f64ad_var {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl Sub for f64ad_locked_var {
    type Output = f64ad_locked_var;

    fn sub(self, rhs: Self) -> Self::Output {
        f64ad_locked_var_operation_two_parents(&self, &rhs, NodeType::SubtractionTwoParents)
    }
}
impl Sub<f64> for f64ad_locked_var {
    type Output = f64ad_locked_var;

    fn sub(self, rhs: f64) -> Self::Output {
        f64ad_locked_var_operation_one_parent(&self, Some(rhs), NodeType::SubtractionOneParentLeft)
    }
}
impl Sub<f64ad_locked_var> for f64 {
    type Output = f64ad_locked_var;

    fn sub(self, rhs: f64ad_locked_var) -> Self::Output {
        f64ad_locked_var_operation_one_parent(&rhs, Some(self), NodeType::SubtractionOneParentRight)
    }
}
impl SubAssign for f64ad_locked_var {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl Sub for f64ad {
    type Output = f64ad;

    fn sub(self, rhs: Self) -> Self::Output {
        match (&self, &rhs) {
            (f64ad::f64ad_var(v1), f64ad::f64ad_var(v2)) => { f64ad::f64ad_var(*v1 - *v2) }
            (f64ad::f64ad_locked_var(v1), f64ad::f64ad_locked_var(v2)) => { f64ad::f64ad_locked_var(*v1 - *v2) }
            (f64ad::f64(v1), f64ad::f64(v2)) => { f64ad::f64(*v1 - *v2) }
            (f64ad::f64ad_var(_), f64ad::f64ad_locked_var(_)) => { panic!("unsupported.") }
            (f64ad::f64ad_locked_var(_), f64ad::f64ad_var(_)) => { panic!("unsupported.") }
            (f64ad::f64ad_var(v1), f64ad::f64(v2)) => { f64ad::f64ad_var(*v1 - *v2) }
            (f64ad::f64(v1), f64ad::f64ad_var(v2)) => { f64ad::f64ad_var(*v1 - *v2) }
            (f64ad::f64ad_locked_var(v1), f64ad::f64(v2)) => { f64ad::f64ad_locked_var(*v1 - *v2) }
            (f64ad::f64(v1), f64ad::f64ad_locked_var(v2)) => { f64ad::f64ad_locked_var(*v1 - *v2) }
        }
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

    fn sub(self, rhs: f64ad) -> Self::Output { return f64ad::f64(self) - rhs; }
}
impl SubAssign for f64ad {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Division ////////////////////////////////////////////////////////////////////////////////////////

impl Div for f64ad_var {
    type Output = f64ad_var;

    fn div(self, rhs: Self) -> Self::Output {
        assert_eq!(self.computation_graph_id, rhs.computation_graph_id);
        let res = self.computation_graph.add_node(NodeType::DivisionTwoParents, tiny_vec!([u32; 2] => self.node_idx, rhs.node_idx), tiny_vec!([f64; 1]));
        return res;
    }
}
impl Div<f64> for f64ad_var {
    type Output = f64ad_var;

    fn div(self, rhs: f64) -> Self::Output {
        let res = self.computation_graph.add_node(NodeType::DivisionOneParentNumerator, tiny_vec!([u32; 2] => self.node_idx), tiny_vec!([f64; 1] => rhs));
        return res;
    }
}
impl Div<f64ad_var> for f64 {
    type Output = f64ad_var;

    fn div(self, rhs: f64ad_var) -> Self::Output {
        let res = rhs.computation_graph.add_node(NodeType::DivisionOneParentDenominator, tiny_vec!([u32; 2] => rhs.node_idx), tiny_vec!([f64; 1] => self));
        return res;
    }
}
impl DivAssign for f64ad_var {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}
impl Div for f64ad_locked_var {
    type Output = f64ad_locked_var;

    fn div(self, rhs: Self) -> Self::Output {
        f64ad_locked_var_operation_two_parents(&self, &rhs, NodeType::DivisionTwoParents)
    }
}
impl Div<f64> for f64ad_locked_var {
    type Output = f64ad_locked_var;

    fn div(self, rhs: f64) -> Self::Output {
        f64ad_locked_var_operation_one_parent(&self, Some(rhs), NodeType::DivisionOneParentNumerator)
    }
}
impl Div<f64ad_locked_var> for f64 {
    type Output = f64ad_locked_var;

    fn div(self, rhs: f64ad_locked_var) -> Self::Output {
        f64ad_locked_var_operation_one_parent(&rhs, Some(self), NodeType::DivisionOneParentDenominator)
    }
}
impl DivAssign for f64ad_locked_var {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}
impl Div for f64ad {
    type Output = f64ad;

    fn div(self, rhs: Self) -> Self::Output {
        match (&self, &rhs) {
            (f64ad::f64ad_var(v1), f64ad::f64ad_var(v2)) => { f64ad::f64ad_var(*v1 / *v2) }
            (f64ad::f64ad_locked_var(v1), f64ad::f64ad_locked_var(v2)) => { f64ad::f64ad_locked_var(*v1 / *v2) }
            (f64ad::f64(v1), f64ad::f64(v2)) => { f64ad::f64(*v1 / *v2) }
            (f64ad::f64ad_var(_), f64ad::f64ad_locked_var(_)) => { panic!("unsupported.") }
            (f64ad::f64ad_locked_var(_), f64ad::f64ad_var(_)) => { panic!("unsupported.") }
            (f64ad::f64ad_var(v1), f64ad::f64(v2)) => { f64ad::f64ad_var(*v1 / *v2) }
            (f64ad::f64(v1), f64ad::f64ad_var(v2)) => { f64ad::f64ad_var(*v1 / *v2) }
            (f64ad::f64ad_locked_var(v1), f64ad::f64(v2)) => { f64ad::f64ad_locked_var(*v1 / *v2) }
            (f64ad::f64(v1), f64ad::f64ad_locked_var(v2)) => { f64ad::f64ad_locked_var(*v1 / *v2) }
        }
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

////////////////////////////////////////////////////////////////////////////////////////////////////

// Remainder ///////////////////////////////////////////////////////////////////////////////////////

impl Rem for f64ad {
    type Output = f64ad;

    fn rem(self, rhs: Self) -> Self::Output {
        self - (self / rhs).floor() * rhs
    }
}
impl RemAssign for f64ad {
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Neg /////////////////////////////////////////////////////////////////////////////////////////////

impl Neg for f64ad_var {
    type Output = f64ad_var;

    fn neg(self) -> Self::Output {
        let res = self.computation_graph.add_node(NodeType::Neg, tiny_vec!([u32; 2] => self.node_idx), tiny_vec!([f64; 1]));
        return res;
    }
}
impl Neg for f64ad_locked_var {
    type Output = f64ad_locked_var;

    fn neg(self) -> Self::Output {
        f64ad_locked_var_operation_one_parent(&self, None, NodeType::Neg)
    }
}
impl Neg for f64ad {
    type Output = f64ad;

    fn neg(self) -> Self::Output {
        return match &self {
            f64ad::f64ad_var(f) => { f64ad::f64ad_var(f.neg()) }
            f64ad::f64ad_locked_var(f) => { f64ad::f64ad_locked_var(f.neg()) }
            f64ad::f64(f) => { f64ad::f64(f.neg()) }
        };
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl PartialEq for f64ad_var {
    fn eq(&self, other: &Self) -> bool {
        assert_eq!(self.computation_graph_id, other.computation_graph_id);
        match self.mode {
            ComputationGraphMode::Standard => { self.value() == other.value() }
            ComputationGraphMode::Lock => { panic!("cannot compute PartialEq on ComputationGraphMode of Lock.  computation graph: {}", self.computation_graph_id) }
        }
    }
}
impl PartialEq for f64ad_locked_var {
    fn eq(&self, _other: &Self) -> bool {
        panic!("cannot compute PartialEq on f64ad_locked_var");
    }
}
impl PartialEq for f64ad {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (f64ad::f64ad_var(v1), f64ad::f64ad_var(v2)) => { *v1 == *v2 }
            (f64ad::f64ad_locked_var(v1), f64ad::f64ad_locked_var(v2)) => { *v1 == *v2 }
            (f64ad::f64(v1), f64ad::f64(v2)) => { *v1 == *v2 }
            _ => { panic!("unsupported.") }
        }
    }
}
impl PartialOrd for f64ad_var {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        assert_eq!(self.computation_graph_id, other.computation_graph_id);

        match &self.mode {
            ComputationGraphMode::Standard => {
                let v1 = self.value();
                let v2 = self.value();

                if v1 == v2 {
                    return Some(Ordering::Equal);
                } else if v1 > v2 {
                    return Some(Ordering::Greater);
                } else if v1 < v2 {
                    return Some(Ordering::Less);
                } else {
                    unreachable!();
                }
            }
            ComputationGraphMode::Lock => {
                panic!("cannot partial_cmp f64ad_var in mode Lock.  Computation graph {}", self.computation_graph_id)
            }
        }
    }
}
impl PartialOrd for f64ad_locked_var {
    fn partial_cmp(&self, _other: &Self) -> Option<Ordering> {
        panic!("cannot partial_cmp f64ad_locked_var");
    }
}
impl PartialOrd for f64ad {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (f64ad::f64ad_var(v1), f64ad::f64ad_var(v2)) => { v1.partial_cmp(v2) }
            (f64ad::f64ad_locked_var(v1), f64ad::f64ad_locked_var(v2)) => { v1.partial_cmp(v2) }
            (f64ad::f64(v1), f64ad::f64(v2)) => { v1.partial_cmp(v2) }
            (f64ad::f64ad_var(_), f64ad::f64ad_locked_var(_)) => { panic!("unsupported.") }
            (f64ad::f64ad_locked_var(_), f64ad::f64ad_var(_)) => { panic!("unsupported.") }
            (f64ad::f64ad_var(v1), f64ad::f64(v2)) => { v1.value().partial_cmp(v2) }
            (f64ad::f64(v1), f64ad::f64ad_var(v2)) => { v1.partial_cmp(&v2.value()) }
            (f64ad::f64ad_locked_var(_), f64ad::f64(_)) => { panic!("unsupported.") }
            (f64ad::f64(_), f64ad::f64ad_locked_var(_)) => { panic!("unsupported.") }
        }
    }
}
impl Display for f64ad {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        f.write_str(&format!("{:?}", self)).expect("error");
        Ok(())
    }
}
