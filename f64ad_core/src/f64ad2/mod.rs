pub mod trait_impls;

use std::cmp::Ordering;
use std::fmt::{Debug, Display, Formatter};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};
use rand::prelude::*;
use simba::scalar::ComplexField;
use tinyvec::*;

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy)]
pub enum f64ad {
    f64ad_var(f64ad_var),
    f64ad_locked_var(f64ad_locked_var),
    f64(f64)
}
impl f64ad {
    pub fn value(&self) -> f64 {
        return match self {
            f64ad::f64ad_var(f) => {f.value()}
            f64ad::f64ad_locked_var(f) => { f.value() }
            f64ad::f64(f) => { *f }
        }
    }
    pub fn node_idx(&self) -> u32 {
        match self {
            f64ad::f64ad_var(a) => { a.node_idx }
            f64ad::f64ad_locked_var(a) => { a.node_idx }
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
}
impl Default for f64ad {
    fn default() -> Self {
        Self::f64(0.0)
    }
}
unsafe impl Sync for f64ad {  }

#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub struct f64ad_var {
    computation_graph_id: usize,
    node_idx: u32,
    mode: ComputationGraphMode,
    computation_graph: ComputationGraphRawPointer
}
impl f64ad_var {
    pub (crate) fn value(&self) -> f64 {
        return unsafe {
            (*self.computation_graph.0).nodes[self.node_idx as usize].value
        }
    }
    pub (crate) fn new(computation_graph_id: usize, node_idx: u32, mode: ComputationGraphMode, computation_graph: *mut ComputationGraph) -> Self {
        Self {
            computation_graph_id,
            node_idx,
            mode,
            computation_graph: ComputationGraphRawPointer(computation_graph)
        }
    }
    fn forward_mode_grad(&self, add_to_computation_graph: bool) -> ForwardModeGradOutput {
        unsafe {
            let l = (*self.computation_graph.0).nodes.len();
            let mut derivs = vec![f64ad::f64(0.0); l];
            if add_to_computation_graph {
                let mut computation_graph = &mut (*self.computation_graph.0);
                let v = computation_graph.spawn_f64ad_var(1.0);
                derivs[self.node_idx as usize] = v;
            } else {
                derivs[self.node_idx as usize] = f64ad::f64(1.0);
            }

            let mut computation_graph = &mut (*self.computation_graph.0);
            for node_idx in 0..l {
                let node = &computation_graph.nodes[node_idx];
                let node_type = node.node_type.clone();
                let parent_nodes = node.parent_nodes.clone();
                let constant_operands = node.constant_operands.clone();
                let derivatives_of_value_wrt_parent_values = node_type.compute_derivatives_of_value_wrt_parent_values(&parent_nodes, &constant_operands, computation_graph, add_to_computation_graph);
                for (i, d) in derivatives_of_value_wrt_parent_values.iter().enumerate() {
                    let parent_idx = parent_nodes[i];
                    let parent_deriv = derivs[parent_idx as usize];
                    derivs[node_idx] += parent_deriv * *d;
                }
            }

            return ForwardModeGradOutput { derivs }
        }
    }
    fn backwards_mode_grad(&self, add_to_computation_graph: bool) -> BackwardsModeGradOutput {
        unsafe {
            let l = (*self.computation_graph.0).nodes.len();
            let mut derivs = vec![f64ad::f64(0.0); l];
            if add_to_computation_graph {
                let mut computation_graph = &mut (*self.computation_graph.0);
                let v = computation_graph.spawn_f64ad_var(1.0);
                derivs[self.node_idx as usize] = v;
            } else {
                derivs[self.node_idx as usize] = f64ad::f64(1.0);
            }

            let mut computation_graph = &mut (*self.computation_graph.0);
            for node_idx in (0..l).rev() {
                let curr_deriv = derivs[node_idx];
                let node = &computation_graph.nodes[node_idx].clone();
                let node_type = node.node_type.clone();
                let parent_nodes = node.parent_nodes.clone();
                let constant_operands = node.constant_operands.clone();
                let derivatives_of_value_wrt_parent_values = node_type.compute_derivatives_of_value_wrt_parent_values(&parent_nodes, &constant_operands, computation_graph, add_to_computation_graph);
                for (i, d) in derivatives_of_value_wrt_parent_values.iter().enumerate() {
                    let parent_idx = parent_nodes[i];
                    derivs[parent_idx as usize] += curr_deriv * *d;
                }
            }

            return BackwardsModeGradOutput { derivs }
        }
    }
}
impl Debug for f64ad_var {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("f64ad_var{ ").expect("error");
        unsafe {
            f.write_str(&format!("value: {:?}, ", (*self.computation_graph.0).nodes[self.node_idx as usize].value)).expect("error");
        }
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

#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub struct f64ad_locked_var {
    locked_computation_graph_id: usize,
    node_idx: u32,
    locked_computation_graph: LockedComputationGraphRawPointer
}
impl f64ad_locked_var {
    pub (crate) fn value(&self) -> f64 {
        return unsafe {
            (*self.locked_computation_graph.0).computation_graph.nodes[self.node_idx as usize].value
        }
    }
}
impl Debug for f64ad_locked_var {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("f64ad_locked_var{ ").expect("error");
        unsafe {
            f.write_str(&format!("value: {:?}, ", (*self.locked_computation_graph.0).computation_graph.nodes[self.node_idx as usize].value)).expect("error");
        }
        f.write_str(&format!("node_idx: {:?}", self.node_idx)).expect("error");
        f.write_str(" }").expect("error");

        Ok(())
    }
}

pub (crate) fn f64ad_locked_var_operation_two_parents(lhs: &f64ad_locked_var, rhs: &f64ad_locked_var, node_type: NodeType) -> f64ad_locked_var {
    assert_eq!(lhs.locked_computation_graph_id, rhs.locked_computation_graph_id);
    unsafe {
        let locked_computation_graph = &mut (*lhs.locked_computation_graph.0);
        let node_idx = locked_computation_graph.curr_count as usize;
        let computation_graph = &mut locked_computation_graph.computation_graph;
        let node = &computation_graph.nodes[node_idx];
        assert_eq!(node.node_type, node_type);
        let value = node.node_type.compute_value(&tiny_vec!([u32; 2] => lhs.node_idx, rhs.node_idx), &tiny_vec!([f64; 1]), computation_graph);
        computation_graph.nodes[node_idx].value = value;
        locked_computation_graph.curr_count += 1;
        return f64ad_locked_var {
            locked_computation_graph_id: lhs.locked_computation_graph_id,
            node_idx: node_idx as u32,
            locked_computation_graph: LockedComputationGraphRawPointer(locked_computation_graph as *mut LockedComputationGraph_)
        };
    }
}
pub (crate) fn f64ad_locked_var_operation_one_parent(v: &f64ad_locked_var, constant_operand: Option<f64>, node_type: NodeType) -> f64ad_locked_var {
    unsafe {
        let locked_computation_graph = &mut (*v.locked_computation_graph.0);
        let node_idx = locked_computation_graph.curr_count as usize;
        let computation_graph = &mut locked_computation_graph.computation_graph;
        let node = &computation_graph.nodes[node_idx];
        assert_eq!(node.node_type, node_type);
        let value = match constant_operand {
            None => { node.node_type.compute_value(&tiny_vec!([u32; 2] => v.node_idx), &tiny_vec!([f64; 1]), computation_graph) }
            Some(constant_operand) => { node.node_type.compute_value(&tiny_vec!([u32; 2] => v.node_idx), &tiny_vec!([f64; 1] => constant_operand), computation_graph) }
        };
        computation_graph.nodes[node_idx].value = value;
        locked_computation_graph.curr_count += 1;
        return f64ad_locked_var {
            locked_computation_graph_id: v.locked_computation_graph_id,
            node_idx: node_idx as u32,
            locked_computation_graph: LockedComputationGraphRawPointer(locked_computation_graph as *mut LockedComputationGraph_)
        };
    }
}

#[derive(Clone, Debug)]
pub struct ComputationGraph {
    id: usize,
    name: String,
    nodes: Vec<F64ADNode>,
    mode: ComputationGraphMode
}
impl ComputationGraph {
    pub fn new(mode: ComputationGraphMode, name: Option<&str>) -> Self {
        let mut rng = rand::thread_rng();
        let id: usize = rng.gen();
        Self {
            id,
            name: match name {
                None => { "".to_string() }
                Some(name) => { name.to_string() }
            },
            nodes: Vec::with_capacity(1_000_000),
            mode
        }
    }
    pub fn reset(&mut self) {
        *self = Self::new(self.mode.clone(), Some(&self.name));
    }
    pub fn spawn_f64ad_var(&mut self, value: f64) -> f64ad {
        let node_idx = self.nodes.len();

        let f = f64ad_var {
            computation_graph_id: self.id,
            node_idx: node_idx as u32,
            mode: self.mode.clone(),
            computation_graph: ComputationGraphRawPointer(self as *mut ComputationGraph)
        };

        let n = F64ADNode {
            node_idx: node_idx as u32,
            value,
            node_type: NodeType::InputVar,
            constant_operands: tiny_vec!([f64; 1]),
            parent_nodes: tiny_vec!([u32; 2]),
            child_nodes: tiny_vec!([u32; 5])
        };

        self.nodes.push(n);

        return f64ad::f64ad_var(f);
    }
    pub fn lock<T>(&self, name: Option<&str>, template_output: T) -> LockedComputationGraph<T> {
        LockedComputationGraph::new(name, self.clone(), template_output)
    }
    fn add_node(&mut self, node_type: NodeType, parent_nodes: TinyVec<[u32; 2]>, constant_operands: TinyVec<[f64; 1]>) -> f64ad_var {
        let value = node_type.compute_value(&parent_nodes, &constant_operands, self);
        let node_idx = self.nodes.len() as u32;
        for parent_node in &parent_nodes { self.nodes[*parent_node as usize].child_nodes.push(node_idx); }
        let node = F64ADNode::new(node_idx, value, node_type, constant_operands, parent_nodes);
        self.nodes.push(node);

        f64ad_var {
            computation_graph_id: self.id,
            node_idx,
            mode: self.mode.clone(),
            computation_graph: ComputationGraphRawPointer(self as *mut ComputationGraph)
        }
    }
}

#[derive(Clone, Debug, Copy)]
pub struct ComputationGraphRawPointer(*mut ComputationGraph);
unsafe impl Sync for ComputationGraphRawPointer { }
unsafe impl Send for ComputationGraphRawPointer { }

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ComputationGraphMode {
    Standard,
    Lock
}

#[derive(Clone, Debug)]
pub (crate) struct LockedComputationGraph_ {
    id: usize,
    name: String,
    computation_graph: ComputationGraph,
    curr_count: u32
}

#[derive(Clone, Debug, Copy)]
pub struct LockedComputationGraphRawPointer(*mut LockedComputationGraph_);
unsafe impl Sync for LockedComputationGraphRawPointer { }
unsafe impl Send for LockedComputationGraphRawPointer { }

#[derive(Clone, Debug)]
pub struct LockedComputationGraph<T> {
    locked_computation_graph: LockedComputationGraph_,
    template_output: T
}
impl<T> LockedComputationGraph<T> {
    fn new(name: Option<&str>, computation_graph: ComputationGraph, template_output: T) -> Self {
        assert_eq!(computation_graph.mode, ComputationGraphMode::Lock, "Computation graph mode must be Lock in order to create a LockedComputationGraph.");

        let mut rng = rand::thread_rng();
        let id: usize = rng.gen();

        let locked_computation_graph = LockedComputationGraph_ {
            id,
            name: match name {
                None => { "".to_string() }
                Some(name) => { name.to_string() }
            },
            computation_graph,
            curr_count: 0
        };

        Self {
            locked_computation_graph,
            template_output
        }
    }
    pub fn set_value(&mut self, node_idx: usize, value: f64) {
        let node = &mut self.locked_computation_graph.computation_graph.nodes[node_idx];
        assert_eq!(node.parent_nodes.len(), 0, "cannot set value of a non-initial node.");
        node.value = value;
    }
    pub fn get_value(&self, idx: usize) -> f64 {
        return self.locked_computation_graph.computation_graph.nodes[idx].value
    }
    pub fn spawn_locked_var(&mut self, value: f64) -> f64ad {
        let node_idx = self.locked_computation_graph.curr_count;
        self.set_value(node_idx as usize, value);
        let f = f64ad_locked_var {
            locked_computation_graph_id: self.locked_computation_graph.id,
            node_idx: node_idx as u32,
            locked_computation_graph: LockedComputationGraphRawPointer(&mut self.locked_computation_graph as *mut LockedComputationGraph_),
        };
        self.locked_computation_graph.curr_count += 1;
        return f64ad::f64ad_locked_var(f);
    }
    pub fn template_output(&self) -> &T {
        &self.template_output
    }
    pub fn push_forward_compute(&mut self) {
        let l = self.locked_computation_graph.computation_graph.nodes.len();
        for idx in 0..l {
            unsafe {
                let node = &self.locked_computation_graph.computation_graph.nodes[idx];
                match node.node_type {
                    NodeType::None => {  }
                    NodeType::InputVar => {  }
                    _ => {
                        let value = node.node_type.compute_value(&node.parent_nodes, &node.constant_operands, &self.locked_computation_graph.computation_graph);
                        self.locked_computation_graph.computation_graph.nodes[idx].value = value;
                    }
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct F64ADNode {
    node_idx: u32,
    value: f64,
    node_type: NodeType,
    constant_operands: TinyVec<[f64; 1]>,
    parent_nodes: TinyVec<[u32; 2]>,
    child_nodes: TinyVec<[u32; 5]>
}
impl F64ADNode {
    pub fn new(node_idx: u32, value: f64, node_type: NodeType, constant_operands: TinyVec<[f64; 1]>, parent_nodes: TinyVec<[u32; 2]>) -> Self {
        Self {
            node_idx,
            value,
            node_type,
            constant_operands,
            parent_nodes,
            child_nodes: tiny_vec!([u32; 5])
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
    PowTwoParents
}
impl NodeType {
    pub fn compute_value(&self, parent_nodes: &TinyVec<[u32; 2]>, constant_operands: &TinyVec<[f64; 1]>, computation_graph: &ComputationGraph) -> f64 {
        match self {
            NodeType::None => { panic!("Cannot compute value on node type None.") }
            NodeType::InputVar => { panic!("Cannot compute value on node type InputVar.") }
            NodeType::AdditionOneParent => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return parent_value + constant_operands[0];
            }
            NodeType::AdditionTwoParents => {
                let parent_value_0 = computation_graph.nodes[parent_nodes[0] as usize].value;
                let parent_value_1 = computation_graph.nodes[parent_nodes[1] as usize].value;
                return parent_value_0 + parent_value_1;
            }
            NodeType::MultiplicationOneParent => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return parent_value * constant_operands[0];
            }
            NodeType::MultiplicationTwoParents => {
                let parent_value_0 = computation_graph.nodes[parent_nodes[0] as usize].value;
                let parent_value_1 = computation_graph.nodes[parent_nodes[1] as usize].value;
                return parent_value_0 * parent_value_1;
            }
            NodeType::SubtractionOneParentLeft => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return parent_value - constant_operands[0];
            }
            NodeType::SubtractionOneParentRight => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return constant_operands[0] - parent_value;
            }
            NodeType::SubtractionTwoParents => {
                let parent_value_0 = computation_graph.nodes[parent_nodes[0] as usize].value;
                let parent_value_1 = computation_graph.nodes[parent_nodes[1] as usize].value;
                return parent_value_0 - parent_value_1;
            }
            NodeType::DivisionOneParentDenominator => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return constant_operands[0] / parent_value;
            }
            NodeType::DivisionOneParentNumerator => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return parent_value / constant_operands[0];
            }
            NodeType::DivisionTwoParents => {
                let parent_value_0 = computation_graph.nodes[parent_nodes[0] as usize].value;
                let parent_value_1 = computation_graph.nodes[parent_nodes[1] as usize].value;
                return parent_value_0 / parent_value_1;
            }
            NodeType::Neg => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return -parent_value;
            }
            NodeType::Abs => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return parent_value.abs();
            }
            NodeType::Signum => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return parent_value.signum();
            }
            NodeType::MaxOneParent => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return parent_value.max(constant_operands[0]);
            }
            NodeType::MaxTwoParents => {
                let parent_value_0 = computation_graph.nodes[parent_nodes[0] as usize].value;
                let parent_value_1 = computation_graph.nodes[parent_nodes[1] as usize].value;
                return parent_value_0.max(parent_value_1);
            }
            NodeType::MinOneParent => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return parent_value.min(constant_operands[0]);
            }
            NodeType::MinTwoParents => {
                let parent_value_0 = computation_graph.nodes[parent_nodes[0] as usize].value;
                let parent_value_1 = computation_graph.nodes[parent_nodes[1] as usize].value;
                return parent_value_0.min(parent_value_1);
            }
            NodeType::Atan2OneParentLeft => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return parent_value.atan2(constant_operands[0]);
            }
            NodeType::Atan2OneParentRight => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return constant_operands[0].atan2(parent_value);
            }
            NodeType::Atan2TwoParents => {
                let parent_value_0 = computation_graph.nodes[parent_nodes[0] as usize].value;
                let parent_value_1 = computation_graph.nodes[parent_nodes[1] as usize].value;
                return parent_value_0.atan2(parent_value_1);
            }
            NodeType::Floor => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return parent_value.floor();
            }
            NodeType::Ceil => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return parent_value.ceil();
            }
            NodeType::Round => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return parent_value.round();
            }
            NodeType::Trunc => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return parent_value.trunc();
            }
            NodeType::Fract => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return parent_value.fract();
            }
            NodeType::Sin => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return parent_value.sin();
            }
            NodeType::Cos => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return parent_value.cos();
            }
            NodeType::Tan => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return parent_value.tan();
            }
            NodeType::Asin => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return parent_value.asin();
            }
            NodeType::Acos => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return parent_value.acos();
            }
            NodeType::Atan => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return parent_value.atan();
            }
            NodeType::Sinh => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return parent_value.sinh();
            }
            NodeType::Cosh => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return parent_value.cosh();
            }
            NodeType::Tanh => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return parent_value.tanh();
            }
            NodeType::Asinh => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return parent_value.asinh();
            }
            NodeType::Acosh => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return parent_value.acosh();
            }
            NodeType::Atanh => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return parent_value.atanh();
            }
            NodeType::LogOneParentArgument => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return parent_value.log(constant_operands[0]);
            }
            NodeType::LogOneParentBase => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return constant_operands[0].log(parent_value);
            }
            NodeType::LogTwoParents => {
                let parent_value_0 = computation_graph.nodes[parent_nodes[0] as usize].value;
                let parent_value_1 = computation_graph.nodes[parent_nodes[1] as usize].value;
                return parent_value_0.log(parent_value_1);
            }
            NodeType::Sqrt => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return parent_value.sqrt();
            }
            NodeType::Exp => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return parent_value.exp();
            }
            NodeType::PowOneParentArgument => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return parent_value.powf(constant_operands[0]);
            }
            NodeType::PowOneParentExponent => {
                let parent_value = computation_graph.nodes[parent_nodes[0] as usize].value;
                return constant_operands[0].powf(parent_value);
            }
            NodeType::PowTwoParents => {
                let parent_value_0 = computation_graph.nodes[parent_nodes[0] as usize].value;
                let parent_value_1 = computation_graph.nodes[parent_nodes[1] as usize].value;
                return parent_value_0.powf(parent_value_1);
            }
        }
    }
    pub fn compute_derivatives_of_value_wrt_parent_values(&self, parent_nodes: &TinyVec<[u32; 2]>, constant_operands: &TinyVec<[f64; 1]>, computation_graph: &mut ComputationGraph, add_to_computation_graph: bool) -> TinyVec<[f64ad; 2]> {
        match self {
            NodeType::None => { panic!("Cannot compute derivatives on node type None.") }
            NodeType::InputVar => { return tiny_vec!([f64ad; 2]) }
            NodeType::AdditionOneParent => { return tiny_vec!([f64ad; 2] => f64ad::f64(1.0)); }
            NodeType::AdditionTwoParents => { return tiny_vec!([f64ad; 2] => f64ad::f64(1.0), f64ad::f64(1.0)); }
            NodeType::MultiplicationOneParent => { return tiny_vec!([f64ad; 2] => f64ad::f64(constant_operands[0])); }
            NodeType::MultiplicationTwoParents => {
                return if add_to_computation_graph {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let parent_node_1 = &computation_graph.nodes[parent_nodes[1] as usize];
                    let id = computation_graph.id;
                    let mode = computation_graph.mode.clone();
                    let node_idx_0 = parent_node_0.node_idx;
                    let node_idx_1 = parent_node_1.node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_1, mode, computation_graph as *mut ComputationGraph));
                    let f1 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph as *mut ComputationGraph));
                    tiny_vec!([f64ad; 2] => f0, f1)
                } else {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let parent_node_1 = &computation_graph.nodes[parent_nodes[1] as usize];
                    tiny_vec!([f64ad; 2] => f64ad::f64(parent_node_1.value), f64ad::f64(parent_node_0.value))
                }
            }
            NodeType::SubtractionOneParentLeft => { return tiny_vec!([f64ad; 2] => f64ad::f64(1.0)); }
            NodeType::SubtractionOneParentRight => { return tiny_vec!([f64ad; 2] => f64ad::f64(-1.0)); }
            NodeType::SubtractionTwoParents => { return tiny_vec!([f64ad; 2] => f64ad::f64(1.0), f64ad::f64(-1.0)); }
            NodeType::DivisionOneParentDenominator => {
                return if add_to_computation_graph {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let id = computation_graph.id;
                    let mode = computation_graph.mode.clone();
                    let node_idx_0 = parent_node_0.node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph as *mut ComputationGraph));
                    let ret = -(constant_operands[0] / (f0 * f0));
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(-(constant_operands[0] / (v * v))))
                }
            }
            NodeType::DivisionOneParentNumerator => {
                return tiny_vec!([f64ad; 2] => f64ad::f64(1.0/constant_operands[0]));
            }
            NodeType::DivisionTwoParents => {
                return if add_to_computation_graph {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let parent_node_1 = &computation_graph.nodes[parent_nodes[1] as usize];
                    let id = computation_graph.id;
                    let mode = computation_graph.mode.clone();
                    let node_idx_0 = parent_node_0.node_idx;
                    let node_idx_1 = parent_node_1.node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph as *mut ComputationGraph));
                    let f1 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_1, mode, computation_graph as *mut ComputationGraph));
                    tiny_vec!([f64ad; 2] => 1.0/f1, -f0/(f1*f1))
                } else {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let parent_node_1 = &computation_graph.nodes[parent_nodes[1] as usize];
                    tiny_vec!([f64ad; 2] => f64ad::f64(1.0/parent_node_1.value), f64ad::f64(-parent_node_0.value/(parent_node_1.value * parent_node_1.value)))
                }
            }
            NodeType::Neg => { return tiny_vec!([f64ad; 2] => f64ad::f64(-1.0)); }
            NodeType::Abs => {
                let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
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
                let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                let v = parent_node_0.value;
                return if v >= constant_operands[0] { tiny_vec!([f64ad; 2] => f64ad::f64(1.0)) }
                else { tiny_vec!([f64ad; 2] => f64ad::f64(0.0)) }
            }
            NodeType::MaxTwoParents => {
                let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                let parent_node_1 = &computation_graph.nodes[parent_nodes[1] as usize];
                let v0 = parent_node_0.value;
                let v1 = parent_node_1.value;
                return if v0 >= v1 { tiny_vec!([f64ad; 2] => f64ad::f64(1.0), f64ad::f64(0.0))  }
                else { tiny_vec!([f64ad; 2] => f64ad::f64(0.0), f64ad::f64(1.0)) }
            }
            NodeType::MinOneParent => {
                let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                let v = parent_node_0.value;
                return if v <= constant_operands[0] { tiny_vec!([f64ad; 2] => f64ad::f64(1.0)) }
                else { tiny_vec!([f64ad; 2] => f64ad::f64(0.0)) }
            }
            NodeType::MinTwoParents => {
                let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                let parent_node_1 = &computation_graph.nodes[parent_nodes[1] as usize];
                let v0 = parent_node_0.value;
                let v1 = parent_node_1.value;
                return if v0 <= v1 { tiny_vec!([f64ad; 2] => f64ad::f64(1.0), f64ad::f64(0.0))  }
                else { tiny_vec!([f64ad; 2] => f64ad::f64(0.0), f64ad::f64(1.0)) }
            }
            NodeType::Atan2OneParentLeft => {
                return if add_to_computation_graph {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let id = computation_graph.id;
                    let mode = computation_graph.mode.clone();
                    let node_idx_0 = parent_node_0.node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph as *mut ComputationGraph));
                    let ret = constant_operands[0]/ (constant_operands[0].powi(2) + (f0 * f0));
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(constant_operands[0]/ (constant_operands[0].powi(2) + v.powi(2))))
                }
            }
            NodeType::Atan2OneParentRight => {
                return if add_to_computation_graph {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let id = computation_graph.id;
                    let mode = computation_graph.mode.clone();
                    let node_idx_0 = parent_node_0.node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph as *mut ComputationGraph));
                    let ret = -constant_operands[0]/ (constant_operands[0].powi(2) + (f0 * f0));
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(-constant_operands[0]/ (constant_operands[0].powi(2) + v.powi(2))))
                }
            }
            NodeType::Atan2TwoParents => {
                return if add_to_computation_graph {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let parent_node_1 = &computation_graph.nodes[parent_nodes[1] as usize];
                    let id = computation_graph.id;
                    let mode = computation_graph.mode.clone();
                    let node_idx_0 = parent_node_0.node_idx;
                    let node_idx_1 = parent_node_1.node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph as *mut ComputationGraph));
                    let f1 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_1, mode, computation_graph as *mut ComputationGraph));
                    tiny_vec!([f64ad; 2] => f1/(f0*f0 + f1*f1), -f0/(f0*f0 + f1*f1))
                } else {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let parent_node_1 = &computation_graph.nodes[parent_nodes[1] as usize];
                    let v0 = parent_node_0.value;
                    let v1 = parent_node_1.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(-v0/(v0*v0 + v1*v1)))
                }
            }
            NodeType::Floor => { return tiny_vec!([f64ad; 2] => f64ad::f64(0.0)); }
            NodeType::Ceil => { return tiny_vec!([f64ad; 2] => f64ad::f64(0.0)); }
            NodeType::Round => { return tiny_vec!([f64ad; 2] => f64ad::f64(0.0)); }
            NodeType::Trunc => { return tiny_vec!([f64ad; 2] => f64ad::f64(0.0)); }
            NodeType::Fract => { return tiny_vec!([f64ad; 2] => f64ad::f64(0.0)); }
            NodeType::Sin => {
                return if add_to_computation_graph {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let id = computation_graph.id;
                    let mode = computation_graph.mode.clone();
                    let node_idx_0 = parent_node_0.node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph as *mut ComputationGraph));
                    let ret = f0.cos();
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(v.cos()))
                }
            }
            NodeType::Cos => {
                return if add_to_computation_graph {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let id = computation_graph.id;
                    let mode = computation_graph.mode.clone();
                    let node_idx_0 = parent_node_0.node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph as *mut ComputationGraph));
                    let ret = -f0.sin();
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(-v.sin()))
                }
            }
            NodeType::Tan => {
                return if add_to_computation_graph {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let id = computation_graph.id;
                    let mode = computation_graph.mode.clone();
                    let node_idx_0 = parent_node_0.node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph as *mut ComputationGraph));
                    let c = f0.cos();
                    let ret = 1.0/(c*c);
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    let c = v.cos();
                    tiny_vec!([f64ad; 2] => f64ad::f64(1.0/(c*c)))
                }
            }
            NodeType::Asin => {
                return if add_to_computation_graph {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let id = computation_graph.id;
                    let mode = computation_graph.mode.clone();
                    let node_idx_0 = parent_node_0.node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph as *mut ComputationGraph));
                    let ret = 1.0/(1.0 - f0*f0).sqrt();
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(1.0/(1.0 - v*v).sqrt()))
                }
            }
            NodeType::Acos => {
                return if add_to_computation_graph {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let id = computation_graph.id;
                    let mode = computation_graph.mode.clone();
                    let node_idx_0 = parent_node_0.node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph as *mut ComputationGraph));
                    let ret = -1.0/(1.0 - f0*f0).sqrt();
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(-1.0/(1.0 - v*v).sqrt()))
                }
            }
            NodeType::Atan => {
                return if add_to_computation_graph {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let id = computation_graph.id;
                    let mode = computation_graph.mode.clone();
                    let node_idx_0 = parent_node_0.node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph as *mut ComputationGraph));
                    let ret = 1.0/(f0*f0 + 1.0);
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(1.0/(v*v + 1.0)))
                }
            }
            NodeType::Sinh => {
                return if add_to_computation_graph {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let id = computation_graph.id;
                    let mode = computation_graph.mode.clone();
                    let node_idx_0 = parent_node_0.node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph as *mut ComputationGraph));
                    let ret = f0.cosh();
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(v.cosh()))
                }
            }
            NodeType::Cosh => {
                return if add_to_computation_graph {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let id = computation_graph.id;
                    let mode = computation_graph.mode.clone();
                    let node_idx_0 = parent_node_0.node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph as *mut ComputationGraph));
                    let ret = f0.sinh();
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(v.sinh()))
                }
            }
            NodeType::Tanh => {
                return if add_to_computation_graph {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let id = computation_graph.id;
                    let mode = computation_graph.mode.clone();
                    let node_idx_0 = parent_node_0.node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph as *mut ComputationGraph));
                    let c = f0.cosh();
                    let ret = 1.0/(c*c);
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(1.0/v.cosh().powi(2)))
                }
            }
            NodeType::Asinh => {
                return if add_to_computation_graph {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let id = computation_graph.id;
                    let mode = computation_graph.mode.clone();
                    let node_idx_0 = parent_node_0.node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph as *mut ComputationGraph));
                    let ret = 1.0 / (f0*f0 + 1.0).sqrt();
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(1.0 / (v*v + 1.0).sqrt()))
                }
            }
            NodeType::Acosh => {
                return if add_to_computation_graph {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let id = computation_graph.id;
                    let mode = computation_graph.mode.clone();
                    let node_idx_0 = parent_node_0.node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph as *mut ComputationGraph));
                    let ret = 1.0 / ((f0 - 1.0).sqrt()*(f0 + 1.0).sqrt());
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(1.0 / ((v - 1.0).sqrt()*(v + 1.0).sqrt())))
                }
            }
            NodeType::Atanh => {
                return if add_to_computation_graph {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let id = computation_graph.id;
                    let mode = computation_graph.mode.clone();
                    let node_idx_0 = parent_node_0.node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph as *mut ComputationGraph));
                    let ret = 1.0 / (1.0 - f0*f0);
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(1.0 / (1.0 - v*v)))
                }
            }
            NodeType::LogOneParentArgument => {
                return if add_to_computation_graph {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let id = computation_graph.id;
                    let mode = computation_graph.mode.clone();
                    let node_idx_0 = parent_node_0.node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph as *mut ComputationGraph));
                    let ret = 1.0 / (f0*constant_operands[0].ln());
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(1.0 / (v*constant_operands[0].ln())))
                }
            }
            NodeType::LogOneParentBase => {
                return if add_to_computation_graph {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let id = computation_graph.id;
                    let mode = computation_graph.mode.clone();
                    let node_idx_0 = parent_node_0.node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph as *mut ComputationGraph));
                    let ly = constant_operands[0].ln();
                    let lx = f0.ln();
                    println!(" >> {:?}", lx);
                    let ret = -ly / (f0*(lx*lx));
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    let ly = constant_operands[0].ln();
                    let lx = v.ln();
                    let ret = -ly / (v*(lx*lx));
                    tiny_vec!([f64ad; 2] => f64ad::f64(ret))
                }
            }
            NodeType::LogTwoParents => {
                return if add_to_computation_graph {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let parent_node_1 = &computation_graph.nodes[parent_nodes[1] as usize];
                    let id = computation_graph.id;
                    let mode = computation_graph.mode.clone();
                    let node_idx_0 = parent_node_0.node_idx;
                    let node_idx_1 = parent_node_1.node_idx;
                    let argument = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph as *mut ComputationGraph));
                    let base = f64ad::f64ad_var(f64ad_var::new(id, node_idx_1, mode, computation_graph as *mut ComputationGraph));
                    let ret0 = 1.0 / (argument * base.ln());
                    let lb = base.ln();
                    let ret1 = -argument.ln() / (base * (lb * lb));
                    tiny_vec!([f64ad; 2] => ret0, ret1)
                } else {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let parent_node_1 = &computation_graph.nodes[parent_nodes[1] as usize];
                    let argument = parent_node_0.value;
                    let base = parent_node_1.value;
                    let ret0 = 1.0 / (argument * base.ln());
                    let lb = base.ln();
                    let ret1 = -argument.ln() / (base * (lb * lb));
                    tiny_vec!([f64ad; 2] => f64ad::f64(ret0), f64ad::f64(ret1))
                }
            }
            NodeType::Sqrt => {
                return if add_to_computation_graph {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let id = computation_graph.id;
                    let mode = computation_graph.mode.clone();
                    let node_idx_0 = parent_node_0.node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph as *mut ComputationGraph));
                    let ret = 1.0/(2.0*f0.sqrt());
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(1.0/(2.0*v.sqrt())))
                }
            }
            NodeType::Exp => {
                return if add_to_computation_graph {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let id = computation_graph.id;
                    let mode = computation_graph.mode.clone();
                    let node_idx_0 = parent_node_0.node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph as *mut ComputationGraph));
                    let ret = f0.exp();
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(v.exp()))
                }
            }
            NodeType::PowOneParentArgument => {
                return if add_to_computation_graph {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let id = computation_graph.id;
                    let mode = computation_graph.mode.clone();
                    let node_idx_0 = parent_node_0.node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph as *mut ComputationGraph));
                    let ret = constant_operands[0] * f0.powf(f64ad::f64(constant_operands[0] - 1.0));
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    tiny_vec!([f64ad; 2] => f64ad::f64(constant_operands[0] * v.powf(constant_operands[0] - 1.0)))
                }
            }
            NodeType::PowOneParentExponent => {
                return if add_to_computation_graph {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let id = computation_graph.id;
                    let mode = computation_graph.mode.clone();
                    let node_idx_0 = parent_node_0.node_idx;
                    let f0 = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph as *mut ComputationGraph));
                    let c = constant_operands[0];
                    let ret = f64ad::f64(c).powf(f0) * c.ln();
                    tiny_vec!([f64ad; 2] => ret)
                } else {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let v = parent_node_0.value;
                    let c = constant_operands[0];
                    tiny_vec!([f64ad; 2] => f64ad::f64(c.powf(v) * c.ln()))
                }
            }
            NodeType::PowTwoParents => {
                return if add_to_computation_graph {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let parent_node_1 = &computation_graph.nodes[parent_nodes[1] as usize];
                    let id = computation_graph.id;
                    let mode = computation_graph.mode.clone();
                    let node_idx_0 = parent_node_0.node_idx;
                    let node_idx_1 = parent_node_1.node_idx;
                    let argument = f64ad::f64ad_var(f64ad_var::new(id, node_idx_0, mode, computation_graph as *mut ComputationGraph));
                    let exponent = f64ad::f64ad_var(f64ad_var::new(id, node_idx_1, mode, computation_graph as *mut ComputationGraph));
                    let ret0 = exponent * argument.powf(exponent - 1.0);
                    let ret1 = argument.powf(exponent) * argument.ln();
                    tiny_vec!([f64ad; 2] => ret0, ret1)
                } else {
                    let parent_node_0 = &computation_graph.nodes[parent_nodes[0] as usize];
                    let parent_node_1 = &computation_graph.nodes[parent_nodes[1] as usize];
                    let argument = parent_node_0.value;
                    let exponent = parent_node_1.value;
                    let ret0 = exponent * argument.powf(exponent - 1.0);
                    let ret1 = argument.powf(exponent) * argument.ln();
                    tiny_vec!([f64ad; 2] => f64ad::f64(ret0), f64ad::f64(ret1))
                }
            }
        }
    }
}

// Addition ////////////////////////////////////////////////////////////////////////////////////////

impl Add for f64ad_var {
    type Output = f64ad_var;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.computation_graph_id, rhs.computation_graph_id);
        let res = unsafe {
            (*self.computation_graph.0).add_node(NodeType::AdditionTwoParents, tiny_vec!([u32; 2] => self.node_idx, rhs.node_idx), tiny_vec!([f64; 1]))
        };
        return res;
    }
}
impl Add<f64> for f64ad_var {
    type Output = f64ad_var;

    fn add(self, rhs: f64) -> Self::Output {
        let res = unsafe {
            (*self.computation_graph.0).add_node(NodeType::AdditionOneParent, tiny_vec!([u32; 2] => self.node_idx), tiny_vec!([f64; 1] => rhs))
        };
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
        let res = unsafe {
            (*self.computation_graph.0).add_node(NodeType::MultiplicationTwoParents, tiny_vec!([u32; 2] => self.node_idx, rhs.node_idx), tiny_vec!([f64; 1]))
        };
        return res;
    }
}
impl Mul<f64> for f64ad_var {
    type Output = f64ad_var;

    fn mul(self, rhs: f64) -> Self::Output {
        let res = unsafe {
            (*self.computation_graph.0).add_node(NodeType::MultiplicationOneParent, tiny_vec!([u32; 2] => self.node_idx), tiny_vec!([f64; 1] => rhs))
        };
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
        let res = unsafe {
            (*self.computation_graph.0).add_node(NodeType::SubtractionTwoParents, tiny_vec!([u32; 2] => self.node_idx, rhs.node_idx), tiny_vec!([f64; 1]))
        };
        return res;
    }
}
impl Sub<f64> for f64ad_var {
    type Output = f64ad_var;

    fn sub(self, rhs: f64) -> Self::Output {
        let res = unsafe {
            (*self.computation_graph.0).add_node(NodeType::SubtractionOneParentLeft, tiny_vec!([u32; 2] => self.node_idx), tiny_vec!([f64; 1] => rhs))
        };
        return res;
    }
}
impl Sub<f64ad_var> for f64 {
    type Output = f64ad_var;

    fn sub(self, rhs: f64ad_var) -> Self::Output {
        let res = unsafe {
            (*rhs.computation_graph.0).add_node(NodeType::SubtractionOneParentRight, tiny_vec!([u32; 2] => rhs.node_idx), tiny_vec!([f64; 1] => self))
        };
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
        *self = *self -rhs;
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
        let res = unsafe {
            (*self.computation_graph.0).add_node(NodeType::DivisionTwoParents, tiny_vec!([u32; 2] => self.node_idx, rhs.node_idx), tiny_vec!([f64; 1]))
        };
        return res;
    }
}
impl Div<f64> for f64ad_var {
    type Output = f64ad_var;

    fn div(self, rhs: f64) -> Self::Output {
        let res = unsafe {
            (*self.computation_graph.0).add_node(NodeType::DivisionOneParentNumerator, tiny_vec!([u32; 2] => self.node_idx), tiny_vec!([f64; 1] => rhs))
        };
        return res;
    }
}
impl Div<f64ad_var> for f64 {
    type Output = f64ad_var;

    fn div(self, rhs: f64ad_var) -> Self::Output {
        let res = unsafe {
            (*rhs.computation_graph.0).add_node(NodeType::DivisionOneParentDenominator, tiny_vec!([u32; 2] => rhs.node_idx), tiny_vec!([f64; 1] => self))
        };
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
        let res = unsafe {
            (*self.computation_graph.0).add_node(NodeType::Neg, tiny_vec!([u32; 2] => self.node_idx), tiny_vec!([f64; 1]))
        };
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
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl PartialEq for f64ad_var {
    fn eq(&self, other: &Self) -> bool {
        assert_eq!(self.computation_graph_id, other.computation_graph_id);
        unsafe {
            match self.mode {
                ComputationGraphMode::Standard => { self.value() == other.value() }
                ComputationGraphMode::Lock => { panic!("cannot compute PartialEq on ComputationGraphMode of Lock.  computation graph: {}", (*self.computation_graph.0).name) }
            }
        }
    }
}
impl PartialEq for f64ad_locked_var {
    fn eq(&self, other: &Self) -> bool {
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
                    return Some(Ordering::Equal)
                } else if v1 > v2 {
                    return Some(Ordering::Greater)
                } else if v1 < v2 {
                    return Some(Ordering::Less)
                } else {
                    unreachable!();
                }
            }
            ComputationGraphMode::Lock => {
                unsafe {
                    panic!("cannot partial_cmp f64ad_var in mode Lock.  Computation graph {}", (*self.computation_graph.0).name)
                }
            }
        }
    }
}
impl PartialOrd for f64ad_locked_var {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        panic!("cannot partial_cmp f64ad_locked_var");
    }
}
impl PartialOrd for f64ad {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (f64ad::f64ad_var(v1), f64ad::f64ad_var(v2)) => { v1.partial_cmp(v2) }
            (f64ad::f64ad_locked_var(v1), f64ad::f64ad_locked_var(v2)) => { v1.partial_cmp(v2) }
            (f64ad::f64(v1), f64ad::f64(v2)) => { v1.partial_cmp(v2) }
            _ => { panic!("unsupported.") }
        }
    }
}

impl Display for f64ad {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}
