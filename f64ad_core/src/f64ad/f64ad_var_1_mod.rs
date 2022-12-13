// Quick first derivatives

use std::cell::RefCell;
use std::fmt::{Debug, Formatter};
use rand::{Rng, thread_rng};
use crate::f64ad::{ComputationGraph, f64ad, GenericComputationGraph, NodeOperandsMode, NodeTypeClass};

#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub struct f64ad_var_1 {
    computation_graph_id: usize,
    node_idx: usize,
    computation_graph: &'static ComputationGraph
}
impl f64ad_var_1 {
    #[allow(dead_code)]
    pub (crate) fn new(computation_graph_id: usize, node_idx: usize, computation_graph: &'static ComputationGraph) -> Self {
        Self {
            computation_graph_id,
            node_idx,
            computation_graph
        }
    }
    #[inline(always)]
    pub fn value(&self) -> f64 {
        self.computation_graph.get_node_value(self.node_idx)
    }
    #[inline(always)]
    pub fn computation_graph_id(&self) -> usize {
        self.computation_graph_id
    }
    #[inline(always)]
    pub fn node_idx(&self) -> usize {
        self.node_idx
    }
    #[inline(always)]
    pub fn computation_graph(&self) -> &'static ComputationGraph {
        self.computation_graph
    }
}
impl Debug for f64ad_var_1 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("f64ad_var_1{ ").expect("error");
        f.write_str(&format!("value: {:?}, ", self.value())).expect("error");
        f.write_str(&format!("node_idx: {:?}", self.node_idx)).expect("error");
        f.write_str(" }").expect("error");

        Ok(())
    }
}

pub struct ComputationGraph1 {
    computation_graph_id: usize,
    generic_computation_graph: RefCell<GenericComputationGraph<F64ADNode1>>,
    pub (crate) paused: bool
}
impl ComputationGraph1 {
    pub (crate) fn new() -> Self {
        let mut rng = thread_rng();
        let id: usize = rng.gen();
        Self {
            computation_graph_id: id,
            generic_computation_graph: RefCell::new(GenericComputationGraph::new()),
            paused: false
        }
    }
    #[inline(always)]
    pub fn add_node(&self, value: f64, node_type_class: NodeTypeClass, node_operands_mode: NodeOperandsMode, parent_0: Option<f64ad>, parent_1: Option<f64ad>, computation_graph: &'static ComputationGraph) -> f64ad {
        let node_idx = self.generic_computation_graph.borrow().curr_idx;
        self.generic_computation_graph.borrow_mut().push(F64ADNode1 {
            node_idx,
            node_type_class,
            node_operands_mode,
            value,
            parent_0,
            parent_1
        });
        let ret = f64ad::f64ad_var_1(f64ad_var_1 {
            computation_graph_id: self.computation_graph_id,
            node_idx,
            computation_graph
        });
        ret
    }
    #[inline(always)]
    pub fn computation_graph_id(&self) -> usize {
        self.computation_graph_id
    }
    #[inline(always)]
    pub fn computation_graph(&self) -> &RefCell<GenericComputationGraph<F64ADNode1>> {
        &self.generic_computation_graph
    }
    #[inline(always)]
    pub fn num_nodes(&self) -> usize {
        self.generic_computation_graph.borrow().curr_idx()
    }
    pub fn soft_reset(&mut self) {
        let mut rng = thread_rng();
        let id: usize = rng.gen();
        self.computation_graph_id = id;
        self.generic_computation_graph.borrow_mut().reset();
    }
    pub fn hard_reset(&mut self) {
        *self = Self::new();
    }
    #[inline(always)]
    pub (crate) fn paused(&self) -> bool { self.paused }
}

pub struct F64ADNode1 {
    node_idx: usize,
    node_type_class: NodeTypeClass,
    node_operands_mode: NodeOperandsMode,
    value: f64,
    parent_0: Option<f64ad>,
    parent_1: Option<f64ad>
}
impl F64ADNode1 {
    #[inline(always)]
    pub fn node_idx(&self) -> usize {
        self.node_idx
    }
    #[inline(always)]
    pub fn value(&self) -> f64 {
        self.value
    }
    #[inline(always)]
    pub fn parent_0(&self) -> &Option<f64ad> {
        &self.parent_0
    }
    #[inline(always)]
    pub fn parent_1(&self) -> &Option<f64ad> {
        &self.parent_1
    }
    #[inline(always)]
    pub fn node_type_class(&self) -> NodeTypeClass {
        self.node_type_class
    }
    #[inline(always)]
    pub fn node_operands_mode(&self) -> NodeOperandsMode {
        self.node_operands_mode
    }
}