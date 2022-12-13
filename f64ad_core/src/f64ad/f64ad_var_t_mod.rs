// Tracer

use std::cell::RefCell;
use std::fmt::{Debug, Formatter};
use rand::{Rng, thread_rng};
use crate::f64ad::{ComputationGraph, f64ad, NodeOperandsMode, NodeTypeClass};

#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub struct f64ad_var_t {
    computation_graph_id: usize,
    node_idx: usize,
    computation_graph: &'static ComputationGraph
}
impl f64ad_var_t {
    pub fn new(computation_graph_id: usize, node_idx: usize, computation_graph: &'static ComputationGraph) -> Self {
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
impl Debug for f64ad_var_t {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("f64ad_var_1{ ").expect("error");
        f.write_str(&format!("value: {:?}, ", self.value())).expect("error");
        f.write_str(&format!("node_idx: {:?}", self.node_idx)).expect("error");
        f.write_str(" }").expect("error");

        Ok(())
    }
}

pub struct ComputationGraphT {
    computation_graph_id: usize,
    computation_graph: RefCell<Vec<F64ADNodeT>>
}
impl ComputationGraphT {
    pub (crate) fn new() -> Self {
        let mut rng = thread_rng();
        let id: usize = rng.gen();
        Self {
            computation_graph_id: id,
            computation_graph: RefCell::new(Vec::new())
        }
    }

    #[inline(always)]
    pub fn add_node(&self, value: f64, node_type_class: NodeTypeClass, node_operands_mode: NodeOperandsMode, parent_0: Option<f64ad>, parent_1: Option<f64ad>, computation_graph: &'static ComputationGraph) -> f64ad {
        let node_idx = self.computation_graph.borrow().len();
        self.computation_graph.borrow_mut().push(F64ADNodeT {
            node_idx,
            node_type_class,
            node_operands_mode,
            value,
            parent_0,
            parent_1
        });
        let ret = f64ad::f64ad_var_t(f64ad_var_t {
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
    pub fn num_nodes(&self) -> usize {
        self.computation_graph.borrow().len()
    }
    pub fn reset(&mut self) {
        *self = Self::new();
    }
    pub fn computation_graph(&self) -> &RefCell<Vec<F64ADNodeT>> {
        &self.computation_graph
    }
}

pub struct F64ADNodeT {
    node_idx: usize,
    node_type_class: NodeTypeClass,
    node_operands_mode: NodeOperandsMode,
    value: f64,
    parent_0: Option<f64ad>,
    parent_1: Option<f64ad>
}
impl F64ADNodeT {
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