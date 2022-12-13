// Locked

use std::cell::RefCell;
use std::fmt::{Debug, Formatter};
use crate::f64ad::{ComputationGraph, f64ad, NodeOperandsMode, NodeTypeClass};

#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub struct f64ad_var_l {
    computation_graph_id: usize,
    node_idx: usize,
    computation_graph: &'static ComputationGraph
}
impl f64ad_var_l {
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
impl Debug for f64ad_var_l {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("f64ad_var_1{ ").expect("error");
        f.write_str(&format!("value: {:?}, ", self.value())).expect("error");
        f.write_str(&format!("node_idx: {:?}", self.node_idx)).expect("error");
        f.write_str(" }").expect("error");

        Ok(())
    }
}

pub struct ComputationGraphL {
    pub (crate) computation_graph_id: usize,
    pub (crate) locked_nodes: RefCell<Vec<F64ADNodeL>>,
    pub (crate) count: RefCell<usize>
}
impl ComputationGraphL {
    #[inline(always)]
    pub fn add_node(&self, value: f64, node_type_class: NodeTypeClass, node_operands_mode: NodeOperandsMode, parent_0: Option<f64ad>, parent_1: Option<f64ad>, computation_graph: &'static ComputationGraph) -> f64ad {
        let idx = self.count.borrow().clone();

        assert_eq!(self.locked_nodes.borrow()[idx].node_type_class, node_type_class, "looks like this computation could not be locked!");
        assert_eq!(self.locked_nodes.borrow()[idx].node_operands_mode, node_operands_mode, "looks like this computation could not be locked!");

        let mut b = self.locked_nodes.borrow_mut();

        b[idx].value = value;
        b[idx].parent_0 = parent_0.clone();
        b[idx].parent_1 = parent_1.clone();

        *self.count.borrow_mut() += 1;

        f64ad::f64ad_var_l(f64ad_var_l {
            computation_graph_id: self.computation_graph_id,
            node_idx: idx,
            computation_graph
        })
    }
    #[inline(always)]
    pub fn computation_graph_id(&self) -> usize {
        self.computation_graph_id
    }
    #[inline(always)]
    pub fn num_nodes(&self) -> usize {
        self.locked_nodes.borrow().len()
    }
    #[inline(always)]
    pub fn locked_nodes(&self) -> &RefCell<Vec<F64ADNodeL>> {
        &self.locked_nodes
    }
    pub fn reset(&mut self) {
        *self.count.borrow_mut() = 0;
    }
}

pub struct F64ADNodeL {
    pub (crate) node_idx: usize,
    pub (crate) node_type_class: NodeTypeClass,
    pub (crate) node_operands_mode: NodeOperandsMode,
    pub (crate) value: f64,
    pub (crate) parent_0: Option<f64ad>,
    pub (crate) parent_1: Option<f64ad>
}
impl F64ADNodeL {
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