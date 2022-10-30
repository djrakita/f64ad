use rand::prelude::*;
use tinyvec::*;

#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub enum f64ad {
    f64ad_var(f64ad_var),
    f64(f64)
}
impl Default for f64ad {
    fn default() -> Self {
        Self::f64(0.0)
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub struct f64ad_var {
    computation_graph_id: usize,
    node_idx: u32,
    computation_graph: *mut ComputationGraph
}
impl f64ad_var {
    pub (crate) fn new(computation_graph_id: usize, node_idx: u32, computation_graph: *mut ComputationGraph) -> Self {
        Self {
            computation_graph_id,
            node_idx,
            computation_graph
        }
    }
}

#[derive(Clone, Debug)]
pub struct ComputationGraph {
    id: usize,
    nodes: Vec<F64ADNode>,
    curr_idx: usize,
    capacity: usize
}
impl ComputationGraph {
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        let id: usize = rng.gen();
        Self {
            id,
            nodes: vec![],
            curr_idx: 0,
            capacity: 0
        }
    }
    pub fn add_node(&mut self, node_type: NodeType, parent_nodes: TinyVec<[u32; 2]>, constant_operands: TinyVec<[f64; 1]>) -> f64ad_var {
        let value = node_type.compute_value(&parent_nodes, &constant_operands, self);
        let node_idx = self.nodes.len() as u32;
        for parent_node in &parent_nodes { self.nodes[*parent_node as usize].child_nodes.push(node_idx); }
        let node = F64ADNode::new(node_idx, value, node_type, constant_operands, parent_nodes);
        if self.curr_idx >= self.capacity {
            self.nodes.push(node);
            self.capacity += 1;
        } else {
            self.nodes[self.curr_idx] = node;
        }
        self.curr_idx += 1;

        f64ad_var {
            computation_graph_id: self.id,
            node_idx: node_idx,
            computation_graph: self as *mut ComputationGraph
        }
    }
    pub fn add_node2(&mut self, node_type: NodeType, parent_nodes: TinyVec<[u32; 2]>, constant_operands: TinyVec<[f64; 1]>) -> f64ad_var {
        let value = node_type.compute_value(&parent_nodes, &constant_operands, self);
        let node_idx = self.nodes.len() as u32;
        for parent_node in &parent_nodes { self.nodes[*parent_node as usize].child_nodes.push(node_idx); }

        if self.curr_idx < self.capacity {
            let node = &mut self.nodes[self.curr_idx];
            // node.node_idx = node_idx;
            node.value = value;
            node.parent_nodes = parent_nodes;
            node.child_nodes.clear();
            node.constant_operands = constant_operands;
        } else {
            let node = F64ADNode::new(node_idx, value, node_type, constant_operands, parent_nodes);
            self.nodes.push(node);
            self.capacity += 1;
        }
        self.curr_idx += 1;

        f64ad_var {
            computation_graph_id: self.id,
            node_idx: node_idx,
            computation_graph: self as *mut ComputationGraph
        }
    }
    pub fn reset(&mut self) {
        self.curr_idx = 0;
    }
}

#[derive(Clone, Debug)]
pub struct F64ADNode {
    node_idx: u32,
    value: f64,
    node_type: NodeType,
    constant_operands: TinyVec<[f64; 1]>,
    parent_nodes: TinyVec<[u32; 2]>,
    child_nodes: TinyVec<[u32; 4]>
}
impl F64ADNode {
    pub fn new(node_idx: u32, value: f64, node_type: NodeType, constant_operands: TinyVec<[f64; 1]>, parent_nodes: TinyVec<[u32; 2]>) -> Self {
        Self {
            node_idx,
            value,
            node_type,
            constant_operands,
            parent_nodes,
            child_nodes: tiny_vec!([u32; 4])
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
    MultiplicationTwoParents
}
impl NodeType {
    pub fn compute_value(&self, parent_nodes: &TinyVec<[u32; 2]>, constant_operands: &TinyVec<[f64; 1]>, computation_graph: &ComputationGraph) -> f64 {
        0.0
    }
    pub fn compute_derivatives_of_value_wrt_parent_values(&self, parent_nodes: &TinyVec<[u32; 2]>, constant_operands: &TinyVec<[f64; 1]>, computation_graph: &mut ComputationGraph, add_to_computation_graph: bool) -> TinyVec<[f64ad; 2]> {
        todo!()
    }
}