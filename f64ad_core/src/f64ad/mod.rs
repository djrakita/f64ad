use std::cell::{RefCell};
use std::fmt::{Debug, Formatter};
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};
use once_cell::sync::OnceCell;
pub mod trait_impls;

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum f64ad {
    f64ad_var(f64ad_var),
    f64(f64)
}
impl f64ad {
    pub fn value(&self) -> f64 {
        match self {
            f64ad::f64ad_var(v) => { v.value() }
            f64ad::f64(v) => { *v }
        }
    }
    pub fn unwrap_f64ad_var(&self) -> f64ad_var {
        match self {
            f64ad::f64ad_var(v) => { *v }
            _ => { panic!("wrong type.") }
        }
    }
    pub fn unwrap_f64(&self) -> f64 {
        match self {
            f64ad::f64(v) => { *v }
            _ => { panic!("wrong type.") }
        }
    }
    pub fn backwards_mode_grad(&self) -> BackwardsModeGradOutput {
        match self {
            f64ad::f64ad_var(v) => { v.backwards_mode_grad() }
            f64ad::f64(_) => { panic!("cannot compute grad on f64.") }
        }
    }
    pub fn forward_mode_grad(&self) -> ForwardModeGradOutput {
        match self {
            f64ad::f64ad_var(v) => { v.forward_mode_grad() }
            f64ad::f64(_) => { panic!("cannot compute grad on f64.") }
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, PartialEq)]
pub struct f64ad_var {
    pub node_idx: usize,
    reset_idx: usize,
    tape: &'static Tape
}
impl f64ad_var {
    pub fn new_from_node_idx(node_idx: usize, tape: &'static Tape) -> Self {
        Self {
            node_idx,
            reset_idx: tape.reset_idx,
            tape
        }
    }
    pub fn value(&self) -> f64 {
        assert_eq!(self.tape.reset_idx, self.reset_idx);

        return self.tape.nodes.borrow()[self.node_idx].value
    }
    pub fn backwards_mode_grad(&self) -> BackwardsModeGradOutput {
        let n = self.tape.len();
        let mut derivs = vec![f64ad::f64(0.0); n];
        derivs[self.node_idx] = f64ad::f64(1.0);
        let mut already_in_dependencies = vec![false; n];
        already_in_dependencies[self.node_idx] = true;

        let mut stack = vec![self.node_idx];
        while !stack.is_empty() {
            let curr_idx = stack.pop().unwrap();
            let curr_node = &self.tape.nodes.borrow()[curr_idx];
            let parent_node_idxs = &curr_node.parent_node_idxs;
            for parent_node_idx in parent_node_idxs {
                if !already_in_dependencies[*parent_node_idx] {
                    stack.push(*parent_node_idx);
                    already_in_dependencies[*parent_node_idx] = true;
                }
            }
        }

        for (dependency_node_idx, b) in already_in_dependencies.iter().enumerate().rev() {
            if *b {
                let node = self.tape.nodes.borrow()[dependency_node_idx].clone();
                let is_deferred = node.derivatives_mode.is_deferred();
                if is_deferred {
                    match &node.derivatives_mode {
                        DerivativesMode::Deferred(f) => {
                            let res = f.0.compute(node.parent_node_idxs.clone(), self.tape);
                            self.tape.nodes.borrow_mut()[dependency_node_idx].derivatives_mode = DerivativesMode::Completed(res);
                        }
                        DerivativesMode::Completed(_) => { unreachable!() }
                    }
                }
            }
        }

        for (dependency_node_idx, b) in already_in_dependencies.iter().enumerate().rev() {
            if *b {
                let node = self.tape.nodes.borrow()[dependency_node_idx].clone();
                let parent_node_idxs = node.parent_node_idxs.clone();
                let dependency_node = derivs[dependency_node_idx].clone();
                match &node.derivatives_mode {
                    DerivativesMode::Completed(derivatives) => {
                        assert_eq!(derivatives.len(), parent_node_idxs.len());
                        for (i, parent_node_idx) in parent_node_idxs.iter().enumerate() {
                            if !(derivatives[i].value() == 0.0 || dependency_node.value() == 0.0) {
                                derivs[*parent_node_idx] += f64ad::f64ad_var(derivatives[i]) * dependency_node;
                            }
                        }
                    }
                    _ => { unreachable!() }
                }
            }
        }
        
        BackwardsModeGradOutput {
            derivs
        }
    }
    pub fn forward_mode_grad(&self) -> ForwardModeGradOutput {
        let n = self.tape.len();
        let mut derivs = vec![f64ad::f64(0.0); n];
        derivs[self.node_idx] = f64ad::f64(1.0);

        let mut already_in_successors = vec![false; n];
        already_in_successors[self.node_idx] = true;

        let mut stack = vec![self.node_idx];
        while !stack.is_empty() {
            let curr_idx = stack.pop().unwrap();
            let curr_node = &self.tape.nodes.borrow()[curr_idx];
            let mut child_node_idxs = &curr_node.child_node_idxs;
            for child_node_idx in child_node_idxs {
                if !already_in_successors[*child_node_idx] {
                    // successor_node_idxs.push(*child_node_idx);
                    /*
                    let binary_search_res = successor_node_idxs.binary_search_by(|x| x.partial_cmp(child_node_idx).unwrap());
                    match binary_search_res {
                        Err(idx) => { successor_node_idxs.insert(idx, *child_node_idx) }
                        _ => { unreachable!() }
                    }
                    */
                    stack.push(*child_node_idx);
                    already_in_successors[*child_node_idx] = true;
                }
            }
        }

        for (successor_node_idx, b) in already_in_successors.iter().enumerate() {
            if *b {
                let node = self.tape.nodes.borrow()[successor_node_idx].clone();
                let is_deferred = node.derivatives_mode.is_deferred();
                if is_deferred {
                    match &node.derivatives_mode {
                        DerivativesMode::Deferred(f) => {
                            let res = f.0.compute(node.parent_node_idxs.clone(), self.tape);
                            self.tape.nodes.borrow_mut()[successor_node_idx].derivatives_mode = DerivativesMode::Completed(res);
                        }
                        DerivativesMode::Completed(_) => { unreachable!() }
                    }
                }
            }
        }

        for (successor_node_idx, b) in already_in_successors.iter().enumerate() {
            if *b {
                let node = self.tape.nodes.borrow()[successor_node_idx].clone();
                match &node.derivatives_mode {
                    DerivativesMode::Completed(derivatives) => {
                        let parent_node_idxs = node.parent_node_idxs;
                        assert_eq!(parent_node_idxs.len(), derivatives.len());
                        for (i, parent_node_idx) in parent_node_idxs.iter().enumerate() {
                            let parent_node = self.tape.nodes.borrow()[*parent_node_idx].clone();
                            // if derivs[parent_node.node_idx].is_some() {
                                let dependency_node = derivs[*parent_node_idx].clone();
                                derivs[successor_node_idx] += f64ad::f64ad_var(derivatives[i]) * dependency_node;
                            // }
                        }
                    }
                    _ => { unreachable!() }
                }
            }
        }

        ForwardModeGradOutput { derivs }
    }
}
impl Debug for f64ad_var {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("f64ad: { ")?;
        f.write_str(&format!("value: {}", self.value()))?;
        f.write_str(" }")?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct BackwardsModeGradOutput {
    derivs: Vec<f64ad>
}
impl BackwardsModeGradOutput {
    pub fn wrt(&self, input: &f64ad) -> f64ad {
        match input {
            f64ad::f64ad_var(input) => {
                assert_eq!(input.tape.reset_idx, input.reset_idx);
                let idx = input.node_idx;
                return self.derivs[idx];
            }
            f64ad::f64(_) => {
                panic!("cannot call wrt on f64.")
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct ForwardModeGradOutput {
    derivs: Vec<f64ad>
}
impl ForwardModeGradOutput {
    pub fn wrt(&self, output: &f64ad) -> f64ad {
        match output {
            f64ad::f64ad_var(output) => {
                assert_eq!(output.tape.reset_idx, output.reset_idx);
                let idx = output.node_idx;
                return self.derivs[idx];
            }
            f64ad::f64(_) => {
                panic!("cannot call wrt on f64.")
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct F64ADNode {
    value: f64,
    node_idx: usize,
    parent_node_idxs: Vec<usize>,
    child_node_idxs: Vec<usize>,
    derivatives_mode: DerivativesMode
}

#[derive(Debug, Clone)]
pub enum DerivativesMode {
    Deferred(DerivativeFunctionBox),
    Completed(Vec<f64ad_var>)
}
impl DerivativesMode {
    fn is_deferred(&self) -> bool {
        match self {
            DerivativesMode::Deferred(_) => { true }
            _ => { false }
        }
    }
    pub fn new_deferred<F: DerivativeFunction + 'static>(f: F) -> Self {
        DerivativesMode::Deferred(DerivativeFunctionBox::new(f))
    }
}

pub trait DerivativeFunction: Debug + DerivativeFunctionClone {
    fn compute(&self, operand_node_idxs: Vec<usize>, tape: &'static Tape) -> Vec<f64ad_var>;
}
pub trait DerivativeFunctionClone {
    fn clone_box(&self) -> Box<dyn DerivativeFunction>;
}
impl<T> DerivativeFunctionClone for T where T: 'static + DerivativeFunction + Clone {
    fn clone_box(&self) -> Box<dyn DerivativeFunction> {
        Box::new(self.clone())
    }
}

#[derive(Debug)]
pub struct DerivativeFunctionBox(Box<dyn DerivativeFunction>);
impl DerivativeFunctionBox {
    pub fn new<F: DerivativeFunction + 'static>(f: F) -> Self {
        Self(Box::new(f))
    }
}
impl Clone for DerivativeFunctionBox {
    fn clone(&self) -> Self {
        return Self(self.0.clone_box());
    }
}

#[derive(Clone, Debug)]
pub struct ConstantDerivativeFunction;
impl DerivativeFunction for ConstantDerivativeFunction {
    fn compute(&self, operand_node_idxs: Vec<usize>, tape: &'static Tape) -> Vec<f64ad_var> {
        let mut out_vec = vec![];
        for _ in &operand_node_idxs {
            out_vec.push(tape.add_node(0.0, operand_node_idxs.clone(), DerivativesMode::Deferred(DerivativeFunctionBox(Box::new(ConstantDerivativeFunction)))))
        }

        out_vec
    }
}

#[derive(Debug, Clone)]
pub struct Tape {
    nodes: RefCell<Vec<F64ADNode>>,
    thread_idx: usize,
    reset_idx: usize
}
impl Tape {
    fn new(reset_idx: usize, thread_idx: usize) -> Self {
        Self {
            nodes: RefCell::new(vec![]),
            thread_idx,
            reset_idx,
        }
    }
    pub fn add_variable(&'static self, value: f64) -> f64ad_var {
        let l = self.len();
        self.nodes.borrow_mut().push(F64ADNode {
            value,
            node_idx: l,
            parent_node_idxs: vec![],
            child_node_idxs: vec![],
            derivatives_mode: DerivativesMode::Completed(vec![])
        });
        f64ad_var {
            node_idx: l,
            reset_idx: self.reset_idx,
            tape: &self
        }
    }
    fn add_node(&'static self, value: f64, parent_node_idxs: Vec<usize>, derivatives: DerivativesMode) -> f64ad_var {
        let node_idx = self.nodes.borrow().len();
        for parent_node_idx in &parent_node_idxs {
            self.nodes.borrow_mut()[*parent_node_idx].child_node_idxs.push(node_idx);
        }
        self.nodes.borrow_mut().push(F64ADNode {
            value,
            node_idx,
            parent_node_idxs,
            child_node_idxs: vec![],
            derivatives_mode: derivatives
        });
        return f64ad_var {
            node_idx,
            reset_idx: self.reset_idx,
            tape: self
        };
    }
    pub fn len(&self) -> usize {
        self.nodes.borrow().len()
    }
}
impl PartialEq for Tape {
    fn eq(&self, other: &Self) -> bool {
        return if self.thread_idx == other.thread_idx && self.reset_idx == other.reset_idx { true } else { false }
    }
}

static mut _GLOBAL_TAPE: OnceCell<Tape> = OnceCell::new();
pub struct GlobalTape;
impl GlobalTape {
    pub fn get() -> &'static Tape {
        return unsafe { _GLOBAL_TAPE.get_or_init(|| Tape::new(0, 0)) } ; }
    pub fn reset() {
        unsafe {
            let tape = _GLOBAL_TAPE.take().unwrap();
            let reset_idx = tape.reset_idx;
            _GLOBAL_TAPE.set(Tape::new(reset_idx+1, 0)).unwrap();
        }
    }
}

static mut _GLOBAL_TAPE_PARALLEL: OnceCell<Vec<Tape>> = OnceCell::new();
pub struct GlobalTapeParallel;
impl GlobalTapeParallel {
    pub fn get(thread_idx: usize) -> &'static Tape {
        let vec = unsafe {
            _GLOBAL_TAPE_PARALLEL.get_or_init(|| {
                let n = num_cpus::get();
                let mut v = vec![];

                for thread_idx in 0..n { v.push(Tape::new(0, thread_idx)); }

                v
            }
            )
        };

        return vec.get(thread_idx).unwrap();
    }
    pub fn reset_all_threads() {
        unsafe {
            let tapes = _GLOBAL_TAPE_PARALLEL.take().unwrap();
            let reset_idx = tapes[0].reset_idx;

            let n = num_cpus::get();
            let mut v = vec![];

            for thread_idx in 0..n { v.push(Tape::new(reset_idx+1, thread_idx)); }

            _GLOBAL_TAPE_PARALLEL.set(v).expect("error"); }
    }
}

// ADDITION //

impl Add<f64ad_var> for f64ad_var {
    type Output = f64ad_var;

    fn add(self, rhs: f64ad_var) -> Self::Output {
        assert_eq!(self.tape, rhs.tape);

        let tape = self.tape;
        return tape.add_node(self.value() + rhs.value(), vec![self.node_idx, rhs.node_idx], DerivativesMode::new_deferred(Add2ParentsDerivative));
    }
}
#[derive(Debug, Clone)]
pub struct Add2ParentsDerivative;
impl DerivativeFunction for Add2ParentsDerivative {
    fn compute(&self, operand_node_idxs: Vec<usize>, tape: &'static Tape) -> Vec<f64ad_var> {
        assert_eq!(operand_node_idxs.len(), 2);

        let out1 = tape.add_node(1.0, operand_node_idxs.clone(), DerivativesMode::new_deferred(ConstantDerivativeFunction));
        let out2 = tape.add_node(1.0, operand_node_idxs.clone(), DerivativesMode::new_deferred(ConstantDerivativeFunction));

        let mut out = vec![
            out1, out2
        ];

        out
    }
}

impl Add<f64ad_var> for f64 {
    type Output = f64ad_var;

    fn add(self, rhs: f64ad_var) -> Self::Output {
        let tape = rhs.tape;
        tape.add_node(rhs.value() + self, vec![rhs.node_idx], DerivativesMode::new_deferred(Add1ParentDerivative))
    }
}
#[derive(Debug, Clone)]
pub struct Add1ParentDerivative;
impl DerivativeFunction for Add1ParentDerivative {
    fn compute(&self, operand_node_idxs: Vec<usize>, tape: &'static Tape) -> Vec<f64ad_var> {
        assert_eq!(operand_node_idxs.len(), 1);

        let out = tape.add_node(1.0, operand_node_idxs.clone(), DerivativesMode::new_deferred(ConstantDerivativeFunction));

        vec![out]
    }
}

impl Add<f64> for f64ad_var {
    type Output = f64ad_var;

    fn add(self, rhs: f64) -> Self::Output {
        return rhs + self;
    }
}

impl AddAssign<f64ad_var> for f64ad_var {
    fn add_assign(&mut self, rhs: f64ad_var) {
        *self = *self + rhs;
    }
}

impl AddAssign<f64> for f64ad_var {
    fn add_assign(&mut self, rhs: f64) {
        *self = *self + rhs;
    }
}

impl Add<f64ad> for f64ad {
    type Output = f64ad;

    fn add(self, rhs: f64ad) -> Self::Output {
        return match (&self, &rhs) {
            (f64ad::f64ad_var(f1), f64ad::f64ad_var(f2)) => {
                f64ad::f64ad_var(*f1 + *f2)
            }
            (f64ad::f64ad_var(f1), f64ad::f64(f2)) => {
                f64ad::f64ad_var(*f1 + *f2)
            }
            (f64ad::f64(f1), f64ad::f64ad_var(f2)) => {
                f64ad::f64ad_var(*f1 + *f2)
            }
            (f64ad::f64(f1), f64ad::f64(f2)) => {
                f64ad::f64(*f1 + *f2)
            }
        }
    }
}
impl AddAssign<f64ad> for f64ad {
    fn add_assign(&mut self, rhs: f64ad) {
        *self = *self + rhs;
    }
}

// SUBTRACTION //
impl Sub<f64ad_var> for f64ad_var {
    type Output = f64ad_var;

    fn sub(self, rhs: f64ad_var) -> Self::Output {
        assert_eq!(self.tape, rhs.tape);

        let tape = self.tape;
        return tape.add_node(self.value() - rhs.value(), vec![self.node_idx, rhs.node_idx], DerivativesMode::new_deferred(Sub2ParentsDerivative));
    }
}
#[derive(Debug, Clone)]
pub struct Sub2ParentsDerivative;
impl DerivativeFunction for Sub2ParentsDerivative {
    fn compute(&self, operand_node_idxs: Vec<usize>, tape: &'static Tape) -> Vec<f64ad_var> {
        assert_eq!(operand_node_idxs.len(), 2);

        let out1 = tape.add_node(1.0, operand_node_idxs.clone(), DerivativesMode::new_deferred(ConstantDerivativeFunction));
        let out2 = tape.add_node(-1.0, operand_node_idxs.clone(), DerivativesMode::new_deferred(ConstantDerivativeFunction));

        let mut out = vec![
            out1, out2
        ];

        out
    }
}

impl Sub<f64ad_var> for f64 {
    type Output = f64ad_var;

    fn sub(self, rhs: f64ad_var) -> Self::Output {
        let tape = rhs.tape;
        tape.add_node(self - rhs.value(), vec![rhs.node_idx], DerivativesMode::new_deferred(SubRight1ParentDerivative))
    }
}
#[derive(Debug, Clone)]
pub struct SubRight1ParentDerivative;
impl DerivativeFunction for SubRight1ParentDerivative {
    fn compute(&self, operand_node_idxs: Vec<usize>, tape: &'static Tape) -> Vec<f64ad_var> {
        assert_eq!(operand_node_idxs.len(), 1);

        let out = tape.add_node(-1.0, operand_node_idxs.clone(), DerivativesMode::new_deferred(ConstantDerivativeFunction));

        vec![out]
    }
}

impl Sub<f64> for f64ad_var {
    type Output = f64ad_var;

    fn sub(self, rhs: f64) -> Self::Output {
        let tape = self.tape;
        tape.add_node(self.value() - rhs, vec![self.node_idx], DerivativesMode::new_deferred(SubLeft1ParentDerivative))
    }
}
#[derive(Debug, Clone)]
pub struct SubLeft1ParentDerivative;
impl DerivativeFunction for SubLeft1ParentDerivative {
    fn compute(&self, operand_node_idxs: Vec<usize>, tape: &'static Tape) -> Vec<f64ad_var> {
        assert_eq!(operand_node_idxs.len(), 1);

        let out = tape.add_node(1.0, operand_node_idxs.clone(), DerivativesMode::new_deferred(ConstantDerivativeFunction));

        vec![out]
    }
}

impl SubAssign<f64ad_var> for f64ad_var {
    fn sub_assign(&mut self, rhs: f64ad_var) {
        *self = *self - rhs;
    }
}

impl SubAssign<f64> for f64ad_var {
    fn sub_assign(&mut self, rhs: f64) {
        *self = *self - rhs;
    }
}

// MULTIPLICATION //

impl Mul<f64ad_var> for f64ad_var {
    type Output = f64ad_var;

    fn mul(self, rhs: f64ad_var) -> Self::Output {
        assert_eq!(self.tape, rhs.tape);

        let tape = self.tape;
        tape.add_node(self.value() * rhs.value(), vec![self.node_idx, rhs.node_idx], DerivativesMode::new_deferred(Mul2ParentsDerivative))
    }
}
#[derive(Debug, Clone)]
pub struct Mul2ParentsDerivative;
impl DerivativeFunction for Mul2ParentsDerivative {
    fn compute(&self, operand_node_idxs: Vec<usize>, tape: &'static Tape) -> Vec<f64ad_var> {
        assert_eq!(operand_node_idxs.len(), 2);

        // let parent1 = tape.nodes.borrow()[operand_node_idxs[0]].clone();
        // let parent2 = tape.nodes.borrow()[operand_node_idxs[1]].clone();

        let out1 = f64ad_var::new_from_node_idx(operand_node_idxs[1], tape);
        let out2 = f64ad_var::new_from_node_idx(operand_node_idxs[0], tape);

        // let out1 = tape.add_node(parent2.value, operand_node_idxs.clone(), DerivativesMode::Deferred(DerivativeFunctionBox(Box::new(ConstantDerivativeFunction))));
        // let out2 = tape.add_node(parent1.value, operand_node_idxs.clone(), DerivativesMode::Deferred(DerivativeFunctionBox(Box::new(ConstantDerivativeFunction))));

        let mut out = vec![
            out1, out2
        ];

        out
    }
}

impl Mul<f64ad_var> for f64 {
    type Output = f64ad_var;

    fn mul(self, rhs: f64ad_var) -> Self::Output {
        let tape = rhs.tape;
        tape.add_node(rhs.value() * self, vec![rhs.node_idx], DerivativesMode::new_deferred(Mul1ParentDerivative { n: self }))
    }
}
#[derive(Debug, Clone)]
pub struct Mul1ParentDerivative {
    n: f64
}
impl DerivativeFunction for Mul1ParentDerivative {
    fn compute(&self, operand_node_idxs: Vec<usize>, tape: &'static Tape) -> Vec<f64ad_var> {
        assert_eq!(operand_node_idxs.len(), 1);

        let out = tape.add_node(self.n, operand_node_idxs.clone(), DerivativesMode::Deferred(DerivativeFunctionBox(Box::new(ConstantDerivativeFunction))));

        vec![out]
    }
}

impl Mul<f64> for f64ad_var {
    type Output = f64ad_var;

    fn mul(self, rhs: f64) -> Self::Output {
        return rhs * self;
    }
}

impl MulAssign<f64ad_var> for f64ad_var {
    fn mul_assign(&mut self, rhs: f64ad_var) {
        *self = *self * rhs;
    }
}

impl MulAssign<f64> for f64ad_var {
    fn mul_assign(&mut self, rhs: f64) {
        *self = *self * rhs;
    }
}

impl Mul<f64ad> for f64ad {
    type Output = f64ad;

    fn mul(self, rhs: f64ad) -> Self::Output {
        return match (&self, &rhs) {
            (f64ad::f64ad_var(f1), f64ad::f64ad_var(f2)) => {
                f64ad::f64ad_var(*f1 * *f2)
            }
            (f64ad::f64ad_var(f1), f64ad::f64(f2)) => {
                f64ad::f64ad_var(*f1 * *f2)
            }
            (f64ad::f64(f1), f64ad::f64ad_var(f2)) => {
                f64ad::f64ad_var(*f1 * *f2)
            }
            (f64ad::f64(f1), f64ad::f64(f2)) => {
                f64ad::f64(*f1 * *f2)
            }
        }
    }
}
impl MulAssign<f64ad> for f64ad {
    fn mul_assign(&mut self, rhs: f64ad) {
        match (self, &rhs) {
            (f64ad::f64ad_var(f1), f64ad::f64ad_var(f2)) => {
                *f1 = *f1 * *f2;
            }
            (f64ad::f64ad_var(f1), f64ad::f64(f2)) => {
                *f1 = *f1 * *f2;
            }
            (f64ad::f64(f1), f64ad::f64ad_var(f2)) => {
                *f1 = *f1 * f2.value();
            }
            (f64ad::f64(f1), f64ad::f64(f2)) => {
                *f1 = *f1 * *f2;
            }
        }
    }
}

// DIVISION //

impl Div<f64ad_var> for f64ad_var {
    type Output = f64ad_var;

    fn div(self, rhs: f64ad_var) -> Self::Output {
        assert_eq!(self.tape, rhs.tape);

        let tape = self.tape;

        if self.node_idx == rhs.node_idx {
            return tape.add_node(1.0, vec![self.node_idx, rhs.node_idx], DerivativesMode::new_deferred(ConstantDerivativeFunction));
        }

        tape.add_node(self.value() / rhs.value(), vec![self.node_idx, rhs.node_idx], DerivativesMode::new_deferred(Div2ParentsDerivative))
    }
}
#[derive(Clone, Debug)]
pub struct Div2ParentsDerivative;
impl DerivativeFunction for Div2ParentsDerivative {
    fn compute(&self, operand_node_idxs: Vec<usize>, tape: &'static Tape) -> Vec<f64ad_var> {
        assert_eq!(operand_node_idxs.len(), 2);

        let operand1_f64ad = f64ad_var {
            node_idx: operand_node_idxs[0],
            reset_idx: tape.reset_idx,
            tape
        };
        let operand2_f64ad = f64ad_var {
            node_idx: operand_node_idxs[1],
            reset_idx: tape.reset_idx,
            tape
        };

        let out1 = 1.0 / operand2_f64ad;
        let out2 = -1.0 * (operand1_f64ad / (operand2_f64ad.internal_powf(2.0)));

        return vec![out1, out2];
    }
}

impl Div<f64ad_var> for f64 {
    type Output = f64ad_var;

    fn div(self, rhs: f64ad_var) -> Self::Output {
        let tape = rhs.tape;
        tape.add_node(self / rhs.value(), vec![rhs.node_idx], DerivativesMode::new_deferred(DivDenominator1ParentDerivative {numerator: self}))
    }
}
#[derive(Clone, Debug)]
pub struct DivDenominator1ParentDerivative {
    numerator: f64
}
impl DerivativeFunction for DivDenominator1ParentDerivative {
    fn compute(&self, operand_node_idxs: Vec<usize>, tape: &'static Tape) -> Vec<f64ad_var> {
        assert_eq!(operand_node_idxs.len(), 1);

        let operand_f64ad = f64ad_var {
            node_idx: operand_node_idxs[0],
            reset_idx: tape.reset_idx,
            tape
        };

        let out = -self.numerator / (operand_f64ad.internal_powf(2.0));

        vec![out]
    }
}

impl Div<f64> for f64ad_var {
    type Output = f64ad_var;

    fn div(self, rhs: f64) -> Self::Output {
        let tape = self.tape;
        tape.add_node(self.value() / rhs, vec![self.node_idx], DerivativesMode::new_deferred(DivNumerator1ParentDerivative { denominator: rhs }))
    }
}
#[derive(Clone, Debug)]
pub struct DivNumerator1ParentDerivative {
    denominator: f64
}
impl DerivativeFunction for DivNumerator1ParentDerivative {
    fn compute(&self, operand_node_idxs: Vec<usize>, tape: &'static Tape) -> Vec<f64ad_var> {
        assert_eq!(operand_node_idxs.len(), 1);

        let out = tape.add_node(1.0 / self.denominator, operand_node_idxs.clone(), DerivativesMode::new_deferred(ConstantDerivativeFunction));

        vec![out]
    }
}

// Negation //

impl Neg for f64ad_var {
    type Output = f64ad_var;

    fn neg(self) -> Self::Output {
        return -1.0 * self;
    }
}

// POWF //

pub trait Powf<T> {
    type Output;
    /// Calculate `powf` for self, where `other` is the power to raise `self` to.
    fn internal_powf(&self, other: T) -> Self::Output;
}

impl Powf<f64ad_var> for f64ad_var {
    type Output = f64ad_var;

    fn internal_powf(&self, other: f64ad_var) -> Self::Output {
        assert_eq!(self.tape, other.tape);
        let tape = self.tape;

        todo!()
        // tape.add_node(self.value().powf(other.value()), vec![self.node_idx, other.node_idx], )
    }
}

impl Powf<f64> for f64ad_var {
    type Output = f64ad_var;

    fn internal_powf(&self, other: f64) -> Self::Output {
        let tape = self.tape;
        tape.add_node( self.value().powf(other), vec![self.node_idx], DerivativesMode::new_deferred(PowfBase1ParentDerivative {n: other}))
    }
}
#[derive(Debug, Clone)]
pub struct PowfBase1ParentDerivative {
    n: f64
}
impl DerivativeFunction for PowfBase1ParentDerivative {
    fn compute(&self, operand_node_idxs: Vec<usize>, tape: &'static Tape) -> Vec<f64ad_var> {
        assert_eq!(operand_node_idxs.len(), 1);

        let operand_f64ad = f64ad_var {
            node_idx: operand_node_idxs[0],
            reset_idx: tape.reset_idx,
            tape
        };

        let out = self.n * operand_f64ad.internal_powf(self.n - 1.0);

        vec![out]
    }
}

// Square root //

impl f64ad_var {
    pub fn internal_sqrt(&self) -> Self {
        return self.internal_powf(0.5);
    }
}

// Trig functions //

impl f64ad_var {
    pub fn internal_sin(&self) -> Self {
        let tape = self.tape;
        tape.add_node(self.value().sin(), vec![self.node_idx], DerivativesMode::new_deferred(SinDerivative))
    }
    pub fn internal_cos(&self) -> Self {
        let tape = self.tape;
        tape.add_node(self.value().cos(), vec![self.node_idx], DerivativesMode::new_deferred(CosDerivative))
    }
    pub fn internal_tan(&self) -> Self {
        return self.internal_sin() / self.internal_cos();
    }
    pub fn internal_asin(&self) -> Self {
        let tape = self.tape;
        tape.add_node(self.value().asin(), vec![self.node_idx], DerivativesMode::new_deferred(ASinDerivative))
    }
    pub fn internal_acos(&self) -> Self {
        let tape = self.tape;
        tape.add_node(self.value().acos(), vec![self.node_idx], DerivativesMode::new_deferred(ACosDerivative))
    }
    pub fn internal_atan(&self) -> Self {
        let tape = self.tape;
        tape.add_node(self.value().atan(), vec![self.node_idx], DerivativesMode::new_deferred(ATanDerivative))
    }
    pub fn internal_sinh(&self) -> Self {
        let tape = self.tape;
        tape.add_node(self.value().sinh(), vec![self.node_idx], DerivativesMode::new_deferred(SinhDerivative))
    }
    pub fn internal_cosh(&self) -> Self {
        let tape = self.tape;
        tape.add_node(self.value().cosh(), vec![self.node_idx], DerivativesMode::new_deferred(CoshDerivative))
    }
    pub fn internal_tanh(&self) -> Self {
        let tape = self.tape;
        tape.add_node(self.value().tanh(), vec![self.node_idx], DerivativesMode::new_deferred(TanhDerivative))
    }
    pub fn internal_asinh(&self) -> Self {
        let tape = self.tape;
        tape.add_node(self.value().asinh(), vec![self.node_idx], DerivativesMode::new_deferred(ASinhDerivative))
    }
    pub fn internal_acosh(&self) -> Self {
        let tape = self.tape;
        tape.add_node(self.value().acosh(), vec![self.node_idx], DerivativesMode::new_deferred(ACoshDerivative))
    }
    pub fn internal_atanh(&self) -> Self {
        let tape = self.tape;
        tape.add_node(self.value().atanh(), vec![self.node_idx], DerivativesMode::new_deferred(ATanhDerivative))
    }
    fn internal_neg_sin(&self) -> Self {
        let tape = self.tape;
        tape.add_node(-self.value().sin(), vec![self.node_idx], DerivativesMode::new_deferred(NegSinDerivative))
    }
    fn internal_neg_cos(&self) -> Self {
        let tape = self.tape;
        tape.add_node(-self.value().cos(), vec![self.node_idx], DerivativesMode::new_deferred(NegCosDerivative))
    }
}

#[derive(Clone, Debug)]
pub struct SinDerivative;
impl DerivativeFunction for SinDerivative {
    fn compute(&self, operand_node_idxs: Vec<usize>, tape: &'static Tape) -> Vec<f64ad_var> {
        assert_eq!(operand_node_idxs.len(), 1);

        let operand_f64ad = f64ad_var::new_from_node_idx(operand_node_idxs[0], tape);
        let out = operand_f64ad.internal_cos();
        return vec![out];
    }
}

#[derive(Clone, Debug)]
pub struct CosDerivative;
impl DerivativeFunction for CosDerivative {
    fn compute(&self, operand_node_idxs: Vec<usize>, tape: &'static Tape) -> Vec<f64ad_var> {
        assert_eq!(operand_node_idxs.len(), 1);

        let operand_f64ad = f64ad_var::new_from_node_idx(operand_node_idxs[0], tape);
        let out = operand_f64ad.internal_neg_sin();

        return vec![out];
    }
}

#[derive(Clone, Debug)]
pub struct NegSinDerivative;
impl DerivativeFunction for NegSinDerivative {
    fn compute(&self, operand_node_idxs: Vec<usize>, tape: &'static Tape) -> Vec<f64ad_var> {
        assert_eq!(operand_node_idxs.len(), 1);

        let operand_f64ad = f64ad_var::new_from_node_idx(operand_node_idxs[0], tape);
        let out = operand_f64ad.internal_neg_cos();
        return vec![out];
    }
}

#[derive(Clone, Debug)]
pub struct NegCosDerivative;
impl DerivativeFunction for NegCosDerivative {
    fn compute(&self, operand_node_idxs: Vec<usize>, tape: &'static Tape) -> Vec<f64ad_var> {
        assert_eq!(operand_node_idxs.len(), 1);

        let operand_f64ad = f64ad_var::new_from_node_idx(operand_node_idxs[0], tape);
        let out = operand_f64ad.internal_sin();

        return vec![out];
    }
}

#[derive(Clone, Debug)]
pub struct TanDerivative;
impl DerivativeFunction for TanDerivative {
    fn compute(&self, operand_node_idxs: Vec<usize>, tape: &'static Tape) -> Vec<f64ad_var> {
        assert_eq!(operand_node_idxs.len(), 1);

        let operand_f64ad_var = f64ad_var::new_from_node_idx(operand_node_idxs[0], tape);

        let out = 1.0 / (operand_f64ad_var.internal_cos().internal_powf(2.0));

        return vec![out];
    }
}

#[derive(Clone, Debug)]
pub struct ASinDerivative;
impl DerivativeFunction for ASinDerivative {
    fn compute(&self, operand_node_idxs: Vec<usize>, tape: &'static Tape) -> Vec<f64ad_var> {
        assert_eq!(operand_node_idxs.len(), 1);

        let operand_f64ad_var = f64ad_var::new_from_node_idx(operand_node_idxs[0], tape);

        let out = 1.0 / (1.0 - operand_f64ad_var.internal_powf(2.0)).internal_sqrt();

        return vec![out];
    }
}

#[derive(Clone, Debug)]
pub struct ACosDerivative;
impl DerivativeFunction for ACosDerivative {
    fn compute(&self, operand_node_idxs: Vec<usize>, tape: &'static Tape) -> Vec<f64ad_var> {
        assert_eq!(operand_node_idxs.len(), 1);

        let operand_f64ad_var = f64ad_var::new_from_node_idx(operand_node_idxs[0], tape);

        let out = -1.0 / (1.0 - operand_f64ad_var.internal_powf(2.0)).internal_sqrt();

        return vec![out];
    }
}

#[derive(Clone, Debug)]
pub struct ATanDerivative;
impl DerivativeFunction for ATanDerivative {
    fn compute(&self, operand_node_idxs: Vec<usize>, tape: &'static Tape) -> Vec<f64ad_var> {
        assert_eq!(operand_node_idxs.len(), 1);

        let operand_f64ad_var = f64ad_var::new_from_node_idx(operand_node_idxs[0], tape);

        let out = 1.0 / (1.0 + operand_f64ad_var.internal_powf(2.0));

        return vec![out];
    }
}

#[derive(Clone, Debug)]
pub struct SinhDerivative;
impl DerivativeFunction for SinhDerivative {
    fn compute(&self, operand_node_idxs: Vec<usize>, tape: &'static Tape) -> Vec<f64ad_var> {
        assert_eq!(operand_node_idxs.len(), 1);

        let operand_f64ad_var = f64ad_var::new_from_node_idx(operand_node_idxs[0], tape);

        let out = operand_f64ad_var.internal_cosh();

        return vec![out];
    }
}

#[derive(Clone, Debug)]
pub struct CoshDerivative;
impl DerivativeFunction for CoshDerivative {
    fn compute(&self, operand_node_idxs: Vec<usize>, tape: &'static Tape) -> Vec<f64ad_var> {
        assert_eq!(operand_node_idxs.len(), 1);

        let operand_f64ad_var = f64ad_var::new_from_node_idx(operand_node_idxs[0], tape);

        let out = operand_f64ad_var.internal_sinh();

        return vec![out];
    }
}

#[derive(Clone, Debug)]
pub struct TanhDerivative;
impl DerivativeFunction for TanhDerivative {
    fn compute(&self, operand_node_idxs: Vec<usize>, tape: &'static Tape) -> Vec<f64ad_var> {
        assert_eq!(operand_node_idxs.len(), 1);

        let operand_f64ad_var = f64ad_var::new_from_node_idx(operand_node_idxs[0], tape);

        let out = 1.0 / (operand_f64ad_var.internal_cosh().internal_powf(2.0));

        return vec![out];
    }
}

#[derive(Clone, Debug)]
pub struct ASinhDerivative;
impl DerivativeFunction for ASinhDerivative {
    fn compute(&self, operand_node_idxs: Vec<usize>, tape: &'static Tape) -> Vec<f64ad_var> {
        assert_eq!(operand_node_idxs.len(), 1);

        let operand_f64ad_var = f64ad_var::new_from_node_idx(operand_node_idxs[0], tape);

        let out = 1.0 / (1.0 + operand_f64ad_var.internal_powf(2.0)).internal_sqrt();

        return vec![out];
    }
}

#[derive(Clone, Debug)]
pub struct ACoshDerivative;
impl DerivativeFunction for ACoshDerivative {
    fn compute(&self, operand_node_idxs: Vec<usize>, tape: &'static Tape) -> Vec<f64ad_var> {
        assert_eq!(operand_node_idxs.len(), 1);

        let operand_f64ad_var = f64ad_var::new_from_node_idx(operand_node_idxs[0], tape);

        let out = 1.0 / (operand_f64ad_var.internal_powf(2.0) - 1.0).internal_sqrt();

        return vec![out];
    }
}

#[derive(Clone, Debug)]
pub struct ATanhDerivative;
impl DerivativeFunction for ATanhDerivative {
    fn compute(&self, operand_node_idxs: Vec<usize>, tape: &'static Tape) -> Vec<f64ad_var> {
        assert_eq!(operand_node_idxs.len(), 1);

        let operand_f64ad_var = f64ad_var::new_from_node_idx(operand_node_idxs[0], tape);

        let out = 1.0 / (1.0 - operand_f64ad_var.internal_powf(2.0));

        return vec![out];
    }
}

// Recip //

impl f64ad_var {
    pub fn internal_recip(&self) -> Self {
        let tape = self.tape;
        tape.add_node(self.value().recip(), vec![self.node_idx], DerivativesMode::new_deferred(RecipDerivative))
    }
}

#[derive(Clone, Debug)]
pub struct RecipDerivative;
impl DerivativeFunction for RecipDerivative {
    fn compute(&self, operand_node_idxs: Vec<usize>, tape: &'static Tape) -> Vec<f64ad_var> {
        assert_eq!(operand_node_idxs.len(), 1);

        let operand_f64ad_var = f64ad_var::new_from_node_idx(operand_node_idxs[0], tape);

        let out = -1.0 / (operand_f64ad_var.internal_powf(2.0));

        return vec![out];
    }
}

// Logarithms //

impl f64ad_var {
    pub fn internal_ln(&self) -> Self {
        let tape = self.tape;
        tape.add_node(self.value().ln(), vec![self.node_idx], DerivativesMode::new_deferred(LnDerivative))
    }
    pub fn internal_log(&self, base: f64) -> Self {
        let tape = self.tape;
        tape.add_node(self.value().log(base), vec![self.node_idx], DerivativesMode::new_deferred(LogDerivative { base, base_ln: base.ln() }))
    }
    pub fn internal_log_variable_base(&self, base: f64ad_var) -> Self {
        return self.internal_ln() / base.internal_ln();
    }
    pub fn internal_log10(&self) -> Self {
        return self.internal_log(10.0);
    }
    pub fn internal_log2(&self) -> Self {
        return self.internal_log(2.0);
    }
    pub fn internal_ln_1p(&self) -> Self {
        return (*self + 1.0).internal_ln();
    }
}

#[derive(Clone, Debug)]
pub struct LnDerivative;
impl DerivativeFunction for LnDerivative {
    fn compute(&self, operand_node_idxs: Vec<usize>, tape: &'static Tape) -> Vec<f64ad_var> {
        assert_eq!(operand_node_idxs.len(), 1);

        let operand_f64ad_var = f64ad_var::new_from_node_idx(operand_node_idxs[0], tape);

        let out = operand_f64ad_var.internal_recip();

        return vec![out];
    }
}

#[derive(Clone, Debug)]
pub struct LogDerivative {
    base: f64,
    base_ln: f64
}
impl DerivativeFunction for LogDerivative {
    fn compute(&self, operand_node_idxs: Vec<usize>, tape: &'static Tape) -> Vec<f64ad_var> {
        assert_eq!(operand_node_idxs.len(), 1);

        let operand_f64ad_var = f64ad_var::new_from_node_idx(operand_node_idxs[0], tape);

        let out = 1.0 / (operand_f64ad_var * self.base_ln);

        return vec![out];
    }
}


