use std::fmt::Debug;
use std::ops::{Index, IndexMut};
use std::time::Instant;
use std::vec::IntoIter;

#[derive(Clone, Debug)]
pub enum PreallocatedVec<T: PreallocatedVecItemDefault + Clone + Debug> {
    StandardVec { v: Vec<T> },
    PreallocateThenSinglePushes { v: Vec<T>, curr_idx: usize },
    PreallocateThenReallocateDouble { v: Vec<T>, curr_idx: usize }
}
impl<T> PreallocatedVec<T> where T: PreallocatedVecItemDefault + Clone + Debug {
    pub fn new_standard_vec() -> Self {
        Self::StandardVec { v: vec![] }
    }
    pub fn new_preallocate_then_single_pushes(n: usize) -> Self {
        Self::PreallocateThenSinglePushes { v: vec![T::preallocated_vec_item_default(); n], curr_idx: 0 }
    }
    pub fn new_preallocate_then_reallocate_double(n: usize) -> Self {
        Self::PreallocateThenReallocateDouble { v: vec![T::preallocated_vec_item_default(); n], curr_idx: 0 }
    }
    pub fn softpush(&mut self) {
        match self {
            PreallocatedVec::StandardVec { v } => {
                v.push(T::preallocated_vec_item_default());
            }
            PreallocatedVec::PreallocateThenSinglePushes { v, curr_idx } => {
                let res = v.get(*curr_idx+1);
                return match res {
                    None => {
                        *self = PreallocatedVec::StandardVec { v: v.clone() };
                        self.softpush();
                    }
                    Some(_) => {
                        *curr_idx += 1;
                    }
                }
            }
            PreallocatedVec::PreallocateThenReallocateDouble { v, curr_idx } => {
                let res = v.get(*curr_idx+1);
                match res {
                    None => {
                        let l = v.len();
                        for _ in 0..l { v.push(T::preallocated_vec_item_default()); }
                        *curr_idx += 1;
                    }
                    Some(_) => {
                        *curr_idx += 1;
                    }
                }
            }
        }
    }
    pub fn push(&mut self, element: T) {
        match self {
            PreallocatedVec::StandardVec { v } => { v.push(element) }
            PreallocatedVec::PreallocateThenSinglePushes { v, curr_idx } => {
                let res = v.get(*curr_idx);
                match res {
                    None => {
                        *self = PreallocatedVec::StandardVec { v: v.clone() };
                        self.push(element);
                    }
                    Some(_) => {
                        v[*curr_idx] = element;
                        *curr_idx += 1;
                    }
                }
            }
            PreallocatedVec::PreallocateThenReallocateDouble { v, curr_idx } => {
                let res = v.get(*curr_idx);
                match res {
                    None => {
                        let l = v.len();
                        for _ in 0..l { v.push(T::preallocated_vec_item_default()); }
                        v[*curr_idx] = element;
                        *curr_idx += 1;
                    }
                    Some(_) => {
                        v[*curr_idx] = element;
                        *curr_idx += 1;
                    }
                }
            }
        }
    }
    pub fn len(&self) -> usize {
        return match self {
            PreallocatedVec::StandardVec { v } => { v.len() }
            PreallocatedVec::PreallocateThenSinglePushes { curr_idx, .. } => { *curr_idx }
            PreallocatedVec::PreallocateThenReallocateDouble { curr_idx, .. } => { *curr_idx }
        }
    }
    pub fn curr_item_ref(&self) -> &T {
        match self {
            PreallocatedVec::StandardVec { v } => { &v[v.len()] }
            PreallocatedVec::PreallocateThenSinglePushes { v, curr_idx } => { &v[*curr_idx] }
            PreallocatedVec::PreallocateThenReallocateDouble { v, curr_idx } => { &v[*curr_idx] }
        }
    }
    pub fn curr_item_mut_ref(&mut self) -> &mut T {
        match self {
            PreallocatedVec::StandardVec { v } => { let l = v.len(); &mut v[l-1] }
            PreallocatedVec::PreallocateThenSinglePushes { v, curr_idx } => { &mut v[*curr_idx] }
            PreallocatedVec::PreallocateThenReallocateDouble { v, curr_idx } => { &mut v[*curr_idx] }
        }
    }
    pub fn softclear(&mut self) {
        match self {
            PreallocatedVec::StandardVec { v } => {
                v.clear()
            }
            PreallocatedVec::PreallocateThenSinglePushes { v, curr_idx } => {
                *curr_idx = 0;
            }
            PreallocatedVec::PreallocateThenReallocateDouble { v, curr_idx } => {
                *curr_idx = 0;
            }
        }
    }
    pub fn hardclear(&mut self) {
        match self {
            PreallocatedVec::StandardVec { v } => {
                v.clear()
            }
            PreallocatedVec::PreallocateThenSinglePushes { v, curr_idx } => {
                v.clear();
                *curr_idx = 0;
            }
            PreallocatedVec::PreallocateThenReallocateDouble { v, curr_idx } => {
                v.clear();
                *curr_idx = 0;
            }
        }
    }
}
impl<T> Index<usize> for PreallocatedVec<T> where T: PreallocatedVecItemDefault + Clone + Debug {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match self {
            PreallocatedVec::StandardVec { v } => {
                &v[index]
            }
            PreallocatedVec::PreallocateThenSinglePushes { v, curr_idx } => {
                assert!(*curr_idx > index);
                &v[index]
            }
            PreallocatedVec::PreallocateThenReallocateDouble { v, curr_idx } => {
                assert!(*curr_idx > index);
                &v[index]
            }
        }
    }
}
impl<T> IndexMut<usize> for PreallocatedVec<T> where T: PreallocatedVecItemDefault + Clone + Debug {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match self {
            PreallocatedVec::StandardVec { v } => {
                &mut v[index]
            }
            PreallocatedVec::PreallocateThenSinglePushes { v, curr_idx } => {
                assert!(*curr_idx > index);
                &mut v[index]
            }
            PreallocatedVec::PreallocateThenReallocateDouble { v, curr_idx } => {
                assert!(*curr_idx > index);
                &mut v[index]
            }
        }
    }
}

pub trait PreallocatedVecItemDefault {
    fn preallocated_vec_item_default() -> Self;
}
impl<T> PreallocatedVecItemDefault for T where T: Default {
    fn preallocated_vec_item_default() -> Self {
        T::default()
    }
}