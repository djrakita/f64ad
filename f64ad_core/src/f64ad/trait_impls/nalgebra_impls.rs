use std::ops::{Mul};
use nalgebra::{Dim, Matrix, RawStorageMut, SimdValue};
use crate::f64ad::f64ad;

impl SimdValue for f64ad {
    type Element = f64ad;
    type SimdBool = bool;

    fn lanes() -> usize {
        4
    }

    fn splat(val: Self::Element) -> Self {
        val
    }

    fn extract(&self, _: usize) -> Self::Element {
        *self
    }

    unsafe fn extract_unchecked(&self, _: usize) -> Self::Element {
        *self
    }

    fn replace(&mut self, _: usize, val: Self::Element) {
        *self = val
    }

    unsafe fn replace_unchecked(&mut self, _: usize, val: Self::Element) {
        *self = val
    }

    fn select(self, cond: Self::SimdBool, other: Self) -> Self {
        if cond {
            self
        } else {
            other
        }
    }
}

impl<R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<f64ad, R, C>> Mul<Matrix<f64ad, R, C, S>> for f64ad {
    type Output = Matrix<f64ad, R, C, S>;

    fn mul(self, rhs: Matrix<f64ad, R, C, S>) -> Self::Output {
        let mut out_clone = rhs.clone();
        for e in out_clone.iter_mut() {
            *e *= self;
        }
        out_clone
    }
}

impl<R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<f64, R, C>> Mul<Matrix<f64, R, C, S>> for f64ad {
    type Output = Matrix<f64, R, C, S>;

    fn mul(self, rhs: Matrix<f64, R, C, S>) -> Self::Output {
        let mut out_clone = rhs.clone();
        for e in out_clone.iter_mut() {
            *e *= self;
        }
        out_clone
    }
}

impl<R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<f64ad, R, C>> Mul<&Matrix<f64ad, R, C, S>> for f64ad {
    type Output = Matrix<f64ad, R, C, S>;

    fn mul(self, rhs: &Matrix<f64ad, R, C, S>) -> Self::Output {
        let mut out_clone = rhs.clone();
        for e in out_clone.iter_mut() {
            *e *= self;
        }
        out_clone
    }
}

impl<R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<f64, R, C>> Mul<&Matrix<f64, R, C, S>> for f64ad {
    type Output = Matrix<f64, R, C, S>;

    fn mul(self, rhs: &Matrix<f64, R, C, S>) -> Self::Output {
        let mut out_clone = rhs.clone();
        for e in out_clone.iter_mut() {
            *e *= self;
        }
        out_clone
    }
}