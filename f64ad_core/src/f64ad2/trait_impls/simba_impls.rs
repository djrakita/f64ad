use num_traits::{Signed, Zero};
use simba::scalar::{ComplexField, Field, RealField, SubsetOf, SupersetOf};
use tinyvec::tiny_vec;
use crate::f64ad2::{f64ad, f64ad_locked_var_operation_one_parent, f64ad_locked_var_operation_two_parents, NodeType};

impl RealField for f64ad {
    fn is_sign_positive(&self) -> bool {
        return self.is_positive()
    }

    fn is_sign_negative(&self) -> bool {
        return self.is_negative()
    }

    fn copysign(self, sign: Self) -> Self {
        return if sign.is_positive() {
            self.abs()
        } else {
            -self.abs()
        }
    }

    fn max(self, other: Self) -> Self {
        match (&self, &other) {
            (f64ad::f64ad_var(v1), f64ad::f64ad_var(v2)) => {
                assert_eq!(v1.computation_graph_id, v2.computation_graph_id);
                let res = unsafe {
                    (*v1.computation_graph.0).add_node(NodeType::MaxTwoParents, tiny_vec!([u32; 2] => v1.node_idx, v2.node_idx), tiny_vec!([f64; 1]))
                };
                return f64ad::f64ad_var(res);
            }
            (f64ad::f64ad_locked_var(v1), f64ad::f64ad_locked_var(v2)) => {
                let res = f64ad_locked_var_operation_two_parents(v1, v2, NodeType::MaxTwoParents);
                return f64ad::f64ad_locked_var(res);
            }
            (f64ad::f64(v1), f64ad::f64(v2)) => {
                return f64ad::f64(v1.max(*v2))
            }
            (f64ad::f64ad_var(_), f64ad::f64ad_locked_var(_)) => { panic!("unsupported.") }
            (f64ad::f64ad_locked_var(_), f64ad::f64ad_var(_)) => { panic!("unsupported.") }
            (f64ad::f64ad_var(v1), f64ad::f64(v2)) => {
                let res = unsafe {
                    (*v1.computation_graph.0).add_node(NodeType::MaxOneParent, tiny_vec!([u32; 2] => v1.node_idx), tiny_vec!([f64; 1] => *v2))
                };
                return f64ad::f64ad_var(res);
            }
            (f64ad::f64(v1), f64ad::f64ad_var(v2)) => {
                let res = unsafe {
                    (*v2.computation_graph.0).add_node(NodeType::MaxOneParent, tiny_vec!([u32; 2] => v2.node_idx), tiny_vec!([f64; 1] => *v1))
                };
                return f64ad::f64ad_var(res);
            }
            (f64ad::f64ad_locked_var(v1), f64ad::f64(v2)) => {
                let res = f64ad_locked_var_operation_one_parent(v1, Some(*v2), NodeType::MaxOneParent);
                return f64ad::f64ad_locked_var(res);
            }
            (f64ad::f64(v1), f64ad::f64ad_locked_var(v2)) => {
                let res = f64ad_locked_var_operation_one_parent(v2, Some(*v1), NodeType::MaxOneParent);
                return f64ad::f64ad_locked_var(res);            }
        }
    }

    fn min(self, other: Self) -> Self {
        match (&self, &other) {
            (f64ad::f64ad_var(v1), f64ad::f64ad_var(v2)) => {
                assert_eq!(v1.computation_graph_id, v2.computation_graph_id);
                let res = unsafe {
                    (*v1.computation_graph.0).add_node(NodeType::MinTwoParents, tiny_vec!([u32; 2] => v1.node_idx, v2.node_idx), tiny_vec!([f64; 1]))
                };
                return f64ad::f64ad_var(res);
            }
            (f64ad::f64ad_locked_var(v1), f64ad::f64ad_locked_var(v2)) => {
                let res = f64ad_locked_var_operation_two_parents(v1, v2, NodeType::MinTwoParents);
                return f64ad::f64ad_locked_var(res);
            }
            (f64ad::f64(v1), f64ad::f64(v2)) => {
                return f64ad::f64(v1.min(*v2))
            }
            (f64ad::f64ad_var(_), f64ad::f64ad_locked_var(_)) => { panic!("unsupported.") }
            (f64ad::f64ad_locked_var(_), f64ad::f64ad_var(_)) => { panic!("unsupported.") }
            (f64ad::f64ad_var(v1), f64ad::f64(v2)) => {
                let res = unsafe {
                    (*v1.computation_graph.0).add_node(NodeType::MinOneParent, tiny_vec!([u32; 2] => v1.node_idx), tiny_vec!([f64; 1] => *v2))
                };
                return f64ad::f64ad_var(res);
            }
            (f64ad::f64(v1), f64ad::f64ad_var(v2)) => {
                let res = unsafe {
                    (*v2.computation_graph.0).add_node(NodeType::MinOneParent, tiny_vec!([u32; 2] => v2.node_idx), tiny_vec!([f64; 1] => *v1))
                };
                return f64ad::f64ad_var(res);
            }
            (f64ad::f64ad_locked_var(v1), f64ad::f64(v2)) => {
                let res = f64ad_locked_var_operation_one_parent(v1, Some(*v2), NodeType::MinOneParent);
                return f64ad::f64ad_locked_var(res);
            }
            (f64ad::f64(v1), f64ad::f64ad_locked_var(v2)) => {
                let res = f64ad_locked_var_operation_one_parent(v2, Some(*v1), NodeType::MinOneParent);
                return f64ad::f64ad_locked_var(res);            }
        }
    }

    fn clamp(self, min: Self, max: Self) -> Self {
        assert!(min <= max);
        return self.max(min).min(max);
    }

    fn atan2(self, other: Self) -> Self {
        todo!()
    }

    fn min_value() -> Option<Self> {
        Some(f64ad::f64(f64::MIN))
    }

    fn max_value() -> Option<Self> {
        Some(f64ad::f64(f64::MAX))
    }

    fn pi() -> Self {
        f64ad::f64(std::f64::consts::PI)
    }

    fn two_pi() -> Self {
        f64ad::f64(2.0 * std::f64::consts::PI)
    }

    fn frac_pi_2() -> Self {
        f64ad::f64(std::f64::consts::FRAC_PI_2)
    }

    fn frac_pi_3() -> Self {
        f64ad::f64(std::f64::consts::FRAC_PI_3)
    }

    fn frac_pi_4() -> Self {
        todo!()
    }

    fn frac_pi_6() -> Self {
        todo!()
    }

    fn frac_pi_8() -> Self {
        todo!()
    }

    fn frac_1_pi() -> Self {
        todo!()
    }

    fn frac_2_pi() -> Self {
        todo!()
    }

    fn frac_2_sqrt_pi() -> Self {
        todo!()
    }

    fn e() -> Self {
        todo!()
    }

    fn log2_e() -> Self {
        todo!()
    }

    fn log10_e() -> Self {
        todo!()
    }

    fn ln_2() -> Self {
        todo!()
    }

    fn ln_10() -> Self {
        todo!()
    }
}

impl ComplexField for f64ad {
    type RealField = f64ad;

    fn from_real(re: Self::RealField) -> Self { re.clone() }
    fn real(self) -> <Self as ComplexField>::RealField { self.clone() }
    fn imaginary(self) -> Self::RealField { Self::zero() }
    fn modulus(self) -> Self::RealField { return self.abs() }
    fn modulus_squared(self) -> Self::RealField { self * self }
    fn argument(self) -> Self::RealField { unimplemented!(); }
    fn norm1(self) -> Self::RealField { return self.abs(); }
    fn scale(self, factor: Self::RealField) -> Self { return self * factor; }
    fn unscale(self, factor: Self::RealField) -> Self { return self / factor; }
    fn floor(self) -> Self { todo!() }
    fn ceil(self) -> Self { todo!() }
    fn round(self) -> Self { todo!() }
    fn trunc(self) -> Self { todo!() }
    fn fract(self) -> Self { todo!() }
    fn mul_add(self, a: Self, b: Self) -> Self { return (self * a) + b; }
    fn abs(self) -> Self::RealField {
        <Self as Signed>::abs(&self)
    }
    fn hypot(self, other: Self) -> Self::RealField {
        return (self.powi(2) + other.powi(2)).sqrt();
    }
    fn recip(self) -> Self { todo!() }
    fn conjugate(self) -> Self { todo!() }
    fn sin(self) -> Self { todo!() }
    fn cos(self) -> Self {
        todo!()
    }
    fn sin_cos(self) -> (Self, Self) {
        return (self.sin(), self.cos())
    }
    fn tan(self) -> Self { todo!() }
    fn asin(self) -> Self { todo!() }
    fn acos(self) -> Self { todo!() }
    fn atan(self) -> Self { todo!() }
    fn sinh(self) -> Self { todo!() }
    fn cosh(self) -> Self { todo!() }
    fn tanh(self) -> Self { todo!() }
    fn asinh(self) -> Self { todo!() }
    fn acosh(self) -> Self { todo!() }
    fn atanh(self) -> Self { todo!() }
    fn log(self, base: Self::RealField) -> Self { todo!() }
    fn log2(self) -> Self { todo!() }
    fn log10(self) -> Self { todo!() }
    fn ln(self) -> Self { todo!() }
    fn ln_1p(self) -> Self { todo!() }
    fn sqrt(self) -> Self { todo!() }
    fn exp(self) -> Self { todo!() }
    fn exp2(self) -> Self { todo!() }
    fn exp_m1(self) -> Self { return self.exp() - 1.0; }
    fn powi(self, n: i32) -> Self { todo!() }
    fn powf(self, n: Self::RealField) -> Self { todo!() }
    fn powc(self, n: Self) -> Self { return self.powf(n); }
    fn cbrt(self) -> Self { todo!() }
    fn is_finite(&self) -> bool {
        todo!()
    }
    fn try_sqrt(self) -> Option<Self> {
        Some(self.sqrt())
    }
}

impl SubsetOf<f64ad> for f64ad {
    fn to_superset(&self) -> f64ad {
        todo!()
    }

    fn from_superset_unchecked(element: &f64ad) -> Self {
        todo!()
    }

    fn is_in_subset(element: &f64ad) -> bool {
        todo!()
    }
}

impl SupersetOf<f64> for f64ad {
    fn is_in_subset(&self) -> bool {
        todo!()
    }

    fn to_subset_unchecked(&self) -> f64 {
        todo!()
    }

    fn from_subset(element: &f64) -> Self {
        todo!()
    }
}

impl Field for f64ad {}