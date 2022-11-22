use num_traits::{Signed, Zero};
use simba::scalar::{ComplexField, Field, RealField, SubsetOf};
use simba::simd::PrimitiveSimdValue;
use tinyvec::tiny_vec;
use crate::f64ad::{ComputationGraphMode, f64ad, f64ad_locked_var_operation_one_parent, f64ad_locked_var_operation_two_parents, NodeType};

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
                let res = v1.computation_graph.add_node(NodeType::MaxTwoParents, tiny_vec!([u32; 2] => v1.node_idx, v2.node_idx), tiny_vec!([f64; 1]));
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
                let res = v1.computation_graph.add_node(NodeType::MaxOneParent, tiny_vec!([u32; 2] => v1.node_idx), tiny_vec!([f64; 1] => *v2));
                return f64ad::f64ad_var(res);
            }
            (f64ad::f64(v1), f64ad::f64ad_var(v2)) => {
                let res = v2.computation_graph.add_node(NodeType::MaxOneParent, tiny_vec!([u32; 2] => v2.node_idx), tiny_vec!([f64; 1] => *v1));
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
                let res = v1.computation_graph.add_node(NodeType::MinTwoParents, tiny_vec!([u32; 2] => v1.node_idx, v2.node_idx), tiny_vec!([f64; 1]));
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
                let res = v1.computation_graph.add_node(NodeType::MinOneParent, tiny_vec!([u32; 2] => v1.node_idx), tiny_vec!([f64; 1] => *v2));
                return f64ad::f64ad_var(res);
            }
            (f64ad::f64(v1), f64ad::f64ad_var(v2)) => {
                let res = v2.computation_graph.add_node(NodeType::MinOneParent, tiny_vec!([u32; 2] => v2.node_idx), tiny_vec!([f64; 1] => *v1));
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
        match (&self, &other) {
            (f64ad::f64ad_var(v1), f64ad::f64ad_var(v2)) => {
                assert_eq!(v1.computation_graph_id, v2.computation_graph_id);
                let res = v1.computation_graph.add_node(NodeType::Atan2TwoParents, tiny_vec!([u32; 2] => v1.node_idx, v2.node_idx), tiny_vec!([f64; 1]));
                return f64ad::f64ad_var(res);
            }
            (f64ad::f64ad_locked_var(v1), f64ad::f64ad_locked_var(v2)) => {
                let res = f64ad_locked_var_operation_two_parents(v1, v2, NodeType::Atan2TwoParents);
                return f64ad::f64ad_locked_var(res);
            }
            (f64ad::f64(v1), f64ad::f64(v2)) => {
                return f64ad::f64(v1.atan2(*v2))
            }
            (f64ad::f64ad_var(_), f64ad::f64ad_locked_var(_)) => { panic!("unsupported.") }
            (f64ad::f64ad_locked_var(_), f64ad::f64ad_var(_)) => { panic!("unsupported.") }
            (f64ad::f64ad_var(v1), f64ad::f64(v2)) => {
                let res = v1.computation_graph.add_node(NodeType::Atan2OneParentLeft, tiny_vec!([u32; 2] => v1.node_idx), tiny_vec!([f64; 1] => *v2));
                return f64ad::f64ad_var(res);
            }
            (f64ad::f64(v1), f64ad::f64ad_var(v2)) => {
                let res = v2.computation_graph.add_node(NodeType::Atan2OneParentRight, tiny_vec!([u32; 2] => v2.node_idx), tiny_vec!([f64; 1] => *v1));
                return f64ad::f64ad_var(res);
            }
            (f64ad::f64ad_locked_var(v1), f64ad::f64(v2)) => {
                let res = f64ad_locked_var_operation_one_parent(v1, Some(*v2), NodeType::Atan2OneParentLeft);
                return f64ad::f64ad_locked_var(res);
            }
            (f64ad::f64(v1), f64ad::f64ad_locked_var(v2)) => {
                let res = f64ad_locked_var_operation_one_parent(v2, Some(*v1), NodeType::Atan2OneParentRight);
                return f64ad::f64ad_locked_var(res);            }
        }
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
        f64ad::f64(std::f64::consts::FRAC_PI_4)
    }

    fn frac_pi_6() -> Self {
        f64ad::f64(std::f64::consts::FRAC_PI_6)
    }

    fn frac_pi_8() -> Self {
        f64ad::f64(std::f64::consts::FRAC_PI_8)
    }

    fn frac_1_pi() -> Self {
        f64ad::f64(std::f64::consts::FRAC_1_PI)
    }

    fn frac_2_pi() -> Self {
        f64ad::f64(std::f64::consts::FRAC_2_PI)
    }

    fn frac_2_sqrt_pi() -> Self {
        f64ad::f64(std::f64::consts::FRAC_2_SQRT_PI)
    }

    fn e() -> Self {
        f64ad::f64(std::f64::consts::E)
    }

    fn log2_e() -> Self {
        f64ad::f64(std::f64::consts::LOG2_E)
    }

    fn log10_e() -> Self {
        f64ad::f64(std::f64::consts::LOG10_E)
    }

    fn ln_2() -> Self {
        f64ad::f64(std::f64::consts::LN_2)
    }

    fn ln_10() -> Self {
        f64ad::f64(std::f64::consts::LN_10)
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
    fn floor(self) -> Self {
        return match &self {
            f64ad::f64ad_var(f) => {
                let res = f.computation_graph.add_node(NodeType::Floor, tiny_vec!([u32; 2] => f.node_idx), tiny_vec!([f64; 1]));
                f64ad::f64ad_var(res)
            }
            f64ad::f64ad_locked_var(f) => {
                f64ad::f64ad_locked_var(f64ad_locked_var_operation_one_parent(f, None, NodeType::Floor))
            }
            f64ad::f64(f) => {
                f64ad::f64(f.floor())
            }
        }
    }
    fn ceil(self) -> Self {
        return match &self {
            f64ad::f64ad_var(f) => {
                let res = f.computation_graph.add_node(NodeType::Ceil, tiny_vec!([u32; 2] => f.node_idx), tiny_vec!([f64; 1]));
                f64ad::f64ad_var(res)
            }
            f64ad::f64ad_locked_var(f) => {
                f64ad::f64ad_locked_var(f64ad_locked_var_operation_one_parent(f, None, NodeType::Ceil))
            }
            f64ad::f64(f) => {
                f64ad::f64(f.ceil())
            }
        }
    }
    fn round(self) -> Self {
        return match &self {
            f64ad::f64ad_var(f) => {
                let res = f.computation_graph.add_node(NodeType::Round, tiny_vec!([u32; 2] => f.node_idx), tiny_vec!([f64; 1]));
                f64ad::f64ad_var(res)
            }
            f64ad::f64ad_locked_var(f) => {
                f64ad::f64ad_locked_var(f64ad_locked_var_operation_one_parent(f, None, NodeType::Round))
            }
            f64ad::f64(f) => {
                f64ad::f64(f.round())
            }
        }
    }
    fn trunc(self) -> Self {
        return match &self {
            f64ad::f64ad_var(f) => {
                let res = f.computation_graph.add_node(NodeType::Trunc, tiny_vec!([u32; 2] => f.node_idx), tiny_vec!([f64; 1]));
                f64ad::f64ad_var(res)
            }
            f64ad::f64ad_locked_var(f) => {
                f64ad::f64ad_locked_var(f64ad_locked_var_operation_one_parent(f, None, NodeType::Trunc))
            }
            f64ad::f64(f) => {
                f64ad::f64(f.trunc())
            }
        }
    }
    fn fract(self) -> Self {
        return match &self {
            f64ad::f64ad_var(f) => {
                let res = f.computation_graph.add_node(NodeType::Fract, tiny_vec!([u32; 2] => f.node_idx), tiny_vec!([f64; 1]));
                f64ad::f64ad_var(res)
            }
            f64ad::f64ad_locked_var(f) => {
                f64ad::f64ad_locked_var(f64ad_locked_var_operation_one_parent(f, None, NodeType::Fract))
            }
            f64ad::f64(f) => {
                f64ad::f64(f.fract())
            }
        }
    }
    fn mul_add(self, a: Self, b: Self) -> Self { return (self * a) + b; }
    fn abs(self) -> Self::RealField {
        <Self as Signed>::abs(&self)
    }
    fn hypot(self, other: Self) -> Self::RealField {
        return (self.powi(2) + other.powi(2)).sqrt();
    }
    fn recip(self) -> Self { return 1.0/self; }
    fn conjugate(self) -> Self { return self; }
    fn sin(self) -> Self {
        return match &self {
            f64ad::f64ad_var(f) => {
                let res = f.computation_graph.add_node(NodeType::Sin, tiny_vec!([u32; 2] => f.node_idx), tiny_vec!([f64; 1]));
                f64ad::f64ad_var(res)
            }
            f64ad::f64ad_locked_var(f) => {
                f64ad::f64ad_locked_var(f64ad_locked_var_operation_one_parent(f, None, NodeType::Sin))
            }
            f64ad::f64(f) => {
                f64ad::f64(f.sin())
            }
        }
    }
    fn cos(self) -> Self {
        return match &self {
            f64ad::f64ad_var(f) => {
                let res = f.computation_graph.add_node(NodeType::Cos, tiny_vec!([u32; 2] => f.node_idx), tiny_vec!([f64; 1]));
                f64ad::f64ad_var(res)
            }
            f64ad::f64ad_locked_var(f) => {
                f64ad::f64ad_locked_var(f64ad_locked_var_operation_one_parent(f, None, NodeType::Cos))
            }
            f64ad::f64(f) => {
                f64ad::f64(f.cos())
            }
        }
    }
    fn sin_cos(self) -> (Self, Self) {
        return (self.sin(), self.cos())
    }
    fn tan(self) -> Self {
        return match &self {
            f64ad::f64ad_var(f) => {
                let res = f.computation_graph.add_node(NodeType::Tan, tiny_vec!([u32; 2] => f.node_idx), tiny_vec!([f64; 1]));
                f64ad::f64ad_var(res)
            }
            f64ad::f64ad_locked_var(f) => {
                f64ad::f64ad_locked_var(f64ad_locked_var_operation_one_parent(f, None, NodeType::Tan))
            }
            f64ad::f64(f) => {
                f64ad::f64(f.tan())
            }
        }
    }
    fn asin(self) -> Self {
        return match &self {
            f64ad::f64ad_var(f) => {
                let res = f.computation_graph.add_node(NodeType::Asin, tiny_vec!([u32; 2] => f.node_idx), tiny_vec!([f64; 1]));
                f64ad::f64ad_var(res)
            }
            f64ad::f64ad_locked_var(f) => {
                f64ad::f64ad_locked_var(f64ad_locked_var_operation_one_parent(f, None, NodeType::Asin))
            }
            f64ad::f64(f) => {
                f64ad::f64(f.asin())
            }
        }
    }
    fn acos(self) -> Self {
        return match &self {
            f64ad::f64ad_var(f) => {
                let res  = f.computation_graph.add_node(NodeType::Acos, tiny_vec!([u32; 2] => f.node_idx), tiny_vec!([f64; 1]));
                f64ad::f64ad_var(res)
            }
            f64ad::f64ad_locked_var(f) => {
                f64ad::f64ad_locked_var(f64ad_locked_var_operation_one_parent(f, None, NodeType::Acos))
            }
            f64ad::f64(f) => {
                f64ad::f64(f.acos())
            }
        }
    }
    fn atan(self) -> Self {
        return match &self {
            f64ad::f64ad_var(f) => {
                let res = f.computation_graph.add_node(NodeType::Atan, tiny_vec!([u32; 2] => f.node_idx), tiny_vec!([f64; 1]));
                f64ad::f64ad_var(res)
            }
            f64ad::f64ad_locked_var(f) => {
                f64ad::f64ad_locked_var(f64ad_locked_var_operation_one_parent(f, None, NodeType::Atan))
            }
            f64ad::f64(f) => {
                f64ad::f64(f.atan())
            }
        }
    }
    fn sinh(self) -> Self {
        return match &self {
            f64ad::f64ad_var(f) => {
                let res = f.computation_graph.add_node(NodeType::Sinh, tiny_vec!([u32; 2] => f.node_idx), tiny_vec!([f64; 1]));
                f64ad::f64ad_var(res)
            }
            f64ad::f64ad_locked_var(f) => {
                f64ad::f64ad_locked_var(f64ad_locked_var_operation_one_parent(f, None, NodeType::Sinh))
            }
            f64ad::f64(f) => {
                f64ad::f64(f.sinh())
            }
        }
    }
    fn cosh(self) -> Self {
        return match &self {
            f64ad::f64ad_var(f) => {
                let res = f.computation_graph.add_node(NodeType::Cosh, tiny_vec!([u32; 2] => f.node_idx), tiny_vec!([f64; 1]));
                f64ad::f64ad_var(res)
            }
            f64ad::f64ad_locked_var(f) => {
                f64ad::f64ad_locked_var(f64ad_locked_var_operation_one_parent(f, None, NodeType::Cosh))
            }
            f64ad::f64(f) => {
                f64ad::f64(f.cosh())
            }
        }
    }
    fn tanh(self) -> Self {
        return match &self {
            f64ad::f64ad_var(f) => {
                let res = f.computation_graph.add_node(NodeType::Tanh, tiny_vec!([u32; 2] => f.node_idx), tiny_vec!([f64; 1]));
                f64ad::f64ad_var(res)
            }
            f64ad::f64ad_locked_var(f) => {
                f64ad::f64ad_locked_var(f64ad_locked_var_operation_one_parent(f, None, NodeType::Tanh))
            }
            f64ad::f64(f) => {
                f64ad::f64(f.tanh())
            }
        }
    }
    fn asinh(self) -> Self {
        return match &self {
            f64ad::f64ad_var(f) => {
                let res = f.computation_graph.add_node(NodeType::Asinh, tiny_vec!([u32; 2] => f.node_idx), tiny_vec!([f64; 1]));
                f64ad::f64ad_var(res)
            }
            f64ad::f64ad_locked_var(f) => {
                f64ad::f64ad_locked_var(f64ad_locked_var_operation_one_parent(f, None, NodeType::Asinh))
            }
            f64ad::f64(f) => {
                f64ad::f64(f.asinh())
            }
        }
    }
    fn acosh(self) -> Self {
        return match &self {
            f64ad::f64ad_var(f) => {
                let res = f.computation_graph.add_node(NodeType::Acosh, tiny_vec!([u32; 2] => f.node_idx), tiny_vec!([f64; 1]));
                f64ad::f64ad_var(res)
            }
            f64ad::f64ad_locked_var(f) => {
                f64ad::f64ad_locked_var(f64ad_locked_var_operation_one_parent(f, None, NodeType::Acosh))
            }
            f64ad::f64(f) => {
                f64ad::f64(f.acosh())
            }
        }
    }
    fn atanh(self) -> Self {
        return match &self {
            f64ad::f64ad_var(f) => {
                let res = f.computation_graph.add_node(NodeType::Atanh, tiny_vec!([u32; 2] => f.node_idx), tiny_vec!([f64; 1]));
                f64ad::f64ad_var(res)
            }
            f64ad::f64ad_locked_var(f) => {
                f64ad::f64ad_locked_var(f64ad_locked_var_operation_one_parent(f, None, NodeType::Atanh))
            }
            f64ad::f64(f) => {
                f64ad::f64(f.atanh())
            }
        }
    }
    fn log(self, base: Self::RealField) -> Self {
        match (&self, &base) {
            (f64ad::f64ad_var(v1), f64ad::f64ad_var(v2)) => {
                assert_eq!(v1.computation_graph_id, v2.computation_graph_id);
                let res = v1.computation_graph.add_node(NodeType::LogTwoParents, tiny_vec!([u32; 2] => v1.node_idx, v2.node_idx), tiny_vec!([f64; 1]));
                return f64ad::f64ad_var(res);
            }
            (f64ad::f64ad_locked_var(v1), f64ad::f64ad_locked_var(v2)) => {
                let res = f64ad_locked_var_operation_two_parents(v1, v2, NodeType::LogTwoParents);
                return f64ad::f64ad_locked_var(res);
            }
            (f64ad::f64(v1), f64ad::f64(v2)) => {
                return f64ad::f64(v1.log(*v2))
            }
            (f64ad::f64ad_var(_), f64ad::f64ad_locked_var(_)) => { panic!("unsupported.") }
            (f64ad::f64ad_locked_var(_), f64ad::f64ad_var(_)) => { panic!("unsupported.") }
            (f64ad::f64ad_var(v1), f64ad::f64(v2)) => {
                let res = v1.computation_graph.add_node(NodeType::LogOneParentArgument, tiny_vec!([u32; 2] => v1.node_idx), tiny_vec!([f64; 1] => *v2));
                return f64ad::f64ad_var(res);
            }
            (f64ad::f64(v1), f64ad::f64ad_var(v2)) => {
                let res = v2.computation_graph.add_node(NodeType::LogOneParentBase, tiny_vec!([u32; 2] => v2.node_idx), tiny_vec!([f64; 1] => *v1));
                return f64ad::f64ad_var(res);
            }
            (f64ad::f64ad_locked_var(v1), f64ad::f64(v2)) => {
                let res = f64ad_locked_var_operation_one_parent(v1, Some(*v2), NodeType::LogOneParentArgument);
                return f64ad::f64ad_locked_var(res);

            }
            (f64ad::f64(v1), f64ad::f64ad_locked_var(v2)) => {
                let res = f64ad_locked_var_operation_one_parent(v2, Some(*v1), NodeType::LogOneParentBase);
                return f64ad::f64ad_locked_var(res);            }
        }
    }
    fn log2(self) -> Self { return self.log(f64ad::f64(2.0)) }
    fn log10(self) -> Self { return self.log(f64ad::f64(10.0))}
    fn ln(self) -> Self { return self.log(f64ad::f64(std::f64::consts::E)) }
    fn ln_1p(self) -> Self { todo!() }
    fn sqrt(self) -> Self {
        return match &self {
            f64ad::f64ad_var(f) => {
                let res = f.computation_graph.add_node(NodeType::Sqrt, tiny_vec!([u32; 2] => f.node_idx), tiny_vec!([f64; 1]));
                f64ad::f64ad_var(res)
            }
            f64ad::f64ad_locked_var(f) => {
                f64ad::f64ad_locked_var(f64ad_locked_var_operation_one_parent(f, None, NodeType::Sqrt))
            }
            f64ad::f64(f) => {
                f64ad::f64(f.sqrt())
            }
        }
    }
    fn exp(self) -> Self {
        return match &self {
            f64ad::f64ad_var(f) => {
                let res = f.computation_graph.add_node(NodeType::Exp, tiny_vec!([u32; 2] => f.node_idx), tiny_vec!([f64; 1]));
                f64ad::f64ad_var(res)
            }
            f64ad::f64ad_locked_var(f) => {
                f64ad::f64ad_locked_var(f64ad_locked_var_operation_one_parent(f, None, NodeType::Exp))
            }
            f64ad::f64(f) => {
                f64ad::f64(f.exp())
            }
        }
    }
    fn exp2(self) -> Self { todo!() }
    fn exp_m1(self) -> Self { return self.exp() - 1.0; }
    fn powi(self, n: i32) -> Self { return self.powf(f64ad::f64(n as f64)) }
    fn powf(self, n: Self::RealField) -> Self {
        match (&self, &n) {
            (f64ad::f64ad_var(v1), f64ad::f64ad_var(v2)) => {
                assert_eq!(v1.computation_graph_id, v2.computation_graph_id);
                let res = v1.computation_graph.add_node(NodeType::PowTwoParents, tiny_vec!([u32; 2] => v1.node_idx, v2.node_idx), tiny_vec!([f64; 1]));
                return f64ad::f64ad_var(res);
            }
            (f64ad::f64ad_locked_var(v1), f64ad::f64ad_locked_var(v2)) => {
                let res = f64ad_locked_var_operation_two_parents(v1, v2, NodeType::PowTwoParents);
                return f64ad::f64ad_locked_var(res);
            }
            (f64ad::f64(v1), f64ad::f64(v2)) => {
                return f64ad::f64(v1.powf(*v2))
            }
            (f64ad::f64ad_var(_), f64ad::f64ad_locked_var(_)) => { panic!("unsupported.") }
            (f64ad::f64ad_locked_var(_), f64ad::f64ad_var(_)) => { panic!("unsupported.") }
            (f64ad::f64ad_var(v1), f64ad::f64(v2)) => {
                let res = v1.computation_graph.add_node(NodeType::PowOneParentArgument, tiny_vec!([u32; 2] => v1.node_idx), tiny_vec!([f64; 1] => *v2));
                return f64ad::f64ad_var(res);
            }
            (f64ad::f64(v1), f64ad::f64ad_var(v2)) => {
                let res = v2.computation_graph.add_node(NodeType::PowOneParentExponent, tiny_vec!([u32; 2] => v2.node_idx), tiny_vec!([f64; 1] => *v1));
                return f64ad::f64ad_var(res);
            }
            (f64ad::f64ad_locked_var(v1), f64ad::f64(v2)) => {
                let res = f64ad_locked_var_operation_one_parent(v1, Some(*v2), NodeType::PowOneParentArgument);
                return f64ad::f64ad_locked_var(res);
            }
            (f64ad::f64(v1), f64ad::f64ad_locked_var(v2)) => {
                let res = f64ad_locked_var_operation_one_parent(v2, Some(*v1), NodeType::PowOneParentExponent);
                return f64ad::f64ad_locked_var(res);            }
        }
    }
    fn powc(self, n: Self) -> Self { return self.powf(n); }
    fn cbrt(self) -> Self { return self.powf(f64ad::f64(1.0/3.0)) }
    fn is_finite(&self) -> bool {
        match self {
            f64ad::f64ad_var(f) => {
                match &f.mode {
                    ComputationGraphMode::Standard => { return f.value().is_finite() }
                    ComputationGraphMode::Lock => { panic!("cannot call is_finite on computation graph of mode Lock.  Computation graph {}", f.computation_graph_id); }
                }
            }
            f64ad::f64ad_locked_var(_) => { panic!("cannot call is_finite on f64ad_locked_var."); }
            f64ad::f64(f) => { return f.is_finite() }
        }
    }
    fn try_sqrt(self) -> Option<Self> {
        Some(self.sqrt())
    }
}

impl SubsetOf<f64ad> for f64ad {
    fn to_superset(&self) -> f64ad {
        self.clone()
    }

    fn from_superset_unchecked(element: &f64ad) -> Self {
        element.clone()
    }

    fn is_in_subset(_element: &f64ad) -> bool {
        true
    }
}

impl Field for f64ad { }

impl PrimitiveSimdValue for f64ad { }

impl SubsetOf<f64ad> for f32 {
    fn to_superset(&self) -> f64ad {
        f64ad::f64(*self as f64)
    }

    fn from_superset_unchecked(element: &f64ad) -> Self {
        element.value() as f32
    }

    fn is_in_subset(_: &f64ad) -> bool {
        false
    }
}

impl SubsetOf<f64ad> for f64 {
    fn to_superset(&self) -> f64ad {
        f64ad::f64(*self as f64)
    }

    fn from_superset_unchecked(element: &f64ad) -> Self {
        element.value()
    }

    fn is_in_subset(_: &f64ad) -> bool {
        false
    }
}

impl SubsetOf<f64ad> for u32 {
    fn to_superset(&self) -> f64ad {
        f64ad::f64(*self as f64)
    }

    fn from_superset_unchecked(element: &f64ad) -> Self {
        element.value() as u32
    }

    fn is_in_subset(_: &f64ad) -> bool {
        false
    }
}

impl SubsetOf<f64ad> for u64 {
    fn to_superset(&self) -> f64ad {
        f64ad::f64(*self as f64)
    }

    fn from_superset_unchecked(element: &f64ad) -> Self {
        element.value() as u64
    }

    fn is_in_subset(_: &f64ad) -> bool {
        false
    }
}

impl SubsetOf<f64ad> for u128 {
    fn to_superset(&self) -> f64ad {
        f64ad::f64(*self as f64)
    }

    fn from_superset_unchecked(element: &f64ad) -> Self {
        element.value() as u128
    }

    fn is_in_subset(_: &f64ad) -> bool {
        false
    }
}

impl SubsetOf<f64ad> for i32 {
    fn to_superset(&self) -> f64ad {
        f64ad::f64(*self as f64)
    }

    fn from_superset_unchecked(element: &f64ad) -> Self {
        element.value() as i32
    }

    fn is_in_subset(_: &f64ad) -> bool {
        false
    }
}

impl SubsetOf<f64ad> for i64 {
    fn to_superset(&self) -> f64ad {
        f64ad::f64(*self as f64)
    }

    fn from_superset_unchecked(element: &f64ad) -> Self {
        element.value() as i64
    }

    fn is_in_subset(_: &f64ad) -> bool {
        false
    }
}

impl SubsetOf<f64ad> for i128 {
    fn to_superset(&self) -> f64ad {
        f64ad::f64(*self as f64)
    }

    fn from_superset_unchecked(element: &f64ad) -> Self {
        element.value() as i128
    }

    fn is_in_subset(_: &f64ad) -> bool {
        false
    }
}
