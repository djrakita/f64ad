use num_traits::{FromPrimitive, Num, One, Signed, Zero};
use tinyvec::tiny_vec;
use crate::f64ad2::{f64ad, f64ad_locked_var_operation_one_parent, f64ad_var, NodeType};

impl Zero for f64ad {
    fn zero() -> Self {
        return f64ad::f64(0.0);
    }

    fn is_zero(&self) -> bool {
        return self.value() == 0.0
    }
}

impl One for f64ad {
    fn one() -> Self {
        Self::f64(1.0)
    }
}

impl Num for f64ad {
    type FromStrRadixErr = ();

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        let val = f64::from_str_radix(str, radix).expect("error");
        Ok(Self::f64(val))
    }
}

impl Signed for f64ad {
    fn abs(&self) -> Self {
        return match self {
            f64ad::f64ad_var(f) => {
                let res = unsafe {
                    (*f.computation_graph.0).add_node(NodeType::Abs, tiny_vec!([u32; 2] => f.node_idx), tiny_vec!([f64; 1]))
                };
                f64ad::f64ad_var(res)
            }
            f64ad::f64ad_locked_var(f) => {
                f64ad::f64ad_locked_var(f64ad_locked_var_operation_one_parent(f, None, NodeType::Abs))
            }
            f64ad::f64(f) => {
                f64ad::f64(f.abs())
            }
        }
    }

    fn abs_sub(&self, other: &Self) -> Self {
        return if *self <= *other {
            f64ad::f64(0.0)
        } else {
            *self - *other
        }
    }

    fn signum(&self) -> Self {
        return match self {
            f64ad::f64ad_var(f) => {
                let res = unsafe {
                    (*f.computation_graph.0).add_node(NodeType::Signum, tiny_vec!([u32; 2] => f.node_idx), tiny_vec!([f64; 1]))
                };
                f64ad::f64ad_var(res)
            }
            f64ad::f64ad_locked_var(f) => {
                f64ad::f64ad_locked_var(f64ad_locked_var_operation_one_parent(f, None, NodeType::Signum))
            }
            f64ad::f64(f) => {
                f64ad::f64(f.signum())
            }
        }
    }

    fn is_positive(&self) -> bool {
        return self.value() > 0.0;
    }

    fn is_negative(&self) -> bool {
        return self.value() < 0.0;
    }
}

impl FromPrimitive for f64ad {
    fn from_i64(n: i64) -> Option<Self> {
        Some(f64ad::f64(n as f64))
    }

    fn from_u64(n: u64) -> Option<Self> {
        Some(f64ad::f64(n as f64))
    }
}