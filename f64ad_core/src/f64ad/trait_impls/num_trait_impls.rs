use num_traits::{Bounded, FromPrimitive, Num, One, Signed, Zero};
use crate::f64ad::{f64ad, f64ad_universal_function_1_operand, NodeTypeClass};

impl Zero for f64ad {
    #[cfg_attr(feature = "inline_on", inline)]
    #[cfg_attr(feature = "inline_always_on", inline(always))]
    fn zero() -> Self {
        return f64ad::f64(0.0);
    }

    #[cfg_attr(feature = "inline_on", inline)]
    #[cfg_attr(feature = "inline_always_on", inline(always))]
    fn is_zero(&self) -> bool {
        return self.value() == 0.0;
    }
}

impl One for f64ad {
    #[cfg_attr(feature = "inline_on", inline)]
    #[cfg_attr(feature = "inline_always_on", inline(always))]
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
    #[cfg_attr(feature = "inline_on", inline)]
    #[cfg_attr(feature = "inline_always_on", inline(always))]
    fn abs(&self) -> Self {
        f64ad_universal_function_1_operand(*self, NodeTypeClass::Abs)
    }

    fn abs_sub(&self, other: &Self) -> Self {
        return if *self <= *other {
            f64ad::f64(0.0)
        } else {
            *self - *other
        };
    }

    fn signum(&self) -> Self {
        f64ad_universal_function_1_operand(*self, NodeTypeClass::Signum)
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

impl Bounded for f64ad {
    fn min_value() -> Self {
        Self::f64(f64::MIN)
    }

    fn max_value() -> Self {
        Self::f64(f64::MAX)
    }
}