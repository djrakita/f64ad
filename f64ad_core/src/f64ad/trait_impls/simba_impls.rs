use num_traits::{Signed, Zero};
use simba::scalar::{ComplexField, Field, RealField, SubsetOf};
use simba::simd::PrimitiveSimdValue;
use crate::f64ad::{f64ad, f64ad_universal_function_1_operand, f64ad_universal_function_2_operands, NodeTypeClass};

impl RealField for f64ad {
    fn is_sign_positive(&self) -> bool {
        return self.is_positive();
    }

    fn is_sign_negative(&self) -> bool {
        return self.is_negative();
    }

    fn copysign(self, sign: Self) -> Self {
        return if sign.is_positive() {
            self.abs()
        } else {
            -self.abs()
        };
    }

    fn max(self, other: Self) -> Self {
        f64ad_universal_function_2_operands(self, other, NodeTypeClass::Max)
    }

    fn min(self, other: Self) -> Self {
        f64ad_universal_function_2_operands(self, other, NodeTypeClass::Min)
    }

    fn clamp(self, min: Self, max: Self) -> Self {
        assert!(min <= max);
        return self.max(min).min(max);
    }

    fn atan2(self, other: Self) -> Self {
        f64ad_universal_function_2_operands(self, other, NodeTypeClass::Atan2)
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

    fn modulus(self) -> Self::RealField { return self.abs(); }

    fn modulus_squared(self) -> Self::RealField { self * self }

    fn argument(self) -> Self::RealField { unimplemented!(); }

    fn norm1(self) -> Self::RealField { return self.abs(); }

    fn scale(self, factor: Self::RealField) -> Self { return self * factor; }

    fn unscale(self, factor: Self::RealField) -> Self { return self / factor; }

    fn floor(self) -> Self {
        f64ad_universal_function_1_operand(self, NodeTypeClass::Floor)
    }

    fn ceil(self) -> Self {
        f64ad_universal_function_1_operand(self, NodeTypeClass::Ceil)
    }

    fn round(self) -> Self {
        f64ad_universal_function_1_operand(self, NodeTypeClass::Round)
    }

    fn trunc(self) -> Self {
        f64ad_universal_function_1_operand(self, NodeTypeClass::Trunc)
    }

    fn fract(self) -> Self {
        f64ad_universal_function_1_operand(self, NodeTypeClass::Fract)
    }

    fn mul_add(self, a: Self, b: Self) -> Self { return (self * a) + b; }

    fn abs(self) -> Self::RealField {
        <Self as Signed>::abs(&self)
    }

    fn hypot(self, other: Self) -> Self::RealField {
        return (self.powi(2) + other.powi(2)).sqrt();
    }

    fn recip(self) -> Self { return 1.0 / self; }

    fn conjugate(self) -> Self { return self; }

    fn sin(self) -> Self {
        f64ad_universal_function_1_operand(self, NodeTypeClass::Sin)
    }

    fn cos(self) -> Self {
        f64ad_universal_function_1_operand(self, NodeTypeClass::Cos)
    }

    fn sin_cos(self) -> (Self, Self) {
        return (self.sin(), self.cos());
    }

    fn tan(self) -> Self { f64ad_universal_function_1_operand(self, NodeTypeClass::Tan) }

    fn asin(self) -> Self { f64ad_universal_function_1_operand(self, NodeTypeClass::Asin) }

    fn acos(self) -> Self {
        f64ad_universal_function_1_operand(self, NodeTypeClass::Acos)
    }

    fn atan(self) -> Self {
        f64ad_universal_function_1_operand(self, NodeTypeClass::Atan)
    }

    fn sinh(self) -> Self {
        f64ad_universal_function_1_operand(self, NodeTypeClass::Sinh)
    }

    fn cosh(self) -> Self {
        f64ad_universal_function_1_operand(self, NodeTypeClass::Cosh)
    }

    fn tanh(self) -> Self {
        f64ad_universal_function_1_operand(self, NodeTypeClass::Tanh)
    }

    fn asinh(self) -> Self { f64ad_universal_function_1_operand(self, NodeTypeClass::Asinh) }

    fn acosh(self) -> Self {
        f64ad_universal_function_1_operand(self, NodeTypeClass::Acosh)
    }

    fn atanh(self) -> Self { f64ad_universal_function_1_operand(self, NodeTypeClass::Atanh) }

    fn log(self, base: Self::RealField) -> Self { f64ad_universal_function_2_operands(self, base, NodeTypeClass::Log) }

    fn log2(self) -> Self { return self.log(f64ad::f64(2.0)); }

    fn log10(self) -> Self { return self.log(f64ad::f64(10.0)); }

    fn ln(self) -> Self { return self.log(f64ad::f64(std::f64::consts::E)); }

    fn ln_1p(self) -> Self { (1.0 + self).ln() }

    fn sqrt(self) -> Self { f64ad_universal_function_1_operand(self, NodeTypeClass::Sqrt) }

    fn exp(self) -> Self { f64ad_universal_function_1_operand(self, NodeTypeClass::Exp) }

    fn exp2(self) -> Self { f64ad::f64(2.0).powf(self) }

    fn exp_m1(self) -> Self { return self.exp() - 1.0; }

    fn powi(self, n: i32) -> Self { return self.powf(f64ad::f64(n as f64)); }

    fn powf(self, n: Self::RealField) -> Self { f64ad_universal_function_2_operands(self, n, NodeTypeClass::Powf) }

    fn powc(self, n: Self) -> Self { return self.powf(n); }

    fn cbrt(self) -> Self { return self.powf(f64ad::f64(1.0 / 3.0)); }

    fn is_finite(&self) -> bool { return self.value().is_finite(); }

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

impl Field for f64ad {}

impl PrimitiveSimdValue for f64ad {}

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