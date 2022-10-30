use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use num_traits::Signed;
use crate::f64ad2::f64ad;

impl UlpsEq for f64ad {
    fn default_max_ulps() -> u32 {
        unimplemented!("take the time to figure this out.")
    }

    fn ulps_eq(&self, _other: &Self, _epsilon: Self::Epsilon, _max_ulps: u32) -> bool {
        unimplemented!("take the time to figure this out.")
    }
}

impl AbsDiffEq for f64ad {
    type Epsilon = f64ad;

    fn default_epsilon() -> Self::Epsilon {
        f64ad::f64(0.000000001)
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        let diff = *self - *other;
        if diff.abs() < epsilon {
            true
        } else {
            false
        }
    }
}

impl RelativeEq for f64ad {
    fn default_max_relative() -> Self::Epsilon {
        f64ad::f64(0.000000001)
    }

    fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, _max_relative: Self::Epsilon) -> bool {
        let diff = *self - *other;
        if diff.abs() < epsilon {
            true
        } else {
            false
        }
    }
}