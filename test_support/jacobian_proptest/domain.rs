use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InputBoxFamily {
    SymmetricFinite,
    PositiveFinite,
    ShiftedFinite,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RangeCert {
    pub lower: f64,
    pub upper: f64,
    pub finite_guaranteed: bool,
    pub min_abs_lower_bound: f64,
}

impl RangeCert {
    pub fn exact(value: f64) -> Self {
        let abs = value.abs();
        Self {
            lower: value,
            upper: value,
            finite_guaranteed: value.is_finite(),
            min_abs_lower_bound: abs,
        }
    }

    pub fn any_finite(lower: f64, upper: f64) -> Self {
        let min_abs = if lower <= 0.0 && upper >= 0.0 {
            0.0
        } else {
            lower.abs().min(upper.abs())
        };
        Self {
            lower,
            upper,
            finite_guaranteed: lower.is_finite() && upper.is_finite(),
            min_abs_lower_bound: min_abs,
        }
    }

    pub fn is_positive_with_margin(&self, margin: f64) -> bool {
        self.finite_guaranteed && self.lower >= margin
    }

    pub fn is_nonzero_with_margin(&self, margin: f64) -> bool {
        self.finite_guaranteed && self.min_abs_lower_bound >= margin
    }

    pub fn within_abs_cap(&self, cap: f64) -> bool {
        self.finite_guaranteed && self.lower.abs() <= cap && self.upper.abs() <= cap
    }
}

impl fmt::Display for RangeCert {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{:.6}, {:.6}] finite={} min_abs_lb={:.6}",
            self.lower, self.upper, self.finite_guaranteed, self.min_abs_lower_bound
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct InputBox {
    pub intervals: Vec<RangeCert>,
}

impl InputBox {
    pub fn new(intervals: Vec<RangeCert>) -> Self {
        Self { intervals }
    }

    pub fn len(&self) -> usize {
        self.intervals.len()
    }

    pub fn is_empty(&self) -> bool {
        self.intervals.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<RangeCert> {
        self.intervals.get(index).copied()
    }
}

impl fmt::Display for InputBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (index, interval) in self.intervals.iter().enumerate() {
            if index > 0 {
                write!(f, ", ")?;
            }
            write!(f, "x[{index}]={interval}")?;
        }
        write!(f, "]")
    }
}

pub fn family_interval(family: InputBoxFamily) -> RangeCert {
    match family {
        InputBoxFamily::SymmetricFinite => RangeCert::any_finite(-0.75, 0.75),
        InputBoxFamily::PositiveFinite => RangeCert::any_finite(0.2, 1.2),
        InputBoxFamily::ShiftedFinite => RangeCert::any_finite(-1.2, -0.2),
    }
}

pub fn negate(cert: RangeCert) -> Option<RangeCert> {
    if !cert.finite_guaranteed {
        return None;
    }
    Some(RangeCert::any_finite(-cert.upper, -cert.lower))
}

pub fn add(lhs: RangeCert, rhs: RangeCert) -> Option<RangeCert> {
    if !(lhs.finite_guaranteed && rhs.finite_guaranteed) {
        return None;
    }
    Some(RangeCert::any_finite(
        lhs.lower + rhs.lower,
        lhs.upper + rhs.upper,
    ))
}

pub fn sub(lhs: RangeCert, rhs: RangeCert) -> Option<RangeCert> {
    if !(lhs.finite_guaranteed && rhs.finite_guaranteed) {
        return None;
    }
    Some(RangeCert::any_finite(
        lhs.lower - rhs.upper,
        lhs.upper - rhs.lower,
    ))
}

pub fn mul(lhs: RangeCert, rhs: RangeCert) -> Option<RangeCert> {
    if !(lhs.finite_guaranteed && rhs.finite_guaranteed) {
        return None;
    }
    let candidates = [
        lhs.lower * rhs.lower,
        lhs.lower * rhs.upper,
        lhs.upper * rhs.lower,
        lhs.upper * rhs.upper,
    ];
    let lower = candidates.into_iter().fold(f64::INFINITY, f64::min);
    let upper = candidates.into_iter().fold(f64::NEG_INFINITY, f64::max);
    Some(RangeCert::any_finite(lower, upper))
}

pub fn square(cert: RangeCert) -> Option<RangeCert> {
    if !cert.finite_guaranteed {
        return None;
    }
    let candidates = [cert.lower * cert.lower, cert.upper * cert.upper];
    let upper = candidates.into_iter().fold(0.0_f64, f64::max);
    let lower = if cert.lower <= 0.0 && cert.upper >= 0.0 {
        0.0
    } else {
        candidates.into_iter().fold(f64::INFINITY, f64::min)
    };
    Some(RangeCert::any_finite(lower, upper))
}

pub fn sin(_: RangeCert) -> RangeCert {
    RangeCert::any_finite(-1.0, 1.0)
}

pub fn cos(_: RangeCert) -> RangeCert {
    RangeCert::any_finite(-1.0, 1.0)
}

pub fn exp(cert: RangeCert, input_cap: f64) -> Option<RangeCert> {
    if !cert.finite_guaranteed || cert.upper > input_cap {
        return None;
    }
    Some(RangeCert::any_finite(cert.lower.exp(), cert.upper.exp()))
}

pub fn sqrt(cert: RangeCert, margin: f64) -> Option<RangeCert> {
    if !cert.is_positive_with_margin(margin) {
        return None;
    }
    Some(RangeCert::any_finite(cert.lower.sqrt(), cert.upper.sqrt()))
}

pub fn log(cert: RangeCert, margin: f64) -> Option<RangeCert> {
    if !cert.is_positive_with_margin(margin) {
        return None;
    }
    Some(RangeCert::any_finite(cert.lower.ln(), cert.upper.ln()))
}

pub fn div(lhs: RangeCert, rhs: RangeCert, nonzero_margin: f64) -> Option<RangeCert> {
    if !(lhs.finite_guaranteed && rhs.is_nonzero_with_margin(nonzero_margin)) {
        return None;
    }
    let reciprocal = RangeCert::any_finite(1.0 / rhs.upper, 1.0 / rhs.lower);
    mul(lhs, reciprocal)
}
