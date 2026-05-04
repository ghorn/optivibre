use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use optimization::ScalarLeaf;
use std::ops::{Add, Div, Mul, Neg, Sub};
use sx_core::SX;

pub trait Scalar:
    ScalarLeaf
    + Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    fn zero() -> Self;
    fn one() -> Self;
    fn from_f64(value: f64) -> Self;
    fn sqrt(self) -> Self;
    fn atan2(self, rhs: Self) -> Self;
    fn min(self, rhs: Self) -> Self;
    fn max(self, rhs: Self) -> Self;
}

impl Scalar for f64 {
    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn from_f64(value: f64) -> Self {
        value
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn atan2(self, rhs: Self) -> Self {
        self.atan2(rhs)
    }

    fn min(self, rhs: Self) -> Self {
        self.min(rhs)
    }

    fn max(self, rhs: Self) -> Self {
        self.max(rhs)
    }
}

impl Scalar for SX {
    fn zero() -> Self {
        SX::from(0.0)
    }

    fn one() -> Self {
        SX::from(1.0)
    }

    fn from_f64(value: f64) -> Self {
        SX::from(value)
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn atan2(self, rhs: Self) -> Self {
        self.atan2(rhs)
    }

    fn min(self, rhs: Self) -> Self {
        self.min(rhs)
    }

    fn max(self, rhs: Self) -> Self {
        self.max(rhs)
    }
}

pub fn square<T: Scalar>(value: T) -> T {
    value * value
}

pub fn clamp<T: Scalar>(value: T, min_value: T, max_value: T) -> T {
    value.max(min_value).min(max_value)
}

pub fn dot<T: Scalar>(lhs: &Vector3<T>, rhs: &Vector3<T>) -> T {
    lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]
}

pub fn cross<T: Scalar>(lhs: &Vector3<T>, rhs: &Vector3<T>) -> Vector3<T> {
    Vector3::new(
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    )
}

pub fn scale<T: Scalar>(value: &Vector3<T>, scalar: T) -> Vector3<T> {
    Vector3::new(value[0] * scalar, value[1] * scalar, value[2] * scalar)
}

pub fn add<T: Scalar>(lhs: &Vector3<T>, rhs: &Vector3<T>) -> Vector3<T> {
    Vector3::new(lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2])
}

pub fn sub<T: Scalar>(lhs: &Vector3<T>, rhs: &Vector3<T>) -> Vector3<T> {
    Vector3::new(lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2])
}

pub fn norm_squared<T: Scalar>(value: &Vector3<T>) -> T {
    dot(value, value)
}

pub fn norm<T: Scalar>(value: &Vector3<T>) -> T {
    (norm_squared(value) + T::from_f64(1.0e-9)).sqrt()
}

pub fn norm_exact<T: Scalar>(value: &Vector3<T>) -> T {
    norm_squared(value).sqrt()
}

pub fn normalize_exact<T: Scalar>(value: &Vector3<T>) -> Vector3<T> {
    scale(value, T::one() / norm_exact(value))
}

pub fn quaternion_conjugate<T: Scalar>(quat: &Quaternion<T>) -> Quaternion<T> {
    Quaternion::new(
        quat.coords[3],
        -quat.coords[0],
        -quat.coords[1],
        -quat.coords[2],
    )
}

pub fn quaternion_multiply<T: Scalar>(lhs: &Quaternion<T>, rhs: &Quaternion<T>) -> Quaternion<T> {
    Quaternion::new(
        lhs.coords[3] * rhs.coords[3]
            - lhs.coords[0] * rhs.coords[0]
            - lhs.coords[1] * rhs.coords[1]
            - lhs.coords[2] * rhs.coords[2],
        lhs.coords[3] * rhs.coords[0]
            + lhs.coords[0] * rhs.coords[3]
            + lhs.coords[1] * rhs.coords[2]
            - lhs.coords[2] * rhs.coords[1],
        lhs.coords[3] * rhs.coords[1] - lhs.coords[0] * rhs.coords[2]
            + lhs.coords[1] * rhs.coords[3]
            + lhs.coords[2] * rhs.coords[0],
        lhs.coords[3] * rhs.coords[2] + lhs.coords[0] * rhs.coords[1]
            - lhs.coords[1] * rhs.coords[0]
            + lhs.coords[2] * rhs.coords[3],
    )
}

pub fn rotate_nav_to_body<T: Scalar>(quat_n2b: &Quaternion<T>, value_n: &Vector3<T>) -> Vector3<T> {
    let pure = Quaternion::new(T::zero(), value_n[0], value_n[1], value_n[2]);
    let rotated = quaternion_multiply(
        &quaternion_conjugate(quat_n2b),
        &quaternion_multiply(&pure, quat_n2b),
    );
    Vector3::new(rotated.coords[0], rotated.coords[1], rotated.coords[2])
}

pub fn rotate_body_to_nav<T: Scalar>(quat_n2b: &Quaternion<T>, value_b: &Vector3<T>) -> Vector3<T> {
    let pure = Quaternion::new(T::zero(), value_b[0], value_b[1], value_b[2]);
    let rotated = quaternion_multiply(
        quat_n2b,
        &quaternion_multiply(&pure, &quaternion_conjugate(quat_n2b)),
    );
    Vector3::new(rotated.coords[0], rotated.coords[1], rotated.coords[2])
}

pub fn nav_down_in_body<T: Scalar>(quat_n2b: &Quaternion<T>) -> Vector3<T> {
    rotate_nav_to_body(quat_n2b, &Vector3::new(T::zero(), T::zero(), T::one()))
}

pub fn roll_pitch_from_quat_n2b<T: Scalar>(quat_n2b: &Quaternion<T>) -> (T, T) {
    let down_b = nav_down_in_body(quat_n2b);
    let roll = down_b[1].atan2(down_b[2]);
    let pitch = (-down_b[0]).atan2((down_b[1] * down_b[1] + down_b[2] * down_b[2]).sqrt());
    (roll, pitch)
}

pub fn roll_angle_from_quat_n2b<T: Scalar>(quat_n2b: &Quaternion<T>) -> T {
    roll_pitch_from_quat_n2b(quat_n2b).0
}

pub fn pitch_angle_from_quat_n2b<T: Scalar>(quat_n2b: &Quaternion<T>) -> T {
    roll_pitch_from_quat_n2b(quat_n2b).1
}

pub fn control_roll_pitch_rad_from_quat_n2b(quat_n2b: &Quaternion<f64>) -> [f64; 2] {
    let (roll, pitch) = roll_pitch_from_quat_n2b(quat_n2b);
    [roll, pitch]
}

pub fn control_roll_pitch_deg_from_quat_n2b(quat_n2b: &Quaternion<f64>) -> [f64; 2] {
    let [roll, pitch] = control_roll_pitch_rad_from_quat_n2b(quat_n2b);
    [roll.to_degrees(), pitch.to_degrees()]
}

pub fn euler_rpy_deg_from_quat_n2b(quat_n2b: &Quaternion<f64>) -> [f64; 3] {
    let unit = UnitQuaternion::from_quaternion(*quat_n2b);
    let (roll, pitch, yaw) = unit.euler_angles();
    [roll.to_degrees(), pitch.to_degrees(), yaw.to_degrees()]
}

pub fn yaw_angle_from_quat_n2b(quat_n2b: &Quaternion<f64>) -> f64 {
    UnitQuaternion::from_quaternion(*quat_n2b).euler_angles().2
}

pub fn ddt_quat_n2b<T: Scalar>(quat_n2b: &Quaternion<T>, omega_b: &Vector3<T>) -> Quaternion<T> {
    let half = T::from_f64(0.5);
    Quaternion::new(
        half * (-quat_n2b.coords[0] * omega_b[0]
            - quat_n2b.coords[1] * omega_b[1]
            - quat_n2b.coords[2] * omega_b[2]),
        half * (quat_n2b.coords[3] * omega_b[0] - quat_n2b.coords[2] * omega_b[1]
            + quat_n2b.coords[1] * omega_b[2]),
        half * (quat_n2b.coords[2] * omega_b[0] + quat_n2b.coords[3] * omega_b[1]
            - quat_n2b.coords[0] * omega_b[2]),
        half * (-quat_n2b.coords[1] * omega_b[0]
            + quat_n2b.coords[0] * omega_b[1]
            + quat_n2b.coords[3] * omega_b[2]),
    )
}

pub fn smooth_enable<T: Scalar>(value: T) -> T {
    let t = clamp(value, T::zero(), T::one());
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t2 * t2;
    t4 / (T::one() - T::from_f64(4.0) * t + T::from_f64(6.0) * t2 - T::from_f64(4.0) * t3
        + T::from_f64(2.0) * t4)
}

pub fn wrap_angle(value: f64) -> f64 {
    let mut wrapped = value;
    while wrapped > std::f64::consts::PI {
        wrapped -= 2.0 * std::f64::consts::PI;
    }
    while wrapped < -std::f64::consts::PI {
        wrapped += 2.0 * std::f64::consts::PI;
    }
    wrapped
}

pub fn circular_mean(values: &[f64]) -> f64 {
    let mut sin_sum = 0.0;
    let mut cos_sum = 0.0;
    for value in values {
        sin_sum += value.sin();
        cos_sum += value.cos();
    }
    sin_sum.atan2(cos_sum)
}

pub fn yaw_quaternion_n2b(yaw: f64) -> Quaternion<f64> {
    *UnitQuaternion::from_euler_angles(0.0, 0.0, yaw).quaternion()
}

pub fn roll_yaw_quaternion_n2b(roll: f64, yaw: f64) -> Quaternion<f64> {
    *UnitQuaternion::from_euler_angles(roll, 0.0, yaw).quaternion()
}

pub fn zero_if_nan(value: f64) -> f64 {
    if value.is_nan() { 0.0 } else { value }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalized(quat: Quaternion<f64>) -> Quaternion<f64> {
        let norm = (quat.coords[3] * quat.coords[3]
            + quat.coords[0] * quat.coords[0]
            + quat.coords[1] * quat.coords[1]
            + quat.coords[2] * quat.coords[2])
            .sqrt();
        Quaternion::new(
            quat.coords[3] / norm,
            quat.coords[0] / norm,
            quat.coords[1] / norm,
            quat.coords[2] / norm,
        )
    }

    fn integrate_once(quat: Quaternion<f64>, omega_b: Vector3<f64>, dt: f64) -> Quaternion<f64> {
        let qdot = ddt_quat_n2b(&quat, &omega_b);
        normalized(Quaternion::new(
            quat.coords[3] + qdot.coords[3] * dt,
            quat.coords[0] + qdot.coords[0] * dt,
            quat.coords[1] + qdot.coords[1] * dt,
            quat.coords[2] + qdot.coords[2] * dt,
        ))
    }

    #[test]
    fn body_x_rate_changes_roll_independently_of_yaw() {
        let dt = 0.02;
        let omega_b = Vector3::new(1.0, 0.0, 0.0);
        for yaw in [
            -std::f64::consts::FRAC_PI_2,
            0.0,
            std::f64::consts::FRAC_PI_2,
        ] {
            let quat = yaw_quaternion_n2b(yaw);
            let next = integrate_once(quat, omega_b, dt);
            let (roll, pitch) = roll_pitch_from_quat_n2b(&next);
            assert!(
                (roll - dt).abs() < 1.0e-4,
                "yaw={yaw}: roll={roll}, expected {dt}"
            );
            assert!(
                pitch.abs() < 1.0e-4,
                "yaw={yaw}: body-x rate leaked into pitch={pitch}"
            );
        }
    }

    #[test]
    fn body_y_rate_changes_pitch_independently_of_yaw() {
        let dt = 0.02;
        let omega_b = Vector3::new(0.0, 1.0, 0.0);
        for yaw in [
            -std::f64::consts::FRAC_PI_2,
            0.0,
            std::f64::consts::FRAC_PI_2,
        ] {
            let quat = yaw_quaternion_n2b(yaw);
            let next = integrate_once(quat, omega_b, dt);
            let (roll, pitch) = roll_pitch_from_quat_n2b(&next);
            assert!(
                roll.abs() < 1.0e-4,
                "yaw={yaw}: body-y rate leaked into roll={roll}"
            );
            assert!(
                (pitch - dt).abs() < 1.0e-4,
                "yaw={yaw}: pitch={pitch}, expected {}",
                dt
            );
        }
    }

    #[test]
    fn roll_yaw_quaternion_preserves_bank_across_heading() {
        let expected_roll = 30.0_f64.to_radians();
        for yaw in [
            -std::f64::consts::FRAC_PI_2,
            0.0,
            std::f64::consts::FRAC_PI_2,
        ] {
            let quat = roll_yaw_quaternion_n2b(expected_roll, yaw);
            let (roll, pitch) = roll_pitch_from_quat_n2b(&quat);
            assert!(
                (roll - expected_roll).abs() < 1.0e-12,
                "yaw={yaw}: roll={roll}, expected {expected_roll}"
            );
            assert!(pitch.abs() < 1.0e-12, "yaw={yaw}: pitch={pitch}");
        }
    }
}
