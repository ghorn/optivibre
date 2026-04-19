use nalgebra::{Quaternion, Vector3};
use optimization::{Vectorize, flat_view, flatten_value, symbolic_value, unflatten_value};
use sx_core::SX;

#[derive(Clone, Debug, PartialEq, optimization::Vectorize)]
struct Pose<T> {
    position: Vector3<T>,
    attitude: Quaternion<T>,
}

#[test]
fn nalgebra_layout_roundtrips_numeric_values() {
    let value = Pose {
        position: Vector3::new(1.0, -2.0, 3.5),
        attitude: Quaternion::new(0.5, 0.25, -0.75, 1.25),
    };
    let flat = flatten_value(&value);
    assert_eq!(flat, vec![1.0, -2.0, 3.5, 0.5, 0.25, -0.75, 1.25]);
    let rebuilt = unflatten_value::<Pose<f64>, f64>(&flat).expect("layout should unflatten");
    assert_eq!(rebuilt, value);
}

#[test]
fn nalgebra_layout_names_match_component_order() {
    let layout = Pose::<f64>::layout_names();
    assert_eq!(layout.position.x, "position.x");
    assert_eq!(layout.position.y, "position.y");
    assert_eq!(layout.position.z, "position.z");
    assert_eq!(layout.attitude.w, "attitude.w");
    assert_eq!(layout.attitude.i, "attitude.i");
    assert_eq!(layout.attitude.j, "attitude.j");
    assert_eq!(layout.attitude.k, "attitude.k");
}

#[test]
fn nalgebra_layout_builds_symbolic_values_and_views() {
    let symbolic = symbolic_value::<Pose<SX>>("state").expect("symbolic value should build");
    let flat = flatten_value(&symbolic);
    assert_eq!(flat.len(), Pose::<SX>::LEN);
    let view = flat_view::<Pose<f64>, f64>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        .expect("flat view should succeed");
    assert_eq!(*view.position.x, 1.0);
    assert_eq!(*view.attitude.k, 7.0);
}
