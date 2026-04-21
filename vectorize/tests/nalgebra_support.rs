#![cfg(feature = "nalgebra")]

use nalgebra::{Quaternion, Vector3, Vector6};
use vectorize::{Vectorize, flat_view, flatten_value, unflatten_value};

#[test]
fn nalgebra_vector_aliases_roundtrip_through_const_generic_impl() {
    let vector = Vector3::new(1.0, 2.0, 3.0);
    assert_eq!(flatten_value(&vector), vec![1.0, 2.0, 3.0]);

    let rebuilt = unflatten_value::<Vector6<f64>, _>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("flat slice should rebuild vector aliases");
    assert_eq!(rebuilt, Vector6::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0));

    let view =
        flat_view::<Vector3<f64>, f64>(&[8.0, 13.0, 21.0]).expect("flat view should succeed");
    assert_eq!(view[0], &8.0);
    assert_eq!(view[1], &13.0);
    assert_eq!(view[2], &21.0);
}

#[test]
fn nalgebra_quaternion_uses_i_j_k_w_layout() {
    let quaternion = Quaternion::new(4.0, 1.0, 2.0, 3.0);
    assert_eq!(flatten_value(&quaternion), vec![1.0, 2.0, 3.0, 4.0]);

    let rebuilt = unflatten_value::<Quaternion<f64>, _>(&[5.0, 6.0, 7.0, 8.0])
        .expect("flat slice should rebuild quaternion");
    assert_eq!(rebuilt, Quaternion::new(8.0, 5.0, 6.0, 7.0));

    let layout_names = <Quaternion<f64> as Vectorize<f64>>::layout_names().flatten_cloned();
    assert_eq!(layout_names, vec!["i", "j", "k", "w"]);
}
