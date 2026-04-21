#![cfg(feature = "nalgebra")]

use nalgebra::{Quaternion, Vector3};
use optimization::{Vectorize, flatten_value, unflatten_value};

#[test]
fn optimization_reexports_nalgebra_vectorize_impls() {
    let vector = Vector3::new(1.0, 2.0, 3.0);
    assert_eq!(flatten_value(&vector), vec![1.0, 2.0, 3.0]);

    let quaternion = unflatten_value::<Quaternion<f64>, _>(&[1.0, 2.0, 3.0, 4.0])
        .expect("optimization re-export should rebuild quaternion");
    assert_eq!(quaternion, Quaternion::new(4.0, 1.0, 2.0, 3.0));

    let layout_names = <Quaternion<f64> as Vectorize<f64>>::layout_names().flatten_cloned();
    assert_eq!(layout_names, vec!["i", "j", "k", "w"]);
}
