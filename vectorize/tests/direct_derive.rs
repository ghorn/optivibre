use vectorize::{Vectorize, flat_view, flatten_value, unflatten_value};

#[derive(Clone, Debug, PartialEq, Vectorize)]
struct Pair<T> {
    x: T,
    y: T,
}

#[test]
fn standalone_crate_supports_derive_and_roundtrip_helpers() {
    let value = Pair { x: 1.0, y: -2.0 };
    assert_eq!(flatten_value(&value), vec![1.0, -2.0]);

    let rebuilt = unflatten_value::<Pair<f64>, _>(&[3.5, 4.5]).expect("flat slice should rebind");
    assert_eq!(rebuilt, Pair { x: 3.5, y: 4.5 });

    let view: PairView<'_, f64> =
        flat_view::<Pair<f64>, f64>(&[8.0, 13.0]).expect("flat view should succeed");
    assert_eq!(*view.x, 8.0);
    assert_eq!(*view.y, 13.0);
}
