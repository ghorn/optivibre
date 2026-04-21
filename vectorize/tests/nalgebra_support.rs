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

#[derive(Clone, Debug, PartialEq)]
struct Tagged<const N: usize, T> {
    dimension: nalgebra::Const<N>,
    value: T,
}

impl<T, const N: usize> Vectorize<T> for Tagged<N, T>
where
    T: vectorize::ScalarLeaf,
{
    type Rebind<U: vectorize::ScalarLeaf> = Tagged<N, U>;
    type View<'a>
        = Tagged<N, &'a T>
    where
        T: 'a;

    const LEN: usize = 1;

    fn flatten_refs<'a>(&'a self, out: &mut Vec<&'a T>) {
        out.push(&self.value);
    }

    fn from_flat_fn<U: vectorize::ScalarLeaf>(f: &mut impl FnMut() -> U) -> Self::Rebind<U> {
        Tagged {
            dimension: nalgebra::Const,
            value: f(),
        }
    }

    fn view<'a>(&'a self) -> Self::View<'a>
    where
        T: 'a,
    {
        Tagged {
            dimension: nalgebra::Const,
            value: &self.value,
        }
    }

    fn view_from_flat_slice<'a>(slice: &'a [T], index: &mut usize) -> Self::View<'a>
    where
        T: 'a,
    {
        let value = &slice[*index];
        *index += 1;
        Tagged {
            dimension: nalgebra::Const,
            value,
        }
    }

    fn flat_layout_names(prefix: &str, out: &mut Vec<String>) {
        out.push(vectorize::extend_layout_name(prefix, "value"));
    }
}

#[test]
fn nalgebra_const_dimension_marker_is_zero_length_and_composes() {
    let tagged = Tagged::<7, f64> {
        dimension: nalgebra::Const,
        value: 3.5,
    };
    assert_eq!(flatten_value(&tagged), vec![3.5]);

    let rebuilt = unflatten_value::<Tagged<11, f64>, _>(&[9.0]).expect("roundtrip should work");
    assert_eq!(
        rebuilt,
        Tagged {
            dimension: nalgebra::Const,
            value: 9.0,
        }
    );

    let layout_names = <Tagged<5, f64> as Vectorize<f64>>::layout_names().flatten_cloned();
    assert_eq!(layout_names, vec!["value"]);
}
