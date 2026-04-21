use sx_core::{SX, SXMatrix, SxError};
use thiserror::Error;

pub use vectorize_derive::Vectorize;

#[cfg(feature = "nalgebra")]
mod nalgebra_support;

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum VectorizeLayoutError {
    #[error("flat layout length mismatch: expected {expected}, got {got}")]
    LengthMismatch { expected: usize, got: usize },
}

pub trait ScalarLeaf: Clone {}

impl ScalarLeaf for SX {}
impl ScalarLeaf for f64 {}
impl ScalarLeaf for Option<f64> {}
impl ScalarLeaf for String {}

#[doc(hidden)]
pub fn extend_layout_name(prefix: &str, component: &str) -> String {
    if prefix.is_empty() {
        component.to_string()
    } else if component.starts_with('[') {
        format!("{prefix}{component}")
    } else {
        format!("{prefix}.{component}")
    }
}

pub trait Vectorize<T: ScalarLeaf>: Sized {
    type Rebind<U: ScalarLeaf>;
    type View<'a>
    where
        Self: 'a,
        T: 'a;

    const LEN: usize;

    fn flatten_refs<'a>(&'a self, out: &mut Vec<&'a T>);

    fn from_flat_fn<U: ScalarLeaf>(f: &mut impl FnMut() -> U) -> Self::Rebind<U>;

    fn view<'a>(&'a self) -> Self::View<'a>
    where
        Self: 'a,
        T: 'a;

    fn view_from_flat_slice<'a>(slice: &'a [T], index: &mut usize) -> Self::View<'a>
    where
        T: 'a;

    fn layout_names() -> Self::Rebind<String> {
        Self::layout_names_with_prefix("")
    }

    fn layout_names_with_prefix(prefix: &str) -> Self::Rebind<String> {
        let mut names = Vec::with_capacity(Self::LEN);
        Self::flat_layout_names(prefix, &mut names);
        let mut names = names.into_iter();
        Self::from_flat_fn(&mut || {
            names
                .next()
                .expect("layout_names should produce exactly one label per flat entry")
        })
    }

    fn flat_layout_names(prefix: &str, out: &mut Vec<String>);

    fn flatten_cloned(&self) -> Vec<T> {
        let mut refs = Vec::with_capacity(Self::LEN);
        self.flatten_refs(&mut refs);
        refs.into_iter().cloned().collect()
    }

    fn from_flat_slice(values: &[T]) -> Result<Self::Rebind<T>, VectorizeLayoutError>
    where
        T: Clone,
    {
        if values.len() != Self::LEN {
            return Err(VectorizeLayoutError::LengthMismatch {
                expected: Self::LEN,
                got: values.len(),
            });
        }
        let mut index = 0usize;
        Ok(Self::from_flat_fn(&mut || {
            let value = values[index].clone();
            index += 1;
            value
        }))
    }
}

impl<T: ScalarLeaf> Vectorize<T> for T {
    type Rebind<U: ScalarLeaf> = U;
    type View<'a>
        = &'a T
    where
        T: 'a;

    const LEN: usize = 1;

    fn flatten_refs<'a>(&'a self, out: &mut Vec<&'a T>) {
        out.push(self);
    }

    fn from_flat_fn<U: ScalarLeaf>(f: &mut impl FnMut() -> U) -> Self::Rebind<U> {
        f()
    }

    fn view<'a>(&'a self) -> Self::View<'a>
    where
        T: 'a,
    {
        self
    }

    fn view_from_flat_slice<'a>(slice: &'a [T], index: &mut usize) -> Self::View<'a>
    where
        T: 'a,
    {
        let value = &slice[*index];
        *index += 1;
        value
    }

    fn flat_layout_names(prefix: &str, out: &mut Vec<String>) {
        out.push(prefix.to_string());
    }
}

impl<T: ScalarLeaf> Vectorize<T> for () {
    type Rebind<U: ScalarLeaf> = ();
    type View<'a>
        = ()
    where
        T: 'a;

    const LEN: usize = 0;

    fn flatten_refs<'a>(&'a self, _out: &mut Vec<&'a T>) {}

    fn from_flat_fn<U: ScalarLeaf>(_f: &mut impl FnMut() -> U) -> Self::Rebind<U> {}

    fn view<'a>(&'a self) -> Self::View<'a>
    where
        T: 'a,
    {
    }

    fn view_from_flat_slice<'a>(_slice: &'a [T], _index: &mut usize) -> Self::View<'a>
    where
        T: 'a,
    {
    }

    fn flat_layout_names(_prefix: &str, _out: &mut Vec<String>) {}
}

impl<T, V, const N: usize> Vectorize<T> for [V; N]
where
    T: ScalarLeaf,
    V: Vectorize<T>,
{
    type Rebind<U: ScalarLeaf> = [V::Rebind<U>; N];
    type View<'a>
        = [V::View<'a>; N]
    where
        T: 'a,
        V: 'a;

    const LEN: usize = N * V::LEN;

    fn flatten_refs<'a>(&'a self, out: &mut Vec<&'a T>) {
        for value in self {
            value.flatten_refs(out);
        }
    }

    fn from_flat_fn<U: ScalarLeaf>(f: &mut impl FnMut() -> U) -> Self::Rebind<U> {
        std::array::from_fn(|_| V::from_flat_fn::<U>(f))
    }

    fn view<'a>(&'a self) -> Self::View<'a>
    where
        T: 'a,
        V: 'a,
    {
        std::array::from_fn(|idx| self[idx].view())
    }

    fn view_from_flat_slice<'a>(slice: &'a [T], index: &mut usize) -> Self::View<'a>
    where
        T: 'a,
        V: 'a,
    {
        std::array::from_fn(|_| V::view_from_flat_slice(slice, index))
    }

    fn flat_layout_names(prefix: &str, out: &mut Vec<String>) {
        for index in 0..N {
            let component = format!("[{index}]");
            V::flat_layout_names(&extend_layout_name(prefix, &component), out);
        }
    }
}

impl<T, A, B> Vectorize<T> for (A, B)
where
    T: ScalarLeaf,
    A: Vectorize<T>,
    B: Vectorize<T>,
{
    type Rebind<U: ScalarLeaf> = (A::Rebind<U>, B::Rebind<U>);
    type View<'a>
        = (A::View<'a>, B::View<'a>)
    where
        T: 'a,
        A: 'a,
        B: 'a;

    const LEN: usize = A::LEN + B::LEN;

    fn flatten_refs<'a>(&'a self, out: &mut Vec<&'a T>) {
        self.0.flatten_refs(out);
        self.1.flatten_refs(out);
    }

    fn from_flat_fn<U: ScalarLeaf>(f: &mut impl FnMut() -> U) -> Self::Rebind<U> {
        (A::from_flat_fn(f), B::from_flat_fn(f))
    }

    fn view<'a>(&'a self) -> Self::View<'a>
    where
        T: 'a,
        A: 'a,
        B: 'a,
    {
        (self.0.view(), self.1.view())
    }

    fn view_from_flat_slice<'a>(slice: &'a [T], index: &mut usize) -> Self::View<'a>
    where
        T: 'a,
        A: 'a,
        B: 'a,
    {
        (
            A::view_from_flat_slice(slice, index),
            B::view_from_flat_slice(slice, index),
        )
    }

    fn flat_layout_names(prefix: &str, out: &mut Vec<String>) {
        A::flat_layout_names(&extend_layout_name(prefix, "[0]"), out);
        B::flat_layout_names(&extend_layout_name(prefix, "[1]"), out);
    }
}

impl<T, A, B, C> Vectorize<T> for (A, B, C)
where
    T: ScalarLeaf,
    A: Vectorize<T>,
    B: Vectorize<T>,
    C: Vectorize<T>,
{
    type Rebind<U: ScalarLeaf> = (A::Rebind<U>, B::Rebind<U>, C::Rebind<U>);
    type View<'a>
        = (A::View<'a>, B::View<'a>, C::View<'a>)
    where
        T: 'a,
        A: 'a,
        B: 'a,
        C: 'a;

    const LEN: usize = A::LEN + B::LEN + C::LEN;

    fn flatten_refs<'a>(&'a self, out: &mut Vec<&'a T>) {
        self.0.flatten_refs(out);
        self.1.flatten_refs(out);
        self.2.flatten_refs(out);
    }

    fn from_flat_fn<U: ScalarLeaf>(f: &mut impl FnMut() -> U) -> Self::Rebind<U> {
        (A::from_flat_fn(f), B::from_flat_fn(f), C::from_flat_fn(f))
    }

    fn view<'a>(&'a self) -> Self::View<'a>
    where
        T: 'a,
        A: 'a,
        B: 'a,
        C: 'a,
    {
        (self.0.view(), self.1.view(), self.2.view())
    }

    fn view_from_flat_slice<'a>(slice: &'a [T], index: &mut usize) -> Self::View<'a>
    where
        T: 'a,
        A: 'a,
        B: 'a,
        C: 'a,
    {
        (
            A::view_from_flat_slice(slice, index),
            B::view_from_flat_slice(slice, index),
            C::view_from_flat_slice(slice, index),
        )
    }

    fn flat_layout_names(prefix: &str, out: &mut Vec<String>) {
        A::flat_layout_names(&extend_layout_name(prefix, "[0]"), out);
        B::flat_layout_names(&extend_layout_name(prefix, "[1]"), out);
        C::flat_layout_names(&extend_layout_name(prefix, "[2]"), out);
    }
}

impl<T, A, B, C, D> Vectorize<T> for (A, B, C, D)
where
    T: ScalarLeaf,
    A: Vectorize<T>,
    B: Vectorize<T>,
    C: Vectorize<T>,
    D: Vectorize<T>,
{
    type Rebind<U: ScalarLeaf> = (A::Rebind<U>, B::Rebind<U>, C::Rebind<U>, D::Rebind<U>);
    type View<'a>
        = (A::View<'a>, B::View<'a>, C::View<'a>, D::View<'a>)
    where
        T: 'a,
        A: 'a,
        B: 'a,
        C: 'a,
        D: 'a;

    const LEN: usize = A::LEN + B::LEN + C::LEN + D::LEN;

    fn flatten_refs<'a>(&'a self, out: &mut Vec<&'a T>) {
        self.0.flatten_refs(out);
        self.1.flatten_refs(out);
        self.2.flatten_refs(out);
        self.3.flatten_refs(out);
    }

    fn from_flat_fn<U: ScalarLeaf>(f: &mut impl FnMut() -> U) -> Self::Rebind<U> {
        (
            A::from_flat_fn(f),
            B::from_flat_fn(f),
            C::from_flat_fn(f),
            D::from_flat_fn(f),
        )
    }

    fn view<'a>(&'a self) -> Self::View<'a>
    where
        T: 'a,
        A: 'a,
        B: 'a,
        C: 'a,
        D: 'a,
    {
        (self.0.view(), self.1.view(), self.2.view(), self.3.view())
    }

    fn view_from_flat_slice<'a>(slice: &'a [T], index: &mut usize) -> Self::View<'a>
    where
        T: 'a,
        A: 'a,
        B: 'a,
        C: 'a,
        D: 'a,
    {
        (
            A::view_from_flat_slice(slice, index),
            B::view_from_flat_slice(slice, index),
            C::view_from_flat_slice(slice, index),
            D::view_from_flat_slice(slice, index),
        )
    }

    fn flat_layout_names(prefix: &str, out: &mut Vec<String>) {
        A::flat_layout_names(&extend_layout_name(prefix, "[0]"), out);
        B::flat_layout_names(&extend_layout_name(prefix, "[1]"), out);
        C::flat_layout_names(&extend_layout_name(prefix, "[2]"), out);
        D::flat_layout_names(&extend_layout_name(prefix, "[3]"), out);
    }
}

impl<T, A, B, C, D, E> Vectorize<T> for (A, B, C, D, E)
where
    T: ScalarLeaf,
    A: Vectorize<T>,
    B: Vectorize<T>,
    C: Vectorize<T>,
    D: Vectorize<T>,
    E: Vectorize<T>,
{
    type Rebind<U: ScalarLeaf> = (
        A::Rebind<U>,
        B::Rebind<U>,
        C::Rebind<U>,
        D::Rebind<U>,
        E::Rebind<U>,
    );
    type View<'a>
        = (
        A::View<'a>,
        B::View<'a>,
        C::View<'a>,
        D::View<'a>,
        E::View<'a>,
    )
    where
        T: 'a,
        A: 'a,
        B: 'a,
        C: 'a,
        D: 'a,
        E: 'a;

    const LEN: usize = A::LEN + B::LEN + C::LEN + D::LEN + E::LEN;

    fn flatten_refs<'a>(&'a self, out: &mut Vec<&'a T>) {
        self.0.flatten_refs(out);
        self.1.flatten_refs(out);
        self.2.flatten_refs(out);
        self.3.flatten_refs(out);
        self.4.flatten_refs(out);
    }

    fn from_flat_fn<U: ScalarLeaf>(f: &mut impl FnMut() -> U) -> Self::Rebind<U> {
        (
            A::from_flat_fn(f),
            B::from_flat_fn(f),
            C::from_flat_fn(f),
            D::from_flat_fn(f),
            E::from_flat_fn(f),
        )
    }

    fn view<'a>(&'a self) -> Self::View<'a>
    where
        T: 'a,
        A: 'a,
        B: 'a,
        C: 'a,
        D: 'a,
        E: 'a,
    {
        (
            self.0.view(),
            self.1.view(),
            self.2.view(),
            self.3.view(),
            self.4.view(),
        )
    }

    fn view_from_flat_slice<'a>(slice: &'a [T], index: &mut usize) -> Self::View<'a>
    where
        T: 'a,
        A: 'a,
        B: 'a,
        C: 'a,
        D: 'a,
        E: 'a,
    {
        (
            A::view_from_flat_slice(slice, index),
            B::view_from_flat_slice(slice, index),
            C::view_from_flat_slice(slice, index),
            D::view_from_flat_slice(slice, index),
            E::view_from_flat_slice(slice, index),
        )
    }

    fn flat_layout_names(prefix: &str, out: &mut Vec<String>) {
        A::flat_layout_names(&extend_layout_name(prefix, "[0]"), out);
        B::flat_layout_names(&extend_layout_name(prefix, "[1]"), out);
        C::flat_layout_names(&extend_layout_name(prefix, "[2]"), out);
        D::flat_layout_names(&extend_layout_name(prefix, "[3]"), out);
        E::flat_layout_names(&extend_layout_name(prefix, "[4]"), out);
    }
}

impl<T, A, B, C, D, E, F> Vectorize<T> for (A, B, C, D, E, F)
where
    T: ScalarLeaf,
    A: Vectorize<T>,
    B: Vectorize<T>,
    C: Vectorize<T>,
    D: Vectorize<T>,
    E: Vectorize<T>,
    F: Vectorize<T>,
{
    type Rebind<U: ScalarLeaf> = (
        A::Rebind<U>,
        B::Rebind<U>,
        C::Rebind<U>,
        D::Rebind<U>,
        E::Rebind<U>,
        F::Rebind<U>,
    );
    type View<'a>
        = (
        A::View<'a>,
        B::View<'a>,
        C::View<'a>,
        D::View<'a>,
        E::View<'a>,
        F::View<'a>,
    )
    where
        T: 'a,
        A: 'a,
        B: 'a,
        C: 'a,
        D: 'a,
        E: 'a,
        F: 'a;

    const LEN: usize = A::LEN + B::LEN + C::LEN + D::LEN + E::LEN + F::LEN;

    fn flatten_refs<'a>(&'a self, out: &mut Vec<&'a T>) {
        self.0.flatten_refs(out);
        self.1.flatten_refs(out);
        self.2.flatten_refs(out);
        self.3.flatten_refs(out);
        self.4.flatten_refs(out);
        self.5.flatten_refs(out);
    }

    fn from_flat_fn<U: ScalarLeaf>(f: &mut impl FnMut() -> U) -> Self::Rebind<U> {
        (
            A::from_flat_fn(f),
            B::from_flat_fn(f),
            C::from_flat_fn(f),
            D::from_flat_fn(f),
            E::from_flat_fn(f),
            F::from_flat_fn(f),
        )
    }

    fn view<'a>(&'a self) -> Self::View<'a>
    where
        T: 'a,
        A: 'a,
        B: 'a,
        C: 'a,
        D: 'a,
        E: 'a,
        F: 'a,
    {
        (
            self.0.view(),
            self.1.view(),
            self.2.view(),
            self.3.view(),
            self.4.view(),
            self.5.view(),
        )
    }

    fn view_from_flat_slice<'a>(slice: &'a [T], index: &mut usize) -> Self::View<'a>
    where
        T: 'a,
        A: 'a,
        B: 'a,
        C: 'a,
        D: 'a,
        E: 'a,
        F: 'a,
    {
        (
            A::view_from_flat_slice(slice, index),
            B::view_from_flat_slice(slice, index),
            C::view_from_flat_slice(slice, index),
            D::view_from_flat_slice(slice, index),
            E::view_from_flat_slice(slice, index),
            F::view_from_flat_slice(slice, index),
        )
    }

    fn flat_layout_names(prefix: &str, out: &mut Vec<String>) {
        A::flat_layout_names(&extend_layout_name(prefix, "[0]"), out);
        B::flat_layout_names(&extend_layout_name(prefix, "[1]"), out);
        C::flat_layout_names(&extend_layout_name(prefix, "[2]"), out);
        D::flat_layout_names(&extend_layout_name(prefix, "[3]"), out);
        E::flat_layout_names(&extend_layout_name(prefix, "[4]"), out);
        F::flat_layout_names(&extend_layout_name(prefix, "[5]"), out);
    }
}

pub fn symbolic_value<T>(prefix: &str) -> Result<T, SxError>
where
    T: Vectorize<SX, Rebind<SX> = T>,
{
    let mut index = 0usize;
    Ok(T::from_flat_fn(&mut || {
        let name = if T::LEN == 1 {
            prefix.to_string()
        } else {
            let current = index;
            index += 1;
            format!("{prefix}_{current}")
        };
        SX::sym(name)
    }))
}

pub fn symbolic_column<T>(value: &T) -> Result<SXMatrix, SxError>
where
    T: Vectorize<SX>,
{
    SXMatrix::dense_column(value.flatten_cloned())
}

pub fn flatten_value<T>(value: &T) -> Vec<f64>
where
    T: Vectorize<f64>,
{
    value.flatten_cloned()
}

pub fn flatten_optional_value<T>(value: &T) -> Vec<Option<f64>>
where
    T: Vectorize<Option<f64>>,
{
    value.flatten_cloned()
}

pub fn unflatten_value<S, T>(values: &[T]) -> Result<S::Rebind<T>, VectorizeLayoutError>
where
    S: Vectorize<T>,
    T: ScalarLeaf + Clone,
{
    S::from_flat_slice(values)
}

pub fn rebind_from_flat<S, T, U>(values: &[U]) -> Result<S::Rebind<U>, VectorizeLayoutError>
where
    S: Vectorize<T>,
    T: ScalarLeaf,
    U: ScalarLeaf + Clone,
{
    if values.len() != S::LEN {
        return Err(VectorizeLayoutError::LengthMismatch {
            expected: S::LEN,
            got: values.len(),
        });
    }
    let mut index = 0usize;
    Ok(S::from_flat_fn(&mut || {
        let value = values[index].clone();
        index += 1;
        value
    }))
}

pub fn flat_view<'a, S, T>(values: &'a [T]) -> Result<S::View<'a>, VectorizeLayoutError>
where
    S: Vectorize<T>,
    T: ScalarLeaf,
{
    if values.len() != S::LEN {
        return Err(VectorizeLayoutError::LengthMismatch {
            expected: S::LEN,
            got: values.len(),
        });
    }
    let mut index = 0usize;
    Ok(S::view_from_flat_slice(values, &mut index))
}
