use crate::{ScalarLeaf, Vectorize, extend_layout_name};
use nalgebra::{ArrayStorage, Quaternion, SVector};

impl<T, const D: usize> Vectorize<T> for SVector<T, D>
where
    T: ScalarLeaf,
{
    type Rebind<U: ScalarLeaf> = SVector<U, D>;
    type View<'a>
        = SVector<&'a T, D>
    where
        T: 'a;

    const LEN: usize = D;

    fn flatten_refs<'a>(&'a self, out: &mut Vec<&'a T>) {
        out.extend(self.iter());
    }

    fn from_flat_fn<U: ScalarLeaf>(f: &mut impl FnMut() -> U) -> Self::Rebind<U> {
        SVector::<U, D>::from_array_storage(ArrayStorage([std::array::from_fn(|_| f())]))
    }

    fn view<'a>(&'a self) -> Self::View<'a>
    where
        T: 'a,
    {
        SVector::<&'a T, D>::from_array_storage(ArrayStorage([std::array::from_fn(|row| {
            &self[row]
        })]))
    }

    fn view_from_flat_slice<'a>(slice: &'a [T], index: &mut usize) -> Self::View<'a>
    where
        T: 'a,
    {
        SVector::<&'a T, D>::from_array_storage(ArrayStorage([std::array::from_fn(|_| {
            let value = &slice[*index];
            *index += 1;
            value
        })]))
    }

    fn flat_layout_names(prefix: &str, out: &mut Vec<String>) {
        for index in 0..D {
            let component = format!("[{index}]");
            out.push(extend_layout_name(prefix, &component));
        }
    }
}

impl<T> Vectorize<T> for Quaternion<T>
where
    T: ScalarLeaf,
{
    type Rebind<U: ScalarLeaf> = Quaternion<U>;
    type View<'a>
        = Quaternion<&'a T>
    where
        T: 'a;

    const LEN: usize = 4;

    fn flatten_refs<'a>(&'a self, out: &mut Vec<&'a T>) {
        out.extend(self.coords.iter());
    }

    fn from_flat_fn<U: ScalarLeaf>(f: &mut impl FnMut() -> U) -> Self::Rebind<U> {
        Quaternion {
            coords: SVector::<U, 4>::from_array_storage(ArrayStorage([[f(), f(), f(), f()]])),
        }
    }

    fn view<'a>(&'a self) -> Self::View<'a>
    where
        T: 'a,
    {
        Quaternion {
            coords: SVector::<&'a T, 4>::from_array_storage(ArrayStorage([[
                &self.coords[0],
                &self.coords[1],
                &self.coords[2],
                &self.coords[3],
            ]])),
        }
    }

    fn view_from_flat_slice<'a>(slice: &'a [T], index: &mut usize) -> Self::View<'a>
    where
        T: 'a,
    {
        Quaternion {
            coords: SVector::<&'a T, 4>::from_array_storage(ArrayStorage([[
                {
                    let value = &slice[*index];
                    *index += 1;
                    value
                },
                {
                    let value = &slice[*index];
                    *index += 1;
                    value
                },
                {
                    let value = &slice[*index];
                    *index += 1;
                    value
                },
                {
                    let value = &slice[*index];
                    *index += 1;
                    value
                },
            ]])),
        }
    }

    fn flat_layout_names(prefix: &str, out: &mut Vec<String>) {
        out.push(extend_layout_name(prefix, "i"));
        out.push(extend_layout_name(prefix, "j"));
        out.push(extend_layout_name(prefix, "k"));
        out.push(extend_layout_name(prefix, "w"));
    }
}
