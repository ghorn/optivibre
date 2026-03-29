pub type Index = usize;
pub type SignedIndex = isize;

pub fn checked_len_product(lhs: Index, rhs: Index) -> Option<usize> {
    lhs.checked_mul(rhs)
}
