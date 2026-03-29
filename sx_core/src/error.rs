use thiserror::Error;

pub type Result<T> = std::result::Result<T, SxError>;

#[derive(Debug, Error)]
pub enum SxError {
    #[error("shape mismatch: {0}")]
    Shape(String),
    #[error("invalid CCS: {0}")]
    Ccs(String),
    #[error("invalid symbolic graph: {0}")]
    Graph(String),
    #[error("unsupported operation: {0}")]
    Unsupported(String),
}
