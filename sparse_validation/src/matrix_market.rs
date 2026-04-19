use std::fs;
use std::path::Path;

use thiserror::Error;

use crate::corpus::SymmetricPatternMatrix;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MatrixSymmetry {
    General,
    Symmetric,
}

#[derive(Debug, Error)]
pub enum MatrixMarketError {
    #[error("unsupported Matrix Market format: {0}")]
    UnsupportedFormat(String),
    #[error("invalid Matrix Market file: {0}")]
    Invalid(String),
    #[error("failed to read Matrix Market file `{path}`: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

pub fn parse_matrix_market_file(path: &Path) -> Result<SymmetricPatternMatrix, MatrixMarketError> {
    let content = fs::read_to_string(path).map_err(|source| MatrixMarketError::Io {
        path: path.display().to_string(),
        source,
    })?;
    parse_matrix_market_str(&content)
}

pub fn parse_matrix_market_str(content: &str) -> Result<SymmetricPatternMatrix, MatrixMarketError> {
    let mut lines = content.lines();
    let header = lines
        .next()
        .ok_or_else(|| MatrixMarketError::Invalid("missing Matrix Market header".into()))?;
    let header_tokens = header.split_whitespace().collect::<Vec<_>>();
    if header_tokens.len() != 5
        || !header_tokens[0].eq_ignore_ascii_case("%%MatrixMarket")
        || !header_tokens[1].eq_ignore_ascii_case("matrix")
        || !header_tokens[2].eq_ignore_ascii_case("coordinate")
    {
        return Err(MatrixMarketError::UnsupportedFormat(header.to_string()));
    }
    let symmetry = match header_tokens[4].to_ascii_lowercase().as_str() {
        "general" => MatrixSymmetry::General,
        "symmetric" => MatrixSymmetry::Symmetric,
        other => {
            return Err(MatrixMarketError::UnsupportedFormat(format!(
                "unsupported symmetry mode `{other}`"
            )));
        }
    };

    let mut shape_line = None;
    for line in lines.by_ref() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('%') {
            continue;
        }
        shape_line = Some(trimmed.to_string());
        break;
    }
    let shape_line =
        shape_line.ok_or_else(|| MatrixMarketError::Invalid("missing shape line".into()))?;
    let shape_tokens = shape_line.split_whitespace().collect::<Vec<_>>();
    if shape_tokens.len() < 3 {
        return Err(MatrixMarketError::Invalid(format!(
            "invalid shape line `{shape_line}`"
        )));
    }
    let nrow = parse_usize(shape_tokens[0], "row count")?;
    let ncol = parse_usize(shape_tokens[1], "column count")?;
    let nnz = parse_usize(shape_tokens[2], "nnz")?;
    if nrow != ncol {
        return Err(MatrixMarketError::Invalid(format!(
            "expected square matrix, got {nrow}x{ncol}"
        )));
    }

    let mut entries = Vec::with_capacity(nnz);
    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('%') {
            continue;
        }
        let tokens = trimmed.split_whitespace().collect::<Vec<_>>();
        if tokens.len() < 2 {
            return Err(MatrixMarketError::Invalid(format!(
                "invalid entry line `{trimmed}`"
            )));
        }
        let row = parse_usize(tokens[0], "row index")?;
        let col = parse_usize(tokens[1], "column index")?;
        if row == 0 || col == 0 {
            return Err(MatrixMarketError::Invalid(format!(
                "Matrix Market indices are 1-based, got ({row}, {col})"
            )));
        }
        entries.push((row - 1, col - 1));
    }
    SymmetricPatternMatrix::from_coordinate_entries(
        nrow,
        &entries,
        symmetry == MatrixSymmetry::Symmetric,
    )
    .map_err(|message| MatrixMarketError::Invalid(message.to_string()))
}

fn parse_usize(token: &str, label: &str) -> Result<usize, MatrixMarketError> {
    token.parse::<usize>().map_err(|_| {
        MatrixMarketError::Invalid(format!(
            "invalid {label} token `{token}` in Matrix Market file"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_small_symmetric_coordinate_matrix() {
        let matrix = parse_matrix_market_str(
            "%%MatrixMarket matrix coordinate real symmetric\n\
             % comment\n\
             3 3 4\n\
             1 1 1.0\n\
             2 1 2.0\n\
             3 2 3.0\n\
             3 3 4.0\n",
        )
        .expect("matrix should parse");
        assert_eq!(matrix.dimension(), 3);
        assert_eq!(matrix.nnz(), 5);
    }
}
