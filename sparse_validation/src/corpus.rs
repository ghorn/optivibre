use std::collections::BTreeSet;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use flate2::read::GzDecoder;
use metis_ordering::CsrGraph;
use tar::Archive;

use crate::matrix_market::parse_matrix_market_file;
use crate::model::{CorpusCaseMetadata, CorpusSourceKind, CorpusTag, ValidationTier};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymmetricPatternMatrix {
    dimension: usize,
    col_ptrs: Vec<usize>,
    row_indices: Vec<usize>,
}

impl SymmetricPatternMatrix {
    pub fn from_undirected_edges(dimension: usize, edges: &[(usize, usize)]) -> Result<Self> {
        let mut positions = edges
            .iter()
            .copied()
            .flat_map(|(lhs, rhs)| [(lhs, rhs), (rhs, lhs)])
            .collect::<Vec<_>>();
        positions.extend((0..dimension).map(|index| (index, index)));
        Self::from_coordinate_entries(dimension, &positions, false)
    }

    pub fn from_coordinate_entries(
        dimension: usize,
        entries: &[(usize, usize)],
        symmetric_input: bool,
    ) -> Result<Self> {
        let mut cols = vec![BTreeSet::new(); dimension];
        for (index, column) in cols.iter_mut().enumerate() {
            column.insert(index);
        }
        for &(row, col) in entries {
            if row >= dimension || col >= dimension {
                bail!("entry ({row}, {col}) out of bounds for {dimension}x{dimension}");
            }
            let pairs = if symmetric_input || row == col {
                vec![(row, col)]
            } else {
                vec![(row, col), (col, row)]
            };
            for (raw_row, raw_col) in pairs {
                let (row, col) = if raw_row >= raw_col {
                    (raw_row, raw_col)
                } else {
                    (raw_col, raw_row)
                };
                cols[col].insert(row);
            }
        }
        let mut col_ptrs = Vec::with_capacity(dimension + 1);
        let mut row_indices = Vec::new();
        col_ptrs.push(0);
        for rows in cols {
            row_indices.extend(rows);
            col_ptrs.push(row_indices.len());
        }
        Ok(Self {
            dimension,
            col_ptrs,
            row_indices,
        })
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn nnz(&self) -> usize {
        self.row_indices.len()
    }

    pub fn col_ptrs(&self) -> &[usize] {
        &self.col_ptrs
    }

    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    pub fn to_graph(&self) -> Result<CsrGraph> {
        Ok(CsrGraph::from_symmetric_csc(
            self.dimension,
            &self.col_ptrs,
            &self.row_indices,
        )?)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DownloadedCorpusSpec {
    pub id: &'static str,
    pub description: &'static str,
    pub url: &'static str,
    pub tags: &'static [CorpusTag],
}

#[derive(Clone, Debug)]
pub struct LoadedCorpusCase {
    pub metadata: CorpusCaseMetadata,
    pub matrix: SymmetricPatternMatrix,
}

pub type SkippedCorpusEntries = Vec<(String, String)>;
pub type LoadedTierCases = (Vec<LoadedCorpusCase>, SkippedCorpusEntries);

enum CorpusSpec {
    Generated {
        metadata: GeneratedCaseMetadata,
        matrix: SymmetricPatternMatrix,
    },
    Downloaded(DownloadedCorpusSpec),
}

#[derive(Clone, Debug)]
struct GeneratedCaseMetadata {
    id: &'static str,
    description: &'static str,
    tags: Vec<CorpusTag>,
    exact_oracle: bool,
    protected_fill_regression: bool,
}

pub fn downloaded_public_corpus_specs() -> &'static [DownloadedCorpusSpec] {
    &[
        DownloadedCorpusSpec {
            id: "hb_can_24",
            description: "Public structural HB/can_24 benchmark",
            url: "https://suitesparse-collection-website.herokuapp.com/MM/HB/can_24.tar.gz",
            tags: &[
                CorpusTag::Public,
                CorpusTag::Structural,
                CorpusTag::TinyExact,
            ],
        },
        DownloadedCorpusSpec {
            id: "hb_bcsstk01",
            description: "Public finite-element HB/bcsstk01 benchmark",
            url: "https://suitesparse-collection-website.herokuapp.com/MM/HB/bcsstk01.tar.gz",
            tags: &[
                CorpusTag::Public,
                CorpusTag::FiniteElement,
                CorpusTag::Structural,
            ],
        },
        DownloadedCorpusSpec {
            id: "hb_bcsstk05",
            description: "Public finite-element HB/bcsstk05 benchmark",
            url: "https://suitesparse-collection-website.herokuapp.com/MM/HB/bcsstk05.tar.gz",
            tags: &[
                CorpusTag::Public,
                CorpusTag::FiniteElement,
                CorpusTag::Structural,
            ],
        },
        DownloadedCorpusSpec {
            id: "boeing_bcsstk34",
            description: "Public Boeing/bcsstk34 benchmark",
            url: "https://suitesparse-collection-website.herokuapp.com/MM/Boeing/bcsstk34.tar.gz",
            tags: &[
                CorpusTag::Public,
                CorpusTag::FiniteElement,
                CorpusTag::Structural,
            ],
        },
    ]
}

pub fn corpus_download_target_path(root: &Path, spec: &DownloadedCorpusSpec) -> PathBuf {
    root.join(format!("{}.mtx", spec.id))
}

pub fn extract_downloaded_corpus_archive(
    archive_path: &Path,
    destination_root: &Path,
    spec: &DownloadedCorpusSpec,
) -> Result<PathBuf> {
    fs::create_dir_all(destination_root)?;
    let file = fs::File::open(archive_path)?;
    let decoder = GzDecoder::new(file);
    let mut archive = Archive::new(decoder);
    let target = corpus_download_target_path(destination_root, spec);
    let mut extracted = None;

    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        let is_matrix_market = path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("mtx"));
        if !is_matrix_market {
            continue;
        }

        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes)?;
        fs::write(&target, bytes)?;
        extracted = Some(target.clone());
        break;
    }

    extracted.ok_or_else(|| {
        anyhow::anyhow!(
            "archive {} did not contain a Matrix Market file for {}",
            archive_path.display(),
            spec.id
        )
    })
}

pub fn load_cases_for_tier(tier: ValidationTier, corpus_root: &Path) -> Result<LoadedTierCases> {
    let mut loaded = generated_specs();
    if tier != ValidationTier::Pr {
        loaded.extend(
            downloaded_public_corpus_specs()
                .iter()
                .cloned()
                .map(CorpusSpec::Downloaded),
        );
    }

    let mut cases = Vec::new();
    let mut skipped = Vec::new();
    for spec in loaded {
        match spec {
            CorpusSpec::Generated { metadata, matrix } => cases.push(LoadedCorpusCase {
                metadata: CorpusCaseMetadata {
                    case_id: metadata.id.to_string(),
                    description: metadata.description.to_string(),
                    source: CorpusSourceKind::Generated,
                    dimension: matrix.dimension(),
                    nnz: matrix.nnz(),
                    tags: metadata.tags,
                    exact_oracle: metadata.exact_oracle,
                    protected_fill_regression: metadata.protected_fill_regression,
                    location: None,
                },
                matrix,
            }),
            CorpusSpec::Downloaded(spec) => {
                let path = corpus_download_target_path(corpus_root, &spec);
                if !path.exists() {
                    skipped.push((
                        spec.id.to_string(),
                        format!("downloaded corpus file missing at {}", path.display()),
                    ));
                    continue;
                }
                let matrix = parse_matrix_market_file(&path)?;
                cases.push(LoadedCorpusCase {
                    metadata: CorpusCaseMetadata {
                        case_id: spec.id.to_string(),
                        description: spec.description.to_string(),
                        source: CorpusSourceKind::Downloaded,
                        dimension: matrix.dimension(),
                        nnz: matrix.nnz(),
                        tags: spec.tags.to_vec(),
                        exact_oracle: matrix.dimension() <= 32,
                        protected_fill_regression: false,
                        location: Some(path),
                    },
                    matrix,
                });
            }
        }
    }
    Ok((cases, skipped))
}

fn generated_specs() -> Vec<CorpusSpec> {
    vec![
        generated_case(
            "path_9",
            "Nine-node path graph",
            vec![CorpusTag::Path, CorpusTag::Structural, CorpusTag::TinyExact],
            path_edges(9),
            9,
            true,
            true,
        ),
        generated_case(
            "grid_3x3",
            "Three-by-three grid graph",
            vec![CorpusTag::Grid, CorpusTag::Structural, CorpusTag::TinyExact],
            grid_edges(3, 3),
            9,
            true,
            true,
        ),
        generated_case(
            "binary_tree_15",
            "Balanced binary tree with 15 nodes",
            vec![CorpusTag::Tree, CorpusTag::Structural, CorpusTag::TinyExact],
            binary_tree_edges(15),
            15,
            true,
            true,
        ),
        generated_case(
            "disconnected_paths",
            "Two disconnected path components",
            vec![
                CorpusTag::Disconnected,
                CorpusTag::Structural,
                CorpusTag::TinyExact,
            ],
            vec![(0, 1), (1, 2), (3, 4), (4, 5)],
            6,
            true,
            true,
        ),
        generated_case(
            "banded_spd_12",
            "Twelve-by-twelve banded SPD structure with bandwidth two",
            vec![
                CorpusTag::BandedSpd,
                CorpusTag::Structural,
                CorpusTag::TinyExact,
            ],
            banded_edges(12, 2),
            12,
            true,
            true,
        ),
        generated_case(
            "arrow_kkt_10",
            "Arrow/KKT-style star structure",
            vec![
                CorpusTag::ArrowKkt,
                CorpusTag::Structural,
                CorpusTag::TinyExact,
            ],
            arrow_edges(10),
            10,
            true,
            true,
        ),
        generated_case(
            "saddle_kkt_8",
            "Eight-node saddle-point KKT structure with three constraints",
            vec![
                CorpusTag::ArrowKkt,
                CorpusTag::Structural,
                CorpusTag::TinyExact,
            ],
            saddle_kkt_edges(),
            8,
            true,
            true,
        ),
        generated_case(
            "banded_kkt_12",
            "Twelve-node banded KKT structure with four constraints",
            vec![
                CorpusTag::ArrowKkt,
                CorpusTag::Structural,
                CorpusTag::TinyExact,
            ],
            banded_kkt_edges(),
            12,
            true,
            true,
        ),
        generated_case(
            "coupled_indefinite_3",
            "Three-node coupled indefinite panel requiring a 2x2 pivot",
            vec![
                CorpusTag::TwoByTwoPivot,
                CorpusTag::Adversarial,
                CorpusTag::Structural,
                CorpusTag::TinyExact,
            ],
            vec![(0, 1), (0, 2), (1, 2)],
            3,
            true,
            true,
        ),
        generated_case(
            "delayed_propagation_6",
            "Six-node indefinite chain requiring delayed pivot propagation across fronts",
            vec![
                CorpusTag::DelayedPivot,
                CorpusTag::Adversarial,
                CorpusTag::Structural,
                CorpusTag::TinyExact,
            ],
            vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
            6,
            true,
            true,
        ),
        generated_case(
            "ladder_14",
            "Ladder graph adversarial separator case",
            vec![
                CorpusTag::Adversarial,
                CorpusTag::Structural,
                CorpusTag::TinyExact,
            ],
            ladder_edges(7),
            14,
            true,
            false,
        ),
    ]
}

fn generated_case(
    id: &'static str,
    description: &'static str,
    tags: Vec<CorpusTag>,
    edges: Vec<(usize, usize)>,
    dimension: usize,
    exact_oracle: bool,
    protected_fill_regression: bool,
) -> CorpusSpec {
    let matrix = SymmetricPatternMatrix::from_undirected_edges(dimension, &edges)
        .expect("generated corpus matrix should be valid");
    CorpusSpec::Generated {
        metadata: GeneratedCaseMetadata {
            id,
            description,
            tags,
            exact_oracle,
            protected_fill_regression,
        },
        matrix,
    }
}

fn path_edges(size: usize) -> Vec<(usize, usize)> {
    (0..size.saturating_sub(1))
        .map(|index| (index, index + 1))
        .collect()
}

fn grid_edges(rows: usize, cols: usize) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    for row in 0..rows {
        for col in 0..cols {
            let index = row * cols + col;
            if row + 1 < rows {
                edges.push((index, index + cols));
            }
            if col + 1 < cols {
                edges.push((index, index + 1));
            }
        }
    }
    edges
}

fn binary_tree_edges(size: usize) -> Vec<(usize, usize)> {
    (1..size).map(|child| ((child - 1) / 2, child)).collect()
}

fn banded_edges(size: usize, bandwidth: usize) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    for row in 0..size {
        for delta in 1..=bandwidth {
            if row + delta < size {
                edges.push((row, row + delta));
            }
        }
    }
    edges
}

fn arrow_edges(size: usize) -> Vec<(usize, usize)> {
    let hub = size - 1;
    (0..hub).map(|index| (index, hub)).collect()
}

fn saddle_kkt_edges() -> Vec<(usize, usize)> {
    vec![
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (1, 5),
        (2, 6),
        (3, 6),
        (1, 7),
        (4, 7),
    ]
}

fn banded_kkt_edges() -> Vec<(usize, usize)> {
    vec![
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 3),
        (2, 4),
        (3, 4),
        (3, 5),
        (4, 5),
        (4, 6),
        (5, 6),
        (5, 7),
        (6, 7),
        (0, 8),
        (1, 8),
        (2, 8),
        (2, 9),
        (3, 9),
        (4, 9),
        (4, 10),
        (5, 10),
        (6, 10),
        (1, 11),
        (6, 11),
        (7, 11),
    ]
}

fn ladder_edges(rungs: usize) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    for rung in 0..rungs {
        let top = rung;
        let bottom = rung + rungs;
        edges.push((top, bottom));
        if rung + 1 < rungs {
            edges.push((top, top + 1));
            edges.push((bottom, bottom + 1));
        }
    }
    edges
}
