use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use metis_ordering::NestedDissectionOptions;
use spral_ssids::{
    NumericFactorOptions, OrderingStrategy, SsidsOptions, SymmetricCscMatrix, analyse, factorize,
};

fn path_matrix(dimension: usize) -> (Vec<usize>, Vec<usize>) {
    let mut col_ptrs = Vec::with_capacity(dimension + 1);
    let mut row_indices = Vec::new();
    col_ptrs.push(0);
    for col in 0..dimension {
        row_indices.push(col);
        if col + 1 < dimension {
            row_indices.push(col + 1);
        }
        col_ptrs.push(row_indices.len());
    }
    (col_ptrs, row_indices)
}

fn grid_matrix(rows: usize, cols: usize) -> (Vec<usize>, Vec<usize>) {
    let dimension = rows * cols;
    let mut columns = vec![Vec::new(); dimension];
    for (index, column) in columns.iter_mut().enumerate() {
        column.push(index);
    }
    for row in 0..rows {
        for col in 0..cols {
            let index = row * cols + col;
            if row + 1 < rows {
                let neighbor = index + cols;
                columns[index].push(neighbor);
            }
            if col + 1 < cols {
                let neighbor = index + 1;
                columns[index].push(neighbor);
            }
        }
    }
    let mut col_ptrs = Vec::with_capacity(dimension + 1);
    let mut row_indices = Vec::new();
    col_ptrs.push(0);
    for mut rows in columns {
        rows.sort_unstable();
        rows.dedup();
        row_indices.extend(rows);
        col_ptrs.push(row_indices.len());
    }
    (col_ptrs, row_indices)
}

fn arrow_matrix(dimension: usize) -> (Vec<usize>, Vec<usize>) {
    let hub = dimension - 1;
    let mut columns = vec![Vec::new(); dimension];
    for (index, column) in columns.iter_mut().enumerate() {
        column.push(index);
    }
    for column in columns.iter_mut().take(hub) {
        column.push(hub);
    }
    let mut col_ptrs = Vec::with_capacity(dimension + 1);
    let mut row_indices = Vec::new();
    col_ptrs.push(0);
    for mut rows in columns {
        rows.sort_unstable();
        rows.dedup();
        row_indices.extend(rows);
        col_ptrs.push(row_indices.len());
    }
    (col_ptrs, row_indices)
}

fn symbolic_cases() -> Vec<(&'static str, Vec<usize>, Vec<usize>)> {
    vec![
        {
            let (col_ptrs, row_indices) = path_matrix(256);
            ("path_256", col_ptrs, row_indices)
        },
        {
            let (col_ptrs, row_indices) = grid_matrix(18, 18);
            ("grid_18x18", col_ptrs, row_indices)
        },
        {
            let (col_ptrs, row_indices) = arrow_matrix(192);
            ("arrow_192", col_ptrs, row_indices)
        },
    ]
}

fn synthetic_values(dimension: usize, col_ptrs: &[usize], row_indices: &[usize]) -> Vec<f64> {
    let mut row_abs_sum = vec![0.0; dimension];
    let mut values = vec![0.0; row_indices.len()];
    for col in 0..dimension {
        for index in col_ptrs[col]..col_ptrs[col + 1] {
            let row = row_indices[index];
            if row == col {
                continue;
            }
            let value = -0.05 * (1.0 + ((row + col) % 3) as f64);
            values[index] = value;
            row_abs_sum[row] += value.abs();
            row_abs_sum[col] += value.abs();
        }
    }
    for col in 0..dimension {
        for index in col_ptrs[col]..col_ptrs[col + 1] {
            let row = row_indices[index];
            if row == col {
                values[index] = row_abs_sum[col] + 1.5;
                break;
            }
        }
    }
    values
}

fn matrix_validation_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("spral_ssids/matrix_validation");
    group.bench_function("grid_20x20_csc_validation", |bench| {
        let (col_ptrs, row_indices) = grid_matrix(20, 20);
        bench.iter_batched(
            || (col_ptrs.clone(), row_indices.clone()),
            |(col_ptrs, row_indices)| {
                let matrix =
                    SymmetricCscMatrix::new(400, &col_ptrs, &row_indices, None).expect("matrix");
                criterion::black_box(matrix.dimension())
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn symbolic_analysis_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("spral_ssids/symbolic_analysis");
    let nd_options = SsidsOptions {
        ordering: OrderingStrategy::NestedDissection(NestedDissectionOptions {
            leaf_size: 16,
            ..NestedDissectionOptions::default()
        }),
    };
    let auto_options = SsidsOptions::default();
    for (name, col_ptrs, row_indices) in symbolic_cases() {
        let dimension = col_ptrs.len() - 1;
        let matrix = SymmetricCscMatrix::new(dimension, &col_ptrs, &row_indices, None)
            .expect("benchmark matrix");
        group.bench_with_input(
            BenchmarkId::new("natural", name),
            &matrix,
            |bench, matrix| {
                bench.iter(|| {
                    analyse(
                        *matrix,
                        &SsidsOptions {
                            ordering: OrderingStrategy::Natural,
                        },
                    )
                    .expect("analysis")
                });
            },
        );
        group.bench_with_input(BenchmarkId::new("amd", name), &matrix, |bench, matrix| {
            bench.iter(|| {
                analyse(
                    *matrix,
                    &SsidsOptions {
                        ordering: OrderingStrategy::ApproximateMinimumDegree,
                    },
                )
                .expect("analysis")
            });
        });
        group.bench_with_input(BenchmarkId::new("auto", name), &matrix, |bench, matrix| {
            bench.iter(|| analyse(*matrix, &auto_options).expect("analysis"));
        });
        group.bench_with_input(
            BenchmarkId::new("nested_dissection", name),
            &matrix,
            |bench, matrix| {
                bench.iter(|| analyse(*matrix, &nd_options).expect("analysis"));
            },
        );
    }
    group.finish();
}

fn numeric_factorization_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("spral_ssids/numeric_factorization");
    let options = NumericFactorOptions::default();
    let nd_options = SsidsOptions {
        ordering: OrderingStrategy::NestedDissection(NestedDissectionOptions {
            leaf_size: 16,
            ..NestedDissectionOptions::default()
        }),
    };
    let amd_options = SsidsOptions {
        ordering: OrderingStrategy::ApproximateMinimumDegree,
    };
    let auto_options = SsidsOptions::default();
    for (name, col_ptrs, row_indices) in symbolic_cases() {
        let dimension = col_ptrs.len() - 1;
        let values = synthetic_values(dimension, &col_ptrs, &row_indices);
        let matrix = SymmetricCscMatrix::new(dimension, &col_ptrs, &row_indices, Some(&values))
            .expect("benchmark matrix");
        let (amd_symbolic, _) = analyse(matrix, &amd_options).expect("analysis");
        let (auto_symbolic, _) = analyse(matrix, &auto_options).expect("analysis");
        let (nd_symbolic, _) = analyse(matrix, &nd_options).expect("analysis");
        let rhs = vec![1.0; dimension];

        group.bench_with_input(
            BenchmarkId::new("factorize_amd", name),
            &(matrix, amd_symbolic.clone()),
            |bench, (matrix, symbolic)| {
                bench.iter(|| factorize(*matrix, symbolic, &options).expect("factor"));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("factorize_auto", name),
            &(matrix, auto_symbolic.clone()),
            |bench, (matrix, symbolic)| {
                bench.iter(|| factorize(*matrix, symbolic, &options).expect("factor"));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("factorize_nested_dissection", name),
            &(matrix, nd_symbolic.clone()),
            |bench, (matrix, symbolic)| {
                bench.iter(|| factorize(*matrix, symbolic, &options).expect("factor"));
            },
        );
        let (mut factor, _) = factorize(matrix, &auto_symbolic, &options).expect("factor");
        group.bench_function(BenchmarkId::new("solve_auto", name), |bench| {
            bench.iter_batched(
                || rhs.clone(),
                |mut rhs| factor.solve_in_place(&mut rhs).expect("solve"),
                BatchSize::SmallInput,
            );
        });
        group.bench_with_input(
            BenchmarkId::new("refactorize_auto", name),
            &(factor.clone(), matrix),
            |bench, (factor, _matrix)| {
                let updated_values = values.iter().map(|value| value * 1.01).collect::<Vec<_>>();
                bench.iter_batched(
                    || {
                        let factor = factor.clone();
                        let updated_matrix = SymmetricCscMatrix::new(
                            dimension,
                            &col_ptrs,
                            &row_indices,
                            Some(&updated_values),
                        )
                        .expect("updated matrix");
                        (factor, updated_matrix)
                    },
                    |(mut factor, updated_matrix)| {
                        factor.refactorize(updated_matrix).expect("refactorize");
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

criterion_group!(
    symbolic_validation,
    matrix_validation_bench,
    symbolic_analysis_bench,
    numeric_factorization_bench
);
criterion_main!(symbolic_validation);
