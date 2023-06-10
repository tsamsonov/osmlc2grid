use pyo3::prelude::*;
use numpy::ndarray::{Array2, ArrayView2};
use numpy::{PyArray2, PyReadonlyArray2, IntoPyArray};

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn rasterspace(_py: Python<'_>, m: &PyModule) -> PyResult<()>
{
    fn rescale(source: ArrayView2<'_, f64>) -> Array2<f64> {
        10_f64 * &source
    }

    fn euclidean_distance(input: ArrayView2<'_, f64>) -> Array2<f64> {
        let shape = input.raw_dim();
        let rows = shape[0];
        let cols = shape[1];
        let mut rx: Array2<f64> = Array2::zeros(shape);
        let mut ry: Array2<f64> = Array2::zeros(shape);

        // partially used code from https://github.com/jblindsay/whitebox-tools/blob/master/whitebox-tools-app/src/tools/gis_analysis/euclidean_distance.rs
        let mut h: f64;
        let mut which_cell: usize;
        let dx: [isize; 8] = [-1, -1, 0, 1, 1, 1, 0, -1];
        let dy: [isize; 8] = [0, -1, -1, -1, 0, 1, 1, 1];
        let gx = [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0];
        let gy = [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0];
        let (mut x, mut y): (usize, usize);
        let (mut xraw, mut yraw): (isize, isize);
        let (mut z, mut z2, mut z_min): (f64, f64, f64);

        let mut output: Array2<f64> = input.map(
            |z| if *z > 0.0 { 0.0 } else { f64::INFINITY }
        );

        for row in 0..rows {
            for col in 0..cols {
                z = output[[row, col]];
                if z != 0.0 {
                    z_min = f64::INFINITY;
                    which_cell = 0;
                    for i in 0..4 {
                        xraw = col as isize + dx[i];
                        yraw = row as isize + dy[i];

                        if xraw >= 0 && xraw < cols as isize && yraw >= 0 && yraw < rows as isize {
                            x = xraw as usize;
                            y = yraw as usize;

                            z2 = output[[y, x]];

                            if z2 != f64::NAN {
                                h = match i {
                                    0 => 2.0 *  rx[[y, x]] + 1.0,
                                    1 => 2.0 * (rx[[y, x]] + ry[[y, x]] + 1.0),
                                    2 => 2.0 *  ry[[y, x]] + 1.0,
                                    _ => 2.0 * (rx[[y, x]] + ry[[y, x]] + 1.0), // 3
                                };
                                z2 += h;
                                if z2 < z_min {
                                    z_min = z2;
                                    which_cell = i;
                                }
                            }

                            if z_min < z {
                                output[[row, col]] = z_min;
                                x = (col as isize + dx[which_cell]) as usize;
                                y = (row as isize + dy[which_cell]) as usize;
                                rx[[row, col]] = rx[[y, x]] + gx[which_cell];
                                ry[[row, col]] = ry[[y, x]] + gy[which_cell];
                            }
                        }
                    }
                }
            }
        }

        for row in (0..rows).rev() {
            for col in (0..cols).rev() {
                z = output[[row, col]];
                if z != 0.0 {
                    z_min = f64::INFINITY;
                    which_cell = 0;
                    for i in 4..8 {
                        xraw = col as isize + dx[i];
                        yraw = row as isize + dy[i];

                        if xraw >= 0 && xraw < cols as isize && yraw >= 0 && yraw < rows as isize {
                            x = xraw as usize;
                            y = yraw as usize;

                            z2 = output[[y, x]];

                            if z2 != f64::NAN {
                                h = match i {
                                    4 => 2.0 *  rx[[y, x]] + 1.0,
                                    5 => 2.0 * (rx[[y, x]] + ry[[y, x]] + 1.0),
                                    6 => 2.0 *  ry[[y, x]] + 1.0,
                                    _ => 2.0 * (rx[[y, x]] + ry[[y, x]] + 1.0), // 3
                                };
                                z2 += h;
                                if z2 < z_min {
                                    z_min = z2;
                                    which_cell = i;
                                }
                            }

                            if z_min < z {
                                output[[row, col]] = z_min;
                                x = (col as isize + dx[which_cell]) as usize;
                                y = (row as isize + dy[which_cell]) as usize;
                                rx[[row, col]] = rx[[y, x]] + gx[which_cell];
                                ry[[row, col]] = ry[[y, x]] + gy[which_cell];
                            }
                        }
                    }
                }
            }
        }

        output.map_inplace(|x| *x = f64::sqrt(*x));

        return output;
    }

    // wrapper of `rescale`
    #[pyfn(m)]
    #[pyo3(name = "rescale")]
    fn rescale_py<'py>(
        py: Python<'py>,
        source: PyReadonlyArray2<'_, f64>
    ) -> &'py PyArray2<f64> {
        let source = source.as_array();
        let result = rescale(source);
        result.into_pyarray(py)
    }

    // wrapper of `euclideandistance`
    #[pyfn(m)]
    #[pyo3(name = "euclidean_distance")]
    fn euclidean_distance_py<'py>(
        py: Python<'py>,
        source: PyReadonlyArray2<'_, f64>
    ) -> &'py PyArray2<f64> {
        let source = source.as_array();
        let result = euclidean_distance(source);
        result.into_pyarray(py)
    }

    Ok(())
}
