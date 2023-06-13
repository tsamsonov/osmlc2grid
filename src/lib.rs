use pyo3::prelude::*;
use numpy::ndarray::{Array2, ArrayView2, Array3, ArrayView3};
use numpy::{PyArray2, PyReadonlyArray2, PyArray3, PyReadonlyArray3, IntoPyArray};
use std::collections::{BinaryHeap};
use ordered_float::OrderedFloat;
use std::thread;
use std::sync::mpsc;
use num_cpus;

enum EuclideanTransform {
    DISTANCE,
    ALLOCATION
}
/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn rasterspace(_py: Python<'_>, m: &PyModule) -> PyResult<()>
{
    fn rescale(source: ArrayView2<'_, f64>) -> Array2<f64> {
        10_f64 * &source
    }

    fn euclidean_transform(input: ArrayView2<'_, f64>, cellsize: f64, transform: EuclideanTransform) -> Array2<f64> {
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

        let mut distance: Array2<f64> = input.map(
            |z| if *z > 0.0 { 0.0 } else { f64::INFINITY }
        );

        let mut allocation: Array2<f64> = input.map(
            |z| if *z > 0.0 { *z } else { f64::INFINITY }
        );

        for row in 0..rows {
            for col in 0..cols {
                z = distance[[row, col]];
                if z != 0.0 {
                    z_min = f64::INFINITY;
                    which_cell = 0;
                    for i in 0..4 {
                        xraw = col as isize + dx[i];
                        yraw = row as isize + dy[i];

                        if xraw >= 0 && xraw < cols as isize && yraw >= 0 && yraw < rows as isize {
                            x = xraw as usize;
                            y = yraw as usize;

                            z2 = distance[[y, x]];

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
                                distance[[row, col]] = z_min;
                                x = (col as isize + dx[which_cell]) as usize;
                                y = (row as isize + dy[which_cell]) as usize;
                                rx[[row, col]] = rx[[y, x]] + gx[which_cell];
                                ry[[row, col]] = ry[[y, x]] + gy[which_cell];
                                allocation[[row, col]] = allocation[[y, x]];
                            }
                        }
                    }
                }
            }
        }

        for row in (0..rows).rev() {
            for col in (0..cols).rev() {
                z = distance[[row, col]];
                if z != 0.0 {
                    z_min = f64::INFINITY;
                    which_cell = 0;
                    for i in 4..8 {
                        xraw = col as isize + dx[i];
                        yraw = row as isize + dy[i];

                        if xraw >= 0 && xraw < cols as isize && yraw >= 0 && yraw < rows as isize {
                            x = xraw as usize;
                            y = yraw as usize;

                            z2 = distance[[y, x]];

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
                                distance[[row, col]] = z_min;
                                x = (col as isize + dx[which_cell]) as usize;
                                y = (row as isize + dy[which_cell]) as usize;
                                rx[[row, col]] = rx[[y, x]] + gx[which_cell];
                                ry[[row, col]] = ry[[y, x]] + gy[which_cell];
                                allocation[[row, col]] = allocation[[y, x]];
                            }
                        }
                    }
                }
            }
        }

        distance.map_inplace(|x| *x = f64::sqrt(*x));
        distance *= cellsize;

        match transform {
            EuclideanTransform::DISTANCE => distance,
            _ => allocation
        }
    }

    fn euclidean_centrality(input: ArrayView2<'_, f64>, cellsize: f64) -> Array2<f64> {
        let distance =
            euclidean_transform(input, cellsize, EuclideanTransform::DISTANCE);
        let allocation=
            euclidean_transform(input, cellsize, EuclideanTransform::ALLOCATION);

        let shape = input.raw_dim();
        let nrow = shape[0];
        let ncol = shape[1];
        let di: [isize; 8] = [-1, -1, 0, 1, 1, 1, 0, -1];
        let dj: [isize; 8] = [0, -1, -1, -1, 0, 1, 1, 1];

        let arcallocation = allocation.to_shared();

        let nproc = num_cpus::get_physical();

        let mut tasks = Vec::new();
        let (tx, rx) = mpsc::channel();

        for proc in 0..nproc {
            let input_ref = arcallocation.clone();
            let tx1 = tx.clone();

            tasks.push(thread::spawn(move || {

                let (mut ik, mut jk): (isize, isize);
                let (mut uik, mut ujk): (usize, usize);

                for i in (0..nrow).filter(|r| r % nproc == proc) {
                    let mut data = vec![0.0; ncol];
                    for j in 0..ncol {
                        for k in 0..8 {
                            ik = i as isize + di[k];
                            jk = j as isize + dj[k];
                            if ik < 0 || ik >= nrow as isize || jk < 0 || jk >= ncol as isize {
                                continue;
                            }

                            uik = ik as usize;
                            ujk = jk as usize;

                            if input_ref[[i, j]] != f64::NAN {
                                if input_ref[[i, j]] != input_ref[[uik, ujk]] {
                                    data[j] = 1.0;
                                    break;
                                }
                            }
                        }
                    }

                    tx1.send((i, data)).unwrap();
                }
            }));
        }

        let mut borders = Array2::<f64>::zeros((nrow, ncol));

        for _ in 0..nrow {
            let data  = rx.recv().expect("Error receiving data from thread");
            for col in 0..ncol {
                borders[[data.0, col]] = data.1[col];
            }
        }

        let distance1 =
            euclidean_transform(borders.view(), cellsize, EuclideanTransform::DISTANCE);

        return &distance / (&distance + &distance1);
    }

    fn euclidean_width(input: ArrayView3<'_, f64>, cellsize: f64) -> Array3<f64> {
        let shape = input.raw_dim();
        let nrow = shape[1];
        let ncol = shape[2];
        let mut output = Array3::<f64>::zeros((4, nrow, ncol));

        let (mut hsum, mut hmean, mut hw, mut radius): (f64, f64, f64, f64);
        let (mut w, mut ik, mut jl): (isize, isize, isize);
        let (mut uik, mut ujl): (usize, usize);
        let mut n: usize;

        let mut queue = BinaryHeap::new();

        for i in 0..nrow {
            for j in 0..ncol {
                if input[[0, i, j]] > 0.0 {
                    queue.push((OrderedFloat(input[[0, i, j]]), i, j));
                }
            }
        }

        while queue.len() > 0 {
            let (ord_radius, i, j) = queue.pop().unwrap();
            let mut covered: Vec<(usize, usize)> = Vec::new();

            radius = f64::from(ord_radius);

            w = (radius / cellsize).floor() as isize;

            hsum = 0.0;
            n = 0;

            for k in (-w+1)..w {
                for l in (-w+1)..w {
                    if k*k + l*l > w*w {
                        continue;
                    }
                    ik = i as isize + k;
                    jl = j as isize + l;

                    if ik < 0 || ik >= nrow as isize || jl < 0 || jl >= ncol as isize {
                        continue;
                    }

                    uik = ik as usize;
                    ujl = jl as usize;

                    if output[[1, uik, ujl]] < 2.0 * radius {
                        covered.push((uik, ujl));
                    }

                    hsum += input[[1, uik, ujl]];
                    n += 1;
                }
            }
            if covered.len() > 0 {
                hmean = hsum / n as f64;
                hw = 0.5 * hmean / radius;
                for (uik, ujl) in covered {
                    output[[0, uik, ujl]] = (i * ncol + j) as f64;
                    output[[1, uik, ujl]] = 2.0 * radius;
                    output[[2, uik, ujl]] = hmean;
                    output[[3, uik, ujl]] = hw;
                }
            }
        }

        return output;
    }

    fn euclidean_width_parallel(input: ArrayView3<'_, f64>, cellsize: f64) -> Array3<f64> {
        let shape = input.raw_dim();
        let nrow = shape[1];
        let ncol = shape[2];
        let mut output = Array3::<f64>::zeros((4, nrow, ncol));

        let arcoutput = output.to_shared();
        let arcinput = input.to_shared();

        let nproc = num_cpus::get_physical();

        let mut tasks = Vec::new();

        for proc in 0..nproc {
            let mut output_ref = arcoutput.clone();
            let input_ref = arcinput.clone();

            tasks.push(thread::spawn(move || {
                let (mut hsum, mut hmean, mut hw, mut radius): (f64, f64, f64, f64);
                let (mut w, mut ik, mut jl): (isize, isize, isize);
                let (mut uik, mut ujl): (usize, usize);
                let mut n: usize;

                let mut queue = BinaryHeap::new();

                for i in (0..nrow).filter(|r| r % nproc == proc) {
                    for j in 0..ncol {
                        if input_ref[[0, i, j]] > 0.0 {
                            queue.push((OrderedFloat(input_ref[[0, i, j]]), i, j));
                        }
                    }
                }

                while queue.len() > 0 {
                    let (ord_radius, i, j) = queue.pop().unwrap();
                    let mut covered: Vec<(usize, usize)> = Vec::new();

                    radius = f64::from(ord_radius);

                    w = (radius / cellsize).floor() as isize;

                    hsum = 0.0;
                    n = 0;

                    for k in (-w+1)..w {
                        for l in (-w+1)..w {
                            if k*k + l*l > w*w {
                                continue;
                            }
                            ik = i as isize + k;
                            jl = j as isize + l;

                            if ik < 0 || ik >= nrow as isize || jl < 0 || jl >= ncol as isize {
                                continue;
                            }

                            uik = ik as usize;
                            ujl = jl as usize;

                            if output_ref[[1, uik, ujl]] < 2.0 * radius {
                                covered.push((uik, ujl));
                            }

                            hsum += input_ref[[1, uik, ujl]];
                            n += 1;
                        }
                    }
                    if covered.len() > 0 {
                        hmean = hsum / n as f64;
                        hw = 0.5 * hmean / radius;
                        for (uik, ujl) in covered {
                            output_ref[[0, uik, ujl]] = (i * ncol + j) as f64;
                            output_ref[[1, uik, ujl]] = 2.0 * radius;
                            output_ref[[2, uik, ujl]] = hmean;
                            output_ref[[3, uik, ujl]] = hw;
                        }
                    }
                }

                return output_ref;
            }))
        }

        for task in tasks {
            let add = task.join().unwrap();
            for i in 0..nrow {
                for j in 0..ncol {
                    if add[[1, i, j]] > output[[1, i, j]] {
                        output[[0, i, j]] = add[[0, i, j]] ;
                        output[[1, i, j]] = add[[1, i, j]] ;
                        output[[2, i, j]] = add[[2, i, j]] ;
                        output[[3, i, j]] = add[[3, i, j]] ;
                    }
                }
            }
        }

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

    // wrapper of `euclidean_distance
    #[pyfn(m)]
    #[pyo3(name = "euclidean_distance")]
    fn euclidean_distance_py<'py>(
        py: Python<'py>,
        source: PyReadonlyArray2<'_, f64>,
        cellsize: f64
    ) -> &'py PyArray2<f64> {
        let source = source.as_array();
        let result = euclidean_transform(source, cellsize, EuclideanTransform::DISTANCE);
        result.into_pyarray(py)
    }

    // wrapper of `euclidean_allocation
    #[pyfn(m)]
    #[pyo3(name = "euclidean_allocation")]
    fn euclidean_allocation_py<'py>(
        py: Python<'py>,
        source: PyReadonlyArray2<'_, f64>,
        cellsize: f64
    ) -> &'py PyArray2<f64> {
        let source = source.as_array();
        let result = euclidean_transform(source, cellsize, EuclideanTransform::ALLOCATION);
        result.into_pyarray(py)
    }

    // wrapper of `euclidean_width`
    #[pyfn(m)]
    #[pyo3(name = "euclidean_centrality")]
    fn euclidean_centrality_py<'py>(
        py: Python<'py>,
        source: PyReadonlyArray2<'_, f64>,
        cellsize: f64
    ) -> &'py PyArray2<f64> {
        let source = source.as_array();
        let result = euclidean_centrality(source, cellsize);
        result.into_pyarray(py)
    }

    // wrapper of `euclidean_width`
    #[pyfn(m)]
    #[pyo3(name = "euclidean_width")]
    fn euclidean_width_py<'py>(
        py: Python<'py>,
        source: PyReadonlyArray3<'_, f64>,
        cellsize: f64
    ) -> &'py PyArray3<f64> {
        let source = source.as_array();
        let result = euclidean_width(source, cellsize);
        result.into_pyarray(py)
    }

    // wrapper of `euclidean_width`
    #[pyfn(m)]
    #[pyo3(name = "euclidean_width_parallel")]
    fn euclidean_width_parallel_py<'py>(
        py: Python<'py>,
        source: PyReadonlyArray3<'_, f64>,
        cellsize: f64
    ) -> &'py PyArray3<f64> {
        let source = source.as_array();
        let result = euclidean_width_parallel(source, cellsize);
        result.into_pyarray(py)
    }

    Ok(())
}
