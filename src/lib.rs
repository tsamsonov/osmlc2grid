use std::cmp::{max, min};
use pyo3::prelude::*;
use numpy::ndarray::{s, Array2, ArrayView2, Array3};
use numpy::{PyArray2, PyReadonlyArray2, PyArray3, IntoPyArray};
use std::collections::{BinaryHeap};
use ordered_float::OrderedFloat;
use std::f64::consts::PI;
use std::thread;
use std::sync::mpsc;
use num_cpus;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

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

    fn euclidean_distance(input: ArrayView2<'_, f64>, cellsize: f64) -> Array2<f64> {
        euclidean_transform(input, cellsize, EuclideanTransform::DISTANCE)
    }

    fn euclidean_allocation(input: ArrayView2<'_, f64>, cellsize: f64) -> Array2<f64> {
        euclidean_transform(input, cellsize, EuclideanTransform::ALLOCATION)
    }

    fn euclidean_antidistance(input: ArrayView2<'_, f64>, cellsize: f64) -> Array2<f64> {

        let allocation=
            euclidean_transform(input, cellsize, EuclideanTransform::ALLOCATION);

        let shape = input.raw_dim();
        let nrow = shape[0];
        let ncol = shape[1];
        let di: [isize; 8] = [-1, -1,  0,  1, 1, 1, 0, -1];
        let dj: [isize; 8] = [ 0, -1, -1, -1, 0, 1, 1,  1];

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

        euclidean_transform(borders.view(), cellsize, EuclideanTransform::DISTANCE)
    }

    fn euclidean_centrality(input: ArrayView2<'_, f64>, cellsize: f64) -> Array2<f64> {
        let distance =
            euclidean_transform(input, cellsize, EuclideanTransform::DISTANCE);
        let anti_distance =
            euclidean_antidistance(input, cellsize);
        &distance / (&distance + &anti_distance)
    }

    fn euclidean_width(distance: ArrayView2<'_, f64>, cellsize: f64) -> Array2<f64> {
        let shape = distance.raw_dim();
        let nrow = shape[0];
        let ncol = shape[1];
        let mut output = Array2::<f64>::zeros((nrow, ncol));

        let arcoutput = output.to_shared();
        let arcinput = distance.to_shared();

        let nproc = num_cpus::get_physical();

        let mut tasks = Vec::new();

        for proc in 0..nproc {
            let mut output_ref = arcoutput.clone();
            let input_ref = arcinput.clone();

            tasks.push(thread::spawn(move || {
                let mut radius: f64;
                let (mut w, mut ik, mut jl): (isize, isize, isize);
                let (mut uik, mut ujl): (usize, usize);

                let mut queue = BinaryHeap::new();

                for i in (0..nrow).filter(|r| r % nproc == proc) {
                    for j in 0..ncol {
                        if input_ref[[i, j]] > 0.0 {
                            queue.push((OrderedFloat(input_ref[[i, j]]), i, j));
                        }
                    }
                }

                while queue.len() > 0 {
                    let (ord_radius, i, j) = queue.pop().unwrap();

                    radius = f64::from(ord_radius);

                    w = (radius / cellsize).floor() as isize;

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

                            if output_ref[[uik, ujl]] < 2.0 * radius {
                                output_ref[[uik, ujl]] = 2.0 * radius;
                            }
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
                    if add[[i, j]] > output[[i, j]] {
                        output[[i, j]] = add[[i, j]] ;
                    }
                }
            }
        }

        return output;
    }

    fn euclidean_width_tiles(distance: ArrayView2<'_, f64>, cellsize: f64, rows: usize, cols: usize, hard: bool) -> Array3<usize> {
        let shape = distance.raw_dim();
        let nrow = shape[0];
        let ncol = shape[1];
        let mut output = Array3::<usize>::zeros((rows, cols, 8));

        let drow = nrow / rows;
        let dcol = ncol / cols;

        let (mut row1, mut row2, mut col1, mut col2): (usize, usize, usize, usize);
        let (mut rowmin, mut colmin, mut rowmax, mut colmax, mut delta, mut radius):
            (isize, isize, isize, isize, isize, isize);

        let (mut rowmin_out, mut colmin_out, mut rowmax_out, mut colmax_out):
            (usize, usize, usize, usize);

        let (mut rowmin_in, mut colmin_in, mut rowmax_in, mut colmax_in):
            (usize, usize, usize, usize);

        let (mut rowmin_in_rad, mut colmin_in_rad, mut rowmax_in_rad, mut colmax_in_rad):
            (usize, usize, usize, usize);

        let mut slices = Array2::default((nrow, ncol));
        let mut ext_slices = Array2::default((nrow, ncol));
        let mut out_slices = Array2::default((nrow, ncol));

        row1 = 0;
        row2 = drow + nrow % rows - 1;

        for row in 0..rows {
            col1 = 0;
            col2 = dcol + ncol % cols - 1;
            for col in 0..cols {

                rowmin = row1 as isize;
                rowmax = row2 as isize;
                colmin = col1 as isize;
                colmax = col2 as isize;

                rowmin_out = row1;
                rowmax_out = row2;
                colmin_out = col1;
                colmax_out = col2;

                rowmin_in = row1;
                rowmax_in = row2;
                colmin_in = col1;
                colmax_in = col2;

                rowmin_in_rad = 0;
                colmin_in_rad = 0;
                rowmax_in_rad = 0;
                colmax_in_rad = 0;

                for i in row1..row2 {
                    for j in col1..col2 {
                        radius = (distance[[i, j]] / cellsize) as isize;

                        delta = i as isize - radius;
                        if delta < rowmin {
                            rowmin = delta;
                            rowmin_out = i;
                        }

                        if (hard || delta < row1 as isize) && radius > rowmin_in_rad as isize {
                            rowmin_in_rad = radius as usize;
                            rowmin_in = i;
                        }

                        delta = j as isize - radius;
                        if delta < colmin {
                            colmin = delta;
                            colmin_out = j;
                        }

                        if (hard || delta < col1 as isize) && radius > colmin_in_rad as isize {
                            colmin_in_rad = radius as usize;
                            colmin_in = j;
                        }

                        delta = i as isize + radius;
                        if delta > rowmax {
                            rowmax = delta;
                            rowmax_out = i;
                        }

                        if (hard || delta > row2 as isize) && radius > rowmax_in_rad as isize {
                            rowmax_in_rad = radius as usize;
                            rowmax_in = i;
                        }

                        delta = j as isize + radius;
                        if delta > colmax {
                            colmax = delta;
                            colmax_out = j;
                        }

                        if (hard || delta > col2 as isize) && radius > colmax_in_rad as isize {
                            colmax_in_rad = radius as usize;
                            colmax_in = j;
                        }
                    }

                    rowmin = max(0, rowmin);
                    colmin = max(0, colmin);

                    rowmax = min((nrow - 1) as isize, rowmax);
                    colmax = min((ncol - 1) as isize, colmax);

                    rowmin_out = max(rowmin_out, rowmin_in);
                    colmin_out = max(colmin_out, colmin_in);

                    rowmax_out = min(rowmax_out, rowmax_in);
                    colmax_out = min(colmax_out, colmax_in);
                }

                slices[[row,col]] = (row1, row2, col1, col2);
                ext_slices[[row,col]] = (rowmin as usize, rowmax as usize, colmin as usize, colmax as usize);
                out_slices[[row,col]] = (rowmin_out, rowmax_out, colmin_out, colmax_out);

                col1  = col2;
                col2 += dcol;
            }
            row1  = row2;
            row2 += drow;
        }

        for row in 0..rows {
            for col in 0..cols {
                if row > 0 {
                    if ext_slices[[row, col]].0 > out_slices[[row-1, col]].1 {
                        ext_slices[[row, col]].0 = out_slices[[row-1, col]].1
                    }
                    if col > 0 {
                        if ext_slices[[row, col]].0 > out_slices[[row-1, col-1]].1 {
                            ext_slices[[row, col]].0 = out_slices[[row-1, col-1]].1
                        }
                    }
                    if col < cols-1 {
                        if ext_slices[[row, col]].0 > out_slices[[row-1, col+1]].1 {
                            ext_slices[[row, col]].0 = out_slices[[row-1, col+1]].1
                        }
                    }
                }
                if col > 0 {
                    if ext_slices[[row, col]].2 > out_slices[[row, col-1]].3 {
                        ext_slices[[row, col]].2 = out_slices[[row, col-1]].3
                    }
                    if row > 0 {
                        if ext_slices[[row, col]].2 > out_slices[[row-1, col-1]].3 {
                            ext_slices[[row, col]].2 = out_slices[[row-1, col-1]].3
                        }
                    }
                    if row < rows-1 {
                        if ext_slices[[row, col]].2 > out_slices[[row+1, col-1]].3 {
                            ext_slices[[row, col]].2 = out_slices[[row+1, col-1]].3
                        }
                    }
                }
                if row < rows-1 {
                    if ext_slices[[row, col]].1 < out_slices[[row+1, col]].0 {
                        ext_slices[[row, col]].1 = out_slices[[row+1, col]].0
                    }
                    if col > 0 {
                        if ext_slices[[row, col]].1 < out_slices[[row+1, col-1]].0 {
                            ext_slices[[row, col]].1 = out_slices[[row+1, col-1]].0
                        }
                    }
                    if col < cols-1 {
                        if ext_slices[[row, col]].1 < out_slices[[row+1, col+1]].0 {
                            ext_slices[[row, col]].1 = out_slices[[row+1, col+1]].0
                        }
                    }
                }
                if col < cols-1 {
                    if ext_slices[[row, col]].3 < out_slices[[row, col+1]].2 {
                        ext_slices[[row, col]].3 = out_slices[[row, col+1]].2
                    }
                    if row > 0 {
                        if ext_slices[[row, col]].3 < out_slices[[row-1, col+1]].2 {
                            ext_slices[[row, col]].3 = out_slices[[row-1, col+1]].2
                        }
                    }
                    if row < rows-1 {
                        if ext_slices[[row, col]].3 < out_slices[[row+1, col+1]].2 {
                            ext_slices[[row, col]].3 = out_slices[[row+1, col+1]].2
                        }
                    }
                }
            }
        }


        for row in 0..rows {
            for col in 0..cols {
                let (row1, row2, col1, col2) = slices[[row,col]];
                let (erow1, erow2, ecol1, ecol2) = ext_slices[[row,col]];
                output[[row, col, 0]] = row1;
                output[[row, col, 1]] = row2;
                output[[row, col, 2]] = col1;
                output[[row, col, 3]] = col2;
                output[[row, col, 4]] = erow1;
                output[[row, col, 5]] = erow2;
                output[[row, col, 6]] = ecol1;
                output[[row, col, 7]] = ecol2;
            }
        }
        return output;
    }

    fn euclidean_width_split(distance: ArrayView2<'_, f64>, cellsize: f64, rows: usize, cols: usize, hard: bool) -> Array2<f64> {
        let shape = distance.raw_dim();
        let nrow = shape[0];
        let ncol = shape[1];
        let mut output = Array2::<f64>::zeros((nrow, ncol));

        let tiles = euclidean_width_tiles(distance, cellsize, rows, cols, hard);

        for row in 0..rows {
            for col in 0..cols {
                let tile = tiles.slice(s![row, col, ..]);

                println!(
                    "Tile {}, {}: [{}, {}] × [{}, {}] -> [{}, {}] × [{}, {}] = {} × {}",
                    row, col,
                    tile[0],  tile[1], tile[2],  tile[3],
                    tile[4],  tile[5], tile[6],  tile[7],
                    tile[5] - tile[4], tile[7] - tile[6]
                );

                let width = euclidean_width(distance.slice(s![tile[4]..tile[5], tile[6]..tile[7]]), cellsize);

                for i in tile[0]..tile[1] {
                    for j in tile[2]..tile[3] {
                        output[[i, j]] = width[[i - tile[4], j - tile[6]]];
                    }
                }
            }
        }

        return output;
    }

    fn euclidean_width_params(distance: ArrayView2<'_, f64>, height: ArrayView2<'_, f64>, cellsize: f64) -> Array3<f64> {
        let shape = distance.raw_dim();
        let nrow = shape[0];
        let ncol = shape[1];

        // let mut output = Array3::<f64>::zeros((5, nrow, ncol));
        let mut output = Array3::<f64>::zeros((2, nrow, ncol));
        let arcoutput = output.to_shared();

        let arcdistance = distance.to_shared();
        let archeight = height.to_shared();

        let nproc = num_cpus::get_physical();

        let mut tasks = Vec::new();

        let m = MultiProgress::new();
        let sty = ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
        ).unwrap()
         .progress_chars("##-");

        let mut bars: Vec<ProgressBar> = Vec::new();
        let pb = m.add(ProgressBar::new(100));
        pb.set_style(sty.clone());
        bars.push(pb);

        for proc in 1..nproc {
            let pb = m.insert_after(&bars[proc-1], ProgressBar::new(100));
            pb.set_style(sty.clone());
            bars.push(pb);
        }

        for proc in 0..nproc {
            let mut output_ref = arcoutput.clone();
            let distance_ref = arcdistance.clone();
            let height_ref = archeight.clone();

            let bars_clone = bars.clone();

            tasks.push(thread::spawn(move || {
                // let mut hw f64;
                let (mut hsum, mut hmean, mut radius): (f64, f64, f64);
                let (mut w, mut ik, mut jl): (isize, isize, isize);
                let (mut uik, mut ujl): (usize, usize);
                let mut n: usize;

                let mut queue = BinaryHeap::new();

                for i in (0..nrow).filter(|r| r % nproc == proc) {
                    for j in 0..ncol {
                        if distance_ref[[i, j]] > 0.0 {
                            queue.push((OrderedFloat(distance_ref[[i, j]]), i, j));
                        }
                        // output_ref[[0, i, j]] = -1.0;
                        // output_ref[[1, i, j]] = -1.0;
                    }
                }

                bars_clone[proc].set_length(queue.len() as u64);

                while queue.len() > 0 {
                    let (ord_radius, i, j) = queue.pop().unwrap();
                    // radius = f64::min(500.0, f64::from(ord_radius));

                    if output_ref[[0, i, j]] >= 1000.0 {
                        continue;
                    }

                    radius = f64::from(ord_radius);

                    bars_clone[proc].inc(1);
                    bars_clone[proc].set_message(format!("radius = {}", radius));

                    let mut covered: Vec<(usize, usize)> = Vec::new();

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

                            if output_ref[[0, uik, ujl]] < 2.0 * radius {
                                covered.push((uik, ujl));
                            }

                            hsum += height_ref[[uik, ujl]];
                            n += 1;
                        }
                    }
                    if covered.len() > 0 {
                        hmean = hsum / n as f64;
                        // hw = 0.5 * hmean / radius;
                        for (uik, ujl) in covered {
                            output_ref[[0, uik, ujl]] = 2.0 * radius;
                            output_ref[[1, uik, ujl]] = hmean;
                            // output_ref[[0, uik, ujl]] = (i * ncol + j) as f64;
                            // output_ref[[1, uik, ujl]] =
                            //     cellsize * ((i as f64 - uik as f64).powi(2) +
                            //         (j as f64 - ujl as f64).powi(2)).sqrt();
                            // output_ref[[4, uik, ujl]] = hw;
                        }
                    }
                }

                bars_clone[proc].finish_with_message("done");

                return output_ref;
            }))
        }

        m.clear().unwrap();

        for task in tasks {
            let add = task.join().unwrap();
            for i in 0..nrow {
                for j in 0..ncol {
                    // if add[[2, i, j]] > output[[2, i, j]] {
                    if add[[0, i, j]] > output[[0, i, j]] {
                        output[[0, i, j]] = add[[0, i, j]];
                        output[[1, i, j]] = add[[1, i, j]];
                        // output[[2, i, j]] = add[[2, i, j]];
                        // output[[3, i, j]] = add[[3, i, j]];
                        // output[[4, i, j]] = add[[4, i, j]];
                    }
                }
            }
        }

        return output
    }

    fn euclidean_width_params_split(distance: ArrayView2<'_, f64>, height: ArrayView2<'_, f64>, cellsize: f64, rows: usize, cols: usize, hard: bool) -> Array3<f64> {
        let shape = distance.raw_dim();
        let nrow = shape[0];
        let ncol = shape[1];
        let mut output = Array3::<f64>::zeros((2, nrow, ncol));
        // let mut output = Array3::<f64>::zeros((5, nrow, ncol));

        // for i in 0..nrow {
        //     for j in 0..ncol {
        //         output[[0, i, j]] = -1.0;
        //     }
        // }

        let tiles = euclidean_width_tiles(distance, cellsize, rows, cols, hard);

        for row in 0..rows {
            for col in 0..cols {
                let tile = tiles.slice(s![row, col, ..]);

                println!(
                    "Tile {}, {}: [{}, {}] × [{}, {}] -> [{}, {}] × [{}, {}] = {} × {}",
                    row, col,
                    tile[0],  tile[1], tile[2],  tile[3],
                    tile[4],  tile[5], tile[6],  tile[7],
                    tile[5] - tile[4], tile[7] - tile[6]
                );

                let params = euclidean_width_params(
                    distance.slice(s![tile[4]..tile[5], tile[6]..tile[7]]),
                    height.slice(s![tile[4]..tile[5], tile[6]..tile[7]]),
                    cellsize
                );

                // let ecols = tile[7] - tile[6];

                for i in tile[0]..tile[1] {
                    for j in tile[2]..tile[3] {
                        output[[0, i, j]] = params[[0, i - tile[4], j - tile[6]]];
                        output[[1, i, j]] = params[[1, i - tile[4], j - tile[6]]];

                        // let eij = tile[[0, i - tile[4], j - tile[6]]];
                        // if eij > 0.0 {
                        //     let ei = tile[4] + eij as usize / ecols;
                        //     let ej = tile[6] + eij as usize % ecols;
                        //     output[[0, i, j]] = (ei * ncol + ej) as f64;
                        // }
                        // output[[3, i, j]] = tile[[3, i - tile[4], j - tile[6]]];
                        // output[[4, i, j]] = tile[[4, i - tile[4], j - tile[6]]];

                    }
                }
                println!("{}, {}", row, col);
            }
        }

        return output;
    }

    fn is_within(i: isize, j: isize, nrow: usize, ncol: usize) -> bool {
        i >= 0 && i < nrow as isize && j >= 0 && j < ncol as isize
    }

    fn get_shifts_j(i0: isize, j0: isize, i1: isize, j1: isize) -> Vec<(isize, isize)> {
        let dj = j1 - j0;
        let mut di = i1 - i0;
        let mut ij = 1;
        if di < 0 {
            ij = -1;
            di = -di;
        }

        let mut d = 2 * di - dj;
        let mut i = i0;

        let mut s = Vec::new();

        for j in j0..j1 {
            s.push((i, j));
            if d > 0 {
                i += ij;
                d += 2 * (di - dj);
            } else {
                d += 2 * di;
            }
        }

        return s;
    }

    fn get_shifts_i(i0: isize, j0: isize, i1: isize, j1: isize) -> Vec<(isize, isize)> {
        let mut dj = j1 - j0;
        let di = i1 - i0;
        let mut ji = 1;
        if dj < 0 {
            ji = -1;
            dj = -dj;
        }

        let mut d = 2 * dj - di;
        let mut j = j0;

        let mut s = Vec::new();

        for i in i0..i1 {
            s.push((i, j));
            if d > 0 {
                j += ji;
                d += 2 * (dj - di);
            } else {
                d += 2 * dj;
            }
        }

        return s;
    }

    fn get_shifts(i0: isize, j0: isize, i1: isize, j1: isize) -> Vec<(isize, isize)> {
        if (i1 - i0).abs() < (j1 - j0).abs() {
            if j0 > j1 {
                let mut shifts = get_shifts_j(i1, j1, i0, j0);
                shifts.reverse();
                shifts
            } else {
                get_shifts_j(i0, j0, i1, j1)
            } 
        } else {
            if i0 > i1 {
                let mut shifts = get_shifts_i(i1, j1, i0, j0);
                shifts.reverse();
                shifts
            } else {
                get_shifts_i(i0, j0, i1, j1)
            }
        }
    }

    fn get_length(si1: isize, sj1: isize, si2: isize, sj2: isize) -> f64 {
        f64::sqrt(isize::pow(si1 - si2, 2) as f64 + isize::pow(sj1 - sj2, 2) as f64)
    }

    fn main_dir_params(distance: ArrayView2<'_, f64>, height: ArrayView2<'_, f64>, shifts: &Vec<Vec<(isize, isize)>>, dir1: usize, dir2: usize, cellsize: f64, progress: &ProgressBar) -> Array2<(usize, f64, f64)> {

        let shape = distance.raw_dim();
        let nrow = shape[0];
        let ncol = shape[1];
        let ndirs = shifts.len();
        let ndirs2 = ndirs/2;

        // dir, len, cossum
        let mut result = Array2::<(usize, f64, f64)>::from_elem(shape, (0, 0.0, 0.0));

        let mut new_dir = Array2::from_elem(shape,true);
        let mut new_len = Array2::<f64>::zeros(shape);

        let mut sk: usize;
        let (mut ik, mut jk): (isize, isize);
        let (mut uik, mut ujk): (usize, usize);
        let mut length: f64;

        let proc_dirs = dir2 - dir1;
        let mut cur_dir = 1;

        for dk in dir1..dir2 {
            let sh = vec![&shifts[dk], &shifts[ndirs2+dk]];
            let mut sidx = vec![0_usize, 0_usize];
            let mut h = vec![0.0, 0.0];

            progress.set_position(0);
            progress.set_message(format!("direction {}/{}", cur_dir, proc_dirs));

            for i in 0..nrow {
                for j in 0..ncol {
                    if distance[[i, j]] <= f64::EPSILON || new_len[[i, j]] > 0.0 {
                        continue;
                    }

                    sidx[0] = 0;
                    sidx[1] = 0;
                    h[0] = 0.0;
                    h[1] = 0.0;
                    for k in 0..2 {
                        sk = 0;
                        for s in sh[k] {
                            ik = i as isize + s.0;
                            jk = j as isize + s.1;
                            if is_within(ik, jk, nrow, ncol) {
                                uik = ik as usize;
                                ujk = jk as usize;
                                if distance[[uik, ujk]] <= f64::EPSILON {
                                    sidx[k] = sk;
                                    h[k] = height[[uik, ujk]];
                                    break;
                                }
                            } else {
                                sidx[k] = sk;
                                h[k] = 0.0;
                                break;
                            }

                            sk += 1;
                        }
                    }

                    length = cellsize * get_length(
                        sh[0][sidx[0]].0, sh[0][sidx[0]].1,
                        sh[1][sidx[1]].0, sh[1][sidx[1]].1
                    );

                    for k in 0..2 {
                        let s = &sh[k];
                        for sk in 0..sidx[k] {
                            uik = (i as isize + s[sk].0) as usize;
                            ujk = (j as isize + s[sk].1) as usize;

                            if length > new_len[[uik, ujk]] {
                                new_len[[uik, ujk]] = length;
                            }

                            if new_dir[[uik, ujk]] {
                                result[[uik, ujk]].2 +=
                                    f64::powi(
                                    f64::cos(
                                        f64::atan2(
                                            h[k],
                                            cellsize * get_length(
                                                s[sidx[k]].0, s[sidx[k]].1, s[sk].0, s[sk].1
                                                )
                                            )
                                        ),
                                    2
                                    );
                                new_dir[[uik, ujk]] = false;
                            }
                        }
                    }

                    progress.inc(1);
                }
            }

            for i in 0..nrow {
                for j in 0..ncol {
                    if new_len[[i, j]] > result[[i, j]].1 {
                        result[[i, j]].1 = new_len[[i, j]];
                        result[[i, j]].0 = dk;
                    }
                }
            }

            new_len.fill(0.0);
            new_dir.fill(true);

            cur_dir += 1;

        }

        progress.finish_with_message("done");

        return result;
    }

    // possible numbers for discr: 1, 2, 3, 4, 5, 6, 9, 10, 12, 15, 18, 20, 30, 36, 45, 60, 90
    fn euclidean_length_params(distance: ArrayView2<'_, f64>, height: ArrayView2<'_, f64>, discr: f64, radius: f64, cellsize: f64) -> Array3<f64> {
        let shape = distance.raw_dim();
        let nrow = shape[0];
        let ncol = shape[1];
        let nelem = nrow * ncol;

        // let mut output = Array3::<f64>::zeros((5, nrow, ncol));
        let mut output = Array3::<f64>::zeros((3, nrow, ncol));

        let r = radius / cellsize;
        let mut a = 0.5 * PI;
        let ndirs = (360.0 / discr) as usize;
        let (mut i, mut j): (isize, isize);
        let mut shifts = Vec::new();

        for _ in 0..ndirs {
            i = (-r * f64::sin(a)) as isize;
            j = (r * f64::cos(a)) as isize;
            shifts.push(get_shifts(0, 0, i, j));
            a -= PI * discr as f64 / 180.0;
        }

        let arcdistance = distance.to_shared();
        let archeight = height.to_shared();

        let nproc = num_cpus::get_physical();

        let mut tasks = Vec::new();

        let m = MultiProgress::new();
        let sty = ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
        ).unwrap()
            .progress_chars("##-");

        let mut bars: Vec<ProgressBar> = Vec::new();
        let pb = m.add(ProgressBar::new(nelem as u64));
        pb.set_style(sty.clone());
        bars.push(pb);

        for proc in 1..nproc {
            let pb = m.insert_after(&bars[proc - 1], ProgressBar::new(nelem as u64));
            pb.set_style(sty.clone());
            bars.push(pb);
        }

        let proc_dirs = ndirs / (2 * nproc);

        let mut dir1 = 0_usize;
        let mut dir2 = proc_dirs + ndirs % (2 * nproc);

        for proc in 0..nproc {
            let distance_ref = arcdistance.clone();
            let height_ref = archeight.clone();
            let shifts_ref = shifts.clone();
            let bars_clone = bars.clone();

            tasks.push(thread::spawn(move || {
                main_dir_params(distance_ref.view(), height_ref.view(), &shifts_ref, dir1, dir2, cellsize, &bars_clone[proc])
            }));

            dir1 = dir2;
            dir2 += proc_dirs;
        }

        m.clear().unwrap();

        for task in tasks {
            let add = task.join().unwrap();
            for i in 0..nrow {
                for j in 0..ncol {
                    if add[[i, j]].1 > output[[1, i, j]] {
                        output[[0, i, j]] = discr * add[[i, j]].0 as f64;
                        output[[1, i, j]] = add[[i, j]].1;
                    }
                    output[[2, i, j]] += 200.0 * add[[i, j]].2 / ndirs as f64;
                }
            }
        }

        return output;
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
        let result = euclidean_distance(source, cellsize);
        result.into_pyarray(py)
    }

    // wrapper of `euclidean_distance
    #[pyfn(m)]
    #[pyo3(name = "euclidean_antidistance")]
    fn euclidean_antidistance_py<'py>(
        py: Python<'py>,
        source: PyReadonlyArray2<'_, f64>,
        cellsize: f64
    ) -> &'py PyArray2<f64> {
        let source = source.as_array();
        let result = euclidean_antidistance(source, cellsize);
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
        let result = euclidean_allocation(source, cellsize);
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
        source: PyReadonlyArray2<'_, f64>,
        cellsize: f64
    ) -> &'py PyArray2<f64> {
        let source = source.as_array();
        let dist = euclidean_distance(source, cellsize);
        let result = euclidean_width(dist.view(), cellsize);
        result.into_pyarray(py)
    }

    // wrapper of `euclidean_width`
    #[pyfn(m)]
    #[pyo3(name = "euclidean_width_split")]
    fn euclidean_width_split_py<'py>(
        py: Python<'py>,
        source: PyReadonlyArray2<'_, f64>,
        cellsize: f64,
        rows: usize,
        cols: usize,
        hard: bool
    ) -> &'py PyArray2<f64> {
        let source = source.as_array();
        let dist = euclidean_distance(source, cellsize);
        let result = euclidean_width_split(
            dist.view(), cellsize, rows, cols, hard
        );
        result.into_pyarray(py)
    }

    // wrapper of `euclidean_width`
    #[pyfn(m)]
    #[pyo3(name = "euclidean_width_tiles")]
    fn euclidean_width_tiles_py<'py>(
        py: Python<'py>,
        source: PyReadonlyArray2<'_, f64>,
        cellsize: f64,
        rows: usize,
        cols: usize,
        hard: bool
    ) -> &'py PyArray3<usize> {
        let dist = source.as_array();
        let result = euclidean_width_tiles(
            dist.view(), cellsize, rows, cols, hard
        );
        result.into_pyarray(py)
    }

    // wrapper of `euclidean_width_params`
    #[pyfn(m)]
    #[pyo3(name = "euclidean_width_params")]
    fn euclidean_width_params_py<'py>(
        py: Python<'py>,
        source: PyReadonlyArray2<'_, f64>,
        cellsize: f64
    ) -> &'py PyArray3<f64> {
        let source = source.as_array();
        let distance = euclidean_distance(source, cellsize);
        let height = euclidean_allocation(source, cellsize);
        let result = euclidean_width_params(
            distance.view(), height.view(), cellsize
        );
        result.into_pyarray(py)
    }

    // wrapper of `euclidean_width_params`
    #[pyfn(m)]
    #[pyo3(name = "euclidean_width_params_split")]
    fn euclidean_width_params_split_py<'py>(
        py: Python<'py>,
        distance: PyReadonlyArray2<'_, f64>,
        allocation: PyReadonlyArray2<'_, f64>,
        cellsize: f64,
        rows: usize,
        cols: usize,
        hard: bool
    ) -> &'py PyArray3<f64> {
        let distance = distance.as_array();
        let height = allocation.as_array();
        let result = euclidean_width_params_split(
            distance.view(), height.view(), cellsize, rows, cols, hard
        );
        result.into_pyarray(py)
    }

    // wrapper of `euclidean_width_params`
    #[pyfn(m)]
    #[pyo3(name = "euclidean_length_params")]
    fn euclidean_length_params_py<'py>(
        py: Python<'py>,
        distance: PyReadonlyArray2<'_, f64>,
        height: PyReadonlyArray2<'_, f64>,
        discr: f64,
        radius: f64,
        cellsize: f64
    ) -> &'py PyArray3<f64> {
        let distance = distance.as_array();
        let height = height.as_array();
        let result = euclidean_length_params(
            distance.view(), height.view(), discr, radius, cellsize
        );
        result.into_pyarray(py)
    }

    Ok(())
}
