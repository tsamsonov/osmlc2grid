use std::cmp::{max, min};
use pyo3::prelude::*;
use numpy::ndarray::{s, Array2, ArrayView2, Array3, Axis};
use numpy::{PyArray2, PyReadonlyArray2, PyArray3, IntoPyArray};
use std::collections::{BinaryHeap};
use ordered_float::OrderedFloat;
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

    fn euclidean_width(input: ArrayView2<'_, f64>, cellsize: f64) -> Array2<f64> {
        let shape = input.raw_dim();
        let nrow = shape[0];
        let ncol = shape[1];
        let mut output = Array2::<f64>::zeros((nrow, ncol));

        let arcoutput = output.to_shared();
        let arcinput = input.to_shared();

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

    fn euclidean_width_tiles(input: ArrayView2<'_, f64>, cellsize: f64, rows: usize, cols: usize) -> Array3<usize> {
        let shape = input.raw_dim();
        let nrow = shape[0];
        let ncol = shape[1];
        let mut output = Array3::<usize>::zeros((rows, cols, 8));

        let drow = nrow / rows;
        let dcol = ncol / cols;

        let (mut row1, mut row2, mut col1, mut col2): (usize, usize, usize, usize);
        let (mut rowmin, mut colmin, mut rowmax, mut colmax, mut delta, mut radius):
            (isize, isize, isize, isize, isize, isize);

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

                for i in row1..row2 {
                    for j in col1..col2 {
                        radius = (input[[i, j]] / cellsize) as isize;

                        delta = i as isize - radius;
                        if delta < rowmin {
                            rowmin = delta;
                        }

                        delta = j as isize - radius;
                        if delta < colmin {
                            colmin = delta;
                        }

                        delta = i as isize + radius;
                        if delta > rowmax {
                            rowmax = delta;
                        }

                        delta = j as isize + radius;
                        if delta > colmax {
                            colmax = delta;
                        }
                    }

                    rowmin = max(0, rowmin);
                    colmin = max(0, colmin);

                    rowmax = min((nrow - 1) as isize, rowmax);
                    colmax = min((ncol - 1) as isize, colmax);
                }

                output[[row, col, 0]] = row1;
                output[[row, col, 1]] = row2;
                output[[row, col, 2]] = col1;
                output[[row, col, 3]] = col2;
                output[[row, col, 4]] = rowmin as usize;
                output[[row, col, 5]] = rowmax as usize;
                output[[row, col, 6]] = colmin as usize;
                output[[row, col, 7]] = colmax as usize;

                println!("[{}, {}] × [{}, {}] -> [{}, {}] × [{}, {}] = {} × {}",
                         row1, row2, col1, col2, rowmin, rowmax, colmin, colmax,
                         rowmax-rowmin, colmax-colmin);

                col1  = col2;
                col2 += dcol;
            }
            row1  = row2;
            row2 += drow;
        }

        return output;
    }

    fn euclidean_width_split(input: ArrayView2<'_, f64>, cellsize: f64, rows: usize, cols: usize) -> Array2<f64> {
        let shape = input.raw_dim();
        let nrow = shape[0];
        let ncol = shape[1];
        let mut output = Array2::<f64>::zeros((nrow, ncol));

        let drow = nrow / rows;
        let dcol = ncol / cols;

        let (mut row1, mut row2, mut col1, mut col2): (usize, usize, usize, usize);
        let (mut rowmin, mut colmin, mut rowmax, mut colmax, mut delta, mut radius):
            (isize, isize, isize, isize, isize, isize);

        let mut slices = Array2::default((nrow, ncol));
        let mut ext_slices = Array2::default((nrow, ncol));

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

                for i in row1..row2 {
                    for j in col1..col2 {
                        radius = (input[[i, j]] / cellsize) as isize;

                        delta = i as isize - radius;
                        if delta < rowmin {
                            rowmin = delta;
                        }

                        delta = j as isize - radius;
                        if delta < colmin {
                            colmin = delta;
                        }

                        delta = i as isize + radius;
                        if delta > rowmax {
                            rowmax = delta;
                        }

                        delta = j as isize + radius;
                        if delta > colmax {
                            colmax = delta;
                        }
                    }

                    rowmin = max(0, rowmin);
                    colmin = max(0, colmin);

                    rowmax = min((nrow - 1) as isize, rowmax);
                    colmax = min((ncol - 1) as isize, colmax);
                }

                slices[[row,col]] = (row1, row2, col1, col2);
                ext_slices[[row,col]] = (rowmin as usize, rowmax as usize, colmin as usize, colmax as usize);

                col1  = col2;
                col2 += dcol;
            }
            row1  = row2;
            row2 += drow;
        }

        for row in 0..rows {
            for col in 0..cols {
                let bbox = slices[[row, col]];
                let ebox = ext_slices[[row, col]];

                println!("[{}, {}] × [{}, {}] -> [{}, {}] × [{}, {}]",
                         bbox.0, bbox.1, bbox.2, bbox.3, ebox.0, ebox.1, ebox.2, ebox.3);

                let tile = euclidean_width(input.slice(s![ebox.0..ebox.1, ebox.2..ebox.3]), cellsize);

                for i in bbox.0..bbox.1 {
                    for j in bbox.2..bbox.3 {
                        output[[i, j]] = tile[[i - ebox.0, j - ebox.2]];
                    }
                }
                println!("{}, {}", row, col);
            }
        }

        return output;
    }

    fn euclidean_width2(input: ArrayView2<'_, f64>, cellsize: f64) -> Array2<f64> {
        let shape = input.raw_dim();
        let nrow = shape[0];
        let ncol = shape[1];
        let mut output = Array2::<f64>::zeros((nrow, ncol));

        let dist = euclidean_distance(input, cellsize);

        let arcinput = dist.to_shared();

        let nproc = num_cpus::get_physical();

        let mut tasks = Vec::new();
        let (tx, rx) = mpsc::channel();

        for proc in 0..nproc {
            let input_ref = arcinput.clone();
            let tx1 = tx.clone();

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
                    let mut covered: Vec<(usize, usize)> = Vec::new();

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

                            covered.push((uik, ujl));
                        }
                    }

                    tx1.send((radius * 2.0, covered)).unwrap();

                }
            }))
        }

        drop(tx);

        for (diameter, covered) in rx {
            for idx in covered {
                if diameter > output[idx] {
                    output[idx] = diameter;
                }
            }
        }

        return output;
    }

    fn euclidean_width_params(distance: ArrayView2<'_, f64>, height: ArrayView2<'_, f64>, cellsize: f64) -> Array3<f64> {
        let shape = distance.raw_dim();
        let nrow = shape[0];
        let ncol = shape[1];

        let mut output = Array3::<f64>::zeros((5, nrow, ncol));
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
                let (mut hsum, mut hmean, mut hw, mut radius): (f64, f64, f64, f64);
                let (mut w, mut ik, mut jl): (isize, isize, isize);
                let (mut uik, mut ujl): (usize, usize);
                let mut n: usize;

                let mut queue = BinaryHeap::new();

                for i in (0..nrow).filter(|r| r % nproc == proc) {
                    for j in 0..ncol {
                        if distance_ref[[i, j]] > 0.0 {
                            queue.push((OrderedFloat(distance_ref[[i, j]]), i, j));
                        }
                        output_ref[[0, i, j]] = -1.0;
                    }
                }

                bars_clone[proc].set_length(queue.len() as u64);

                while queue.len() > 0 {
                    let (ord_radius, i, j) = queue.pop().unwrap();

                    bars_clone[proc].inc(1);

                    // radius = f64::min(500.0, f64::from(ord_radius));
                    radius = f64::from(ord_radius);

                    if 0.5 * output_ref[[1, i, j]] > (output_ref[[2, i, j]] + radius) {
                        continue;
                    }

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

                            if output_ref[[1, uik, ujl]] < 2.0 * radius {
                                covered.push((uik, ujl));
                            }

                            hsum += height_ref[[uik, ujl]];
                            n += 1;
                        }
                    }
                    if covered.len() > 0 {
                        hmean = hsum / n as f64;
                        hw = 0.5 * hmean / radius;
                        for (uik, ujl) in covered {
                            output_ref[[0, uik, ujl]] = (i * ncol + j) as f64;
                            output_ref[[1, uik, ujl]] = 2.0 * radius;
                            output_ref[[2, uik, ujl]] =
                                cellsize * ((i as f64 - uik as f64).powi(2) +
                                (j as f64 - ujl as f64).powi(2)).sqrt();
                            output_ref[[3, uik, ujl]] = hmean;
                            output_ref[[4, uik, ujl]] = hw;
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
                    if add[[1, i, j]] > output[[1, i, j]] {
                        output[[0, i, j]] = add[[0, i, j]];
                        output[[1, i, j]] = add[[1, i, j]];
                        output[[2, i, j]] = add[[2, i, j]];
                        output[[3, i, j]] = add[[3, i, j]];
                        output[[4, i, j]] = add[[4, i, j]];
                    }
                }
            }
        }

        return output
    }

    fn euclidean_width_params_split(distance: ArrayView2<'_, f64>, height: ArrayView2<'_, f64>, cellsize: f64, rows: usize, cols: usize) -> Array3<f64> {
        let shape = distance.raw_dim();
        let nrow = shape[0];
        let ncol = shape[1];
        let mut output = Array3::<f64>::zeros((4, nrow, ncol));
        for i in 0..nrow {
            for j in 0..ncol {
                output[[0, i, j]] = -1.0;
            }
        }

        let drow = nrow / rows;
        let dcol = ncol / cols;

        let (mut row1, mut row2, mut col1, mut col2): (usize, usize, usize, usize);
        let (mut rowmin, mut colmin, mut rowmax, mut colmax, mut delta, mut radius):
            (isize, isize, isize, isize, isize, isize);

        let mut slices = Array2::default((nrow, ncol));
        let mut ext_slices = Array2::default((nrow, ncol));

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

                for i in row1..row2 {
                    for j in col1..col2 {
                        radius = (distance[[i, j]] / cellsize) as isize;

                        delta = i as isize - radius;
                        if delta < rowmin {
                            rowmin = delta;
                        }

                        delta = j as isize - radius;
                        if delta < colmin {
                            colmin = delta;
                        }

                        delta = i as isize + radius;
                        if delta > rowmax {
                            rowmax = delta;
                        }

                        delta = j as isize + radius;
                        if delta > colmax {
                            colmax = delta;
                        }
                    }

                    rowmin = max(0, rowmin);
                    colmin = max(0, colmin);

                    rowmax = min((nrow - 1) as isize, rowmax);
                    colmax = min((ncol - 1) as isize, colmax);
                }

                slices[[row,col]] = (row1, row2, col1, col2);
                ext_slices[[row,col]] = (rowmin as usize, rowmax as usize, colmin as usize, colmax as usize);

                col1  = col2;
                col2 += dcol;
            }
            row1  = row2;
            row2 += drow;
        }

        for row in 0..rows {
            for col in 0..cols {
                let bbox = slices[[row, col]];
                let ebox = ext_slices[[row, col]];

                println!("[{}, {}] × [{}, {}] -> [{}, {}] × [{}, {}]",
                         bbox.0, bbox.1, bbox.2, bbox.3, ebox.0, ebox.1, ebox.2, ebox.3);

                let tile = euclidean_width_params(
                    distance.slice(s![ebox.0..ebox.1, ebox.2..ebox.3]),
                    height.slice(s![ebox.0..ebox.1, ebox.2..ebox.3]),
                    cellsize
                );

                let ecols = ebox.3 - ebox.2;

                for i in bbox.0..bbox.1 {
                    for j in bbox.2..bbox.3 {
                        let eij = tile[[0, i - ebox.0, j - ebox.2]];
                        if eij > 0.0 {
                            let ei = ebox.0 + eij as usize / ecols;
                            let ej = ebox.2 + eij as usize % ecols;
                            output[[0, i, j]] = (ei * ncol + ej) as f64;
                        }
                        output[[1, i, j]] = tile[[1, i - ebox.0, j - ebox.2]];
                        output[[2, i, j]] = tile[[2, i - ebox.0, j - ebox.2]];
                        output[[3, i, j]] = tile[[3, i - ebox.0, j - ebox.2]];
                    }
                }
                println!("{}, {}", row, col);
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
    #[pyo3(name = "euclidean_width2")]
    fn euclidean_width2_py<'py>(
        py: Python<'py>,
        source: PyReadonlyArray2<'_, f64>,
        cellsize: f64
    ) -> &'py PyArray2<f64> {
        let source = source.as_array();
        let result = euclidean_width2(source, cellsize);
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
        cols: usize
    ) -> &'py PyArray2<f64> {
        let source = source.as_array();
        let dist = euclidean_distance(source, cellsize);
        let result = euclidean_width_split(dist.view(), cellsize, rows, cols);
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
        cols: usize
    ) -> &'py PyArray3<usize> {
        let dist = source.as_array();
        let result = euclidean_width_tiles(dist.view(), cellsize, rows, cols);
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
        let mut result = euclidean_width_params(distance.view(), height.view(), cellsize);
        result.push(Axis(0), distance.view()).unwrap();
        result.push(Axis(0), height.view()).unwrap();
        result.into_pyarray(py)
    }

    // wrapper of `euclidean_width_params`
    #[pyfn(m)]
    #[pyo3(name = "euclidean_width_params_split2")]
    fn euclidean_width_params_split2_py<'py>(
        py: Python<'py>,
        distance: PyReadonlyArray2<'_, f64>,
        allocation: PyReadonlyArray2<'_, f64>,
        cellsize: f64,
        rows: usize,
        cols: usize
    ) -> &'py PyArray3<f64> {
        let distance = distance.as_array();
        let height = allocation.as_array();
        let mut result = euclidean_width_params_split(distance.view(), height.view(), cellsize, rows, cols);
        result.push(Axis(0), distance.view()).unwrap();
        result.push(Axis(0), height.view()).unwrap();
        result.into_pyarray(py)
    }


    Ok(())
}
