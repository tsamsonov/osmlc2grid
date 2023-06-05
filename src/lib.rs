use pyo3::prelude::*;
use numpy::ndarray::{ArrayD, ArrayViewD};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn, IntoPyArray};

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn rasterspace(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    fn rescale(source: ArrayViewD<'_, f64>) -> ArrayD<f64> {
        10_f64 * &source
    }

    // wrapper of `rescale`
    #[pyfn(m)]
    #[pyo3(name = "rescale")]
    fn rescale_py<'py>(
        py: Python<'py>,
        source: PyReadonlyArrayDyn<'_, f64>
    ) -> &'py PyArrayDyn<f64> {
        let source = source.as_array();
        let result = rescale(source);
        result.into_pyarray(py)
    }

    Ok(())
}
