// pyo3 bindings, exposes python module

use pyo3::prelude::*;

pub mod vocab;
pub mod pretokenize;

#[pymodule]
fn core(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}