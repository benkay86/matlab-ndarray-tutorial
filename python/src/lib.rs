//! Example of a Python extension module to let us call into Rust code from
//! Python using [numpy](https://numpy.org/) arrays.
//!
//! See https://pyo3.rs for more information.  Note that all exported functions
//! must be public (pub).
//!
//! To build this particular example:
//!
//! ```bash
//! # Enter the project directory.
//! cd /path/to/matlab-ndarray-tutorial/python
//! # Create a python virtual environment in `env` (can use any name).
//! python -m venv env
//! # Enter/activate the virtual environment.
//! # Use the `exit` command to leave the virtual environment.
//! source env/bin/activate
//! # Install maturin.
//! pip install maturin
//! # Use maturin to build the example.
//! # Maturin will run cargo for you.  You don't necessarily need to build the
//! # project in advance.
//! maturin develop
//! # Test out the devel build.
//! python
//! >>> # recall the extension module is named rust_ext
//! >>> import rust_ext
//! >>> from rust_ext import *
//! >>> add_scalars(1., 2.)
//! 3
//! >>> import numpy as np
//! >>> arr = np.array([[1., 2.], [3., 4.]])
//! >>> arr
//! array([[1., 2.],
//!        [3., 4.]])
//! >>> add_scalar_to_array(1., arr)
//! array([[2., 3.],
//!        [4., 5.]])
//! >>> # The array was not modified.
//! >>> arr
//! array([[1., 2.],
//!        [3., 4.]])
//! >>> add_scalar_to_array_inplace(1., arr)
//! >>> # Note the absence of a returned sum.
//! >>> # Instead, the array was modified in place.
//! >>> arr
//! array([[2., 3.],
//!        [4., 5.]])
//! >>> quit()
//! # Build a wheel that will work on the host machine.
//! maturin build
//! # Here is the wheel.
//! ls ../target/wheels
//! # To build for multiple target and publish, refer to:
//! # https://github.com/PyO3/maturin
//! exit
//! ```

extern crate blas_src;

use ndarray::{ArrayView, ArrayViewMut, IxDyn};
use numpy::{IntoPyArray, PyArray, PyReadonlyArray};
use pyo3::proc_macro::{pyclass, pyfunction, pymethods, pymodule, pyproto};
use pyo3::types::PyModule;
use pyo3::{wrap_pyfunction, PyObjectProtocol, PyResult, Python};

// Very simple function that takes and returns trivially copyable types.
// Pyo3 takes care of automatically converting between Python and Rust f64.
// Don't need the numpy crate for this example.
//
// Unfortunately, Python doesn't support overloaded or generic functions.
// So we have to pick a data type (here, f64) and can't for example, be generic
// over std::ops::Add.  Try passing an integer into this function on the Python
// side to get an error.
//
// Note the documentation comments for the function automatically become the
// Python function signature and help text.

/// add_scalars(x, y, /)
/// ---
///
/// Add two 64 bit floating points numbers x and y together.  Returns the sum.
#[pyfunction]
pub fn add_scalars(x: f64, y: f64) -> f64 {
    x + y
}

// More complex example where we add a scalar to each element of an array and
// return the sum.  Here we use some of the numpy crate's types to facilitate
// low-cost conversion between numpy and ndarray arrays.
//
// The PyReadonlyArray is actually a reference to the array's data, which is
// allocated on Python's heap.  Since we are going to hold a reference to
// Python's memory, we also need to hold the global interpreter lock (GIL),
// which is acquired by the first argument `py`.

/// add_scalar_to_array(scal, arr, /)
/// ---
///
/// Add the scalar value scal to each element of the array arr.
/// Returns the sum.
#[pyfunction]
pub fn add_scalar_to_array<'py>(
    py: Python<'py>,
    scal: f64,
    arr: PyReadonlyArray<f64, IxDyn>,
) -> &'py PyArray<f64, IxDyn> {
    // Convert the immutable reference to the array on Python's heap into a
    // read-only ArrayView.  The ArrayView still points to the memory
    // in Python, so it's essentially a zero-cost conversion.
    //
    // If we want to copy over the memory onto Rust's heap we could instead do:
    // let arr: Array<f64, IxDyn> = arr.to_owned_array();
    // But this involves an expensive reallocation, and is really only
    // necessary if we need to extend the lifetime of the array so that we can
    // release the GIL.
    let arr: ArrayView<f64, IxDyn> = arr.as_array();
    // Compute the sum with the usual ndarray syntax.
    // The variable sum is an Array<f64, IxDyn> allocated on Rusts's heap.
    let sum = &arr + scal;
    // Convert the sum back into a numpy array.  Note the use of the GIL token,
    // py, to perform the allocation on Python's heap.
    // This is a zero/low-cost conversion in that the array data will remain on
    // Rusts's heap (i.e. it isn't copied onto Python's heap).  Only the array's
    // metadata is allocated on Python's heap.  When Python eventually garbage
    // collects the returned array, Rust will automatically free the memory from
    // the Rust heap.
    return sum.into_pyarray(py);
}

// Even more complex example where we modify the array in place rather than
// storing the sum in a newly-allocated array.  Note that we must still acquire
// the GIL as the first argument, _py, but do not actually need to use it for
// anything.
//
// Note that in each of the above examples our function returned something.
// Pyo3 automatically took care of wrapping the return value in a PyResult
// for us.  This function doesn't return anything, so we must explicitly return
// PyResult<()>.

/// add_scalar_to_array_inplace(scal, arr, /)
/// ---
///
/// Add the scalar value scal to each element of the array arr.
/// Stores the sum in arr, modifying it in place.  Returns nothing.
#[pyfunction]
pub fn add_scalar_to_array_inplace(
    _py: Python<'_>,
    scal: f64,
    arr: &PyArray<f64, IxDyn>,
) -> PyResult<()> {
    // Uncomment the following line to get undefined behavior.
    // let other_arr: ArrayView<f64, IxDyn> = py.readonly().as_array();
    // Convert the numpy array to a mutable ndarray ArrayViewMut.
    // This is unsafe becausew we must guarantee to the compiler that no other
    // references to the array exist.  For example, if we create an immutable
    // reference to the array by uncommenting the line above, then the following
    // unsafe code would have undefined behavior.
    let mut arr: ArrayViewMut<f64, IxDyn> = unsafe { arr.as_array_mut() };
    // Compute the sum in place.
    arr += scal;
    // Return success.
    Ok(())
}

// So far we've just created free functions that can be called from Python.
// Let's get more sophisticated and create a Python class from a Rust struct.
//
// First the struct.

/// Point(x, y, /)
/// ---
///
/// Create a new point with coordinates x and y.
#[pyclass]
#[derive(Clone)]
struct Point {
    // The #[pyo3(get, set)] make automatic getters and setters from us so the
    // caller can "get" the value with `mypoint.x`
    // and "set" it with `mypoint.x = 0.`.
    #[pyo3(get, set)]
    x: f64,
    #[pyo3(get, set)]
    y: f64,
}

// Now the class methods.
#[pymethods]
impl Point {
    // Tell Python this is the "new" method for constructing the class.
    // Also provide default arguments of zero for x and y.
    #[new]
    #[args(x = "0.", y = "0.")]
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// distance(other, /)
    /// ---
    ///
    /// Compute the distance between this point to and another point, other.
    /// If no other point is given, computes distance from the origin (0, 0).
    #[args(other = "None")]
    fn distance(&self, other: Option<Self>) -> f64 {
        let other = other.unwrap_or(Self { x: 0., y: 0. });
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        f64::sqrt(dx * dx + dy * dy)
    }
}

// We can make our class inspectable and printable by implementing the
// respective Python "protocols."  Although we don't need it in this example,
// if you write a Python class that holds references to other Python objects,
// read about how to implement the garbage collector protocol:
// https://pyo3.rs/master/class/protocols.html
#[pyproto]
impl PyObjectProtocol for Point {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Point ({}, {})", self.x, self.y))
    }
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("({}, {})", self.x, self.y))
    }
}

// Here is where we actually create the python module.
// The function to create the module should have the same name as the module.
// We add each function to the module using PyModule::add_wrapped() combined
// with the wrap_pyfunction! macro.
// We add our class to the module in one fell swoop with PyModule::add_class().
#[pymodule]
pub fn rust_ext(_: Python, module: &PyModule) -> PyResult<()> {
    module.add_wrapped(wrap_pyfunction!(add_scalars))?;
    module.add_wrapped(wrap_pyfunction!(add_scalar_to_array))?;
    module.add_wrapped(wrap_pyfunction!(add_scalar_to_array_inplace))?;
    module.add_class::<Point>()?;
    Ok(())
}
