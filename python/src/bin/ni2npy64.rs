//! Somewhat more sophisticated example of calling Python from Rust.
//! Compiles to a Rust executable that takes two command line arguments:
//!
//! ni2npy64 input output.npy
//!
//! Uses Python's nibabel to read the data from the input file (can be any
//! neuroimaging file format supported by nibabel) into a Rust ndarray.
//!
//! Converts the data type to f64 and writes the array out to a numpy file.
//!
//! Per [this workaround](https://pyo3.rs/v0.14.2/faq.html#i-cant-run-cargo-test-im-having-linker-issues-like-symbol-not-found-or-undefined-reference-to-_pyexc_systemerror),
//! run this example with:
//!
//! ```ignore
//! cargo run --no-default-features --bin ni2npy64
//! ```

extern crate blas_src;

use ndarray::{Array, IxDyn};
use ndarray_npy::WriteNpyExt;
use numpy::PyReadonlyArray;
use pyo3::exceptions::PyModuleNotFoundError;
use pyo3::types::PyModule;
use pyo3::{IntoPy, Py, PyAny, PyErr, Python};
use std::fs::File;
use std::path::{Path, PathBuf};

// Type-erased error for easy use of the ? operator.
type AnyError = Box<dyn std::error::Error + Send + Sync>;

/// Invalid command line arguments.
pub struct CmdArgsError {
    /// Number of arguments passed on the command line,
    /// not counting the name of the program itself.
    pub n_args: usize,
}
impl std::fmt::Display for CmdArgsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.n_args == 0 {
            // Use ANSI VT100 terminal escape characters to erase
            // Rust's auto-generated error message, and display
            // informational message instead.
            writeln!(f, "\x1B[2K\x1B[1000DExtract 64-bit float data from neuroimaging file into numpy array file.")?;
        } else {
            // Keep the error message and explain what happened.
            writeln!(
                f,
                "Wrong number of command line arguments: found {}, expected 2.",
                self.n_args
            )?;
        }
        write!(f, "Usage: ni2py64 input output.npy")
    }
}
impl std::fmt::Debug for CmdArgsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as std::fmt::Display>::fmt(self, f)
    }
}
impl std::error::Error for CmdArgsError {}

/// Unsupported data type (dtype) in numpy array.
pub struct DTypeError {
    /// The name of the unsupported datatype, if any.  E.g. "float64"
    pub name: Option<String>,
    /// The Python dtype variable we couldn't support.
    pub dtype: Py<PyAny>,
}
impl std::fmt::Display for DTypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.name {
            Some(ref name) => write!(f, "Unsupported dtype \"{}\".", name),
            None => write!(f, "Unsupported dtype."),
        }
    }
}
impl std::fmt::Debug for DTypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as std::fmt::Display>::fmt(self, f)
    }
}
impl std::error::Error for DTypeError {}

/// Python can't find the nibabel module.
pub struct NibabelNotFoundError {
    // The Python exception that caused this error,
    // which is guaranteed to be a PyModuleNotFoundError.
    py_err: PyErr,
    // Storing a PyErr instead of a Py<PyModuleNotFoundError> makes it easier
    // for us to be thread-safe while also implementing
    // std::error::Error::source().
}
impl NibabelNotFoundError {
    /// Create a new NibabelNotFoundError.
    pub fn new<'py>(py_err: &'py PyModuleNotFoundError, py: Python<'py>) -> Self {
        // Obtain a GIL-independent reference to the Python exception
        // and upcast to a PyErr.
        let py_err: Py<PyModuleNotFoundError> = py_err.into_py(py);
        Self {
            py_err: PyErr::from_instance(py_err.as_ref(py)),
        }
    }
    /// Get the Python exception that caused this error.
    pub fn py_err(&self) -> &PyErr {
        &self.py_err
    }
    /// Get the Python exception that caused this error,
    /// which is guaranteed to be a PyModuleNotFoundError.
    pub fn downcast_py_err<'py>(&'py self, py: Python<'py>) -> &'py PyModuleNotFoundError {
        self.py_err
            .instance(py)
            .downcast::<PyModuleNotFoundError>()
            // Won't panic because PyErr was constructed from PyModuleNotFoundError.
            .unwrap()
    }
}
impl<'py> std::fmt::Display for NibabelNotFoundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Python could not find the nibabel module.  Did you install nibabel?"
        )
    }
}
impl<'py> std::fmt::Debug for NibabelNotFoundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as std::fmt::Display>::fmt(self, f)
    }
}
impl<'py> std::error::Error for NibabelNotFoundError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.py_err as &(dyn std::error::Error + 'static))
    }
}

// Process command line arguments, returning a tuple (infile, outfile).
fn process_cmd_args() -> Result<(PathBuf, PathBuf), CmdArgsError> {
    // Get an iterator over the command line arguments.
    let mut cmd_args = std::env::args_os();
    // Check that we have the correct number of arguments.
    // The first "argument" is actually the name of the binary,
    // so we need 3 "arguments" to get the 2 file name arguments.
    if cmd_args.len() != 3 {
        Err(CmdArgsError {
            n_args: cmd_args.len() - 1,
        })?;
    }
    // Extract the input and output file names from the iterator.
    // Unwrapping guaranteed not to panic because we checked to make sure
    // there are 3 items in the iterator.
    let infile = cmd_args.nth(1).unwrap().into();
    let outfile = cmd_args.next().unwrap().into();
    Ok((infile, outfile))
}

// Write the ndarray `arr` to a numpy array file named `filename`.
fn write_npy<T, P>(arr: &T, filename: P) -> Result<(), AnyError>
where
    T: WriteNpyExt,
    P: AsRef<Path>,
{
    // Open the file for writing.
    let writer = File::create(filename)?;
    // Get the super-easy-to-use ndarray_npy crate to do the work for us!
    arr.write_npy(writer)?;
    Ok(())
}

fn main() -> Result<(), AnyError> {
    // Extract input and output file names from command line arguments.
    let (infile, outfile) = process_cmd_args()?;

    // Initialize the Python interpreter.
    // Calling this more than once will have no effect after the first call.
    pyo3::prepare_freethreaded_python();

    // Acquire the global interpreter lock.
    // The lock will automatically released when `gil_guard` is dropped,
    // in this case near the end of the program.
    let gil_guard = Python::acquire_gil();
    let py = gil_guard.python(); // token bound to the lifetime of the GIL

    // Import the nibabel module.  If this doesn't work then generate
    // a friendly error message reminding the user to install nibabel.
    let nibabel = match PyModule::import(py, "nibabel") {
        Err(py_err) => match py_err.instance(py).downcast::<PyModuleNotFoundError>() {
            // Expect a PyModuleNotFoundError.
            Ok(mod_err) => Err(NibabelNotFoundError::new(mod_err, py))?,
            // Pass through other exceptions.
            Err(_) => Err(py_err)?,
        },
        Ok(nibabel) => nibabel,
    };

    // Get nibabel to load the image for us.
    let img = nibabel.call_method("load", (infile,), None)?;

    // Get the "data" portion of the image, a numpy array.
    // The `get_fdata()` method should nominally do the hard work of converting
    // the data type to f64 for us.
    let data = img.call_method("get_fdata", (), None)?;

    // Get the data type of the image data.
    let dtype = data.getattr("dtype")?; // The data type as a python object.
    let dtype_name: &str = dtype
        .getattr("name") // Try to get the "name" of the data type as a string.
        .or_else(|_| {
            Err(DTypeError {
                name: None,
                dtype: dtype.into_py(py),
            })
        })?
        .extract()?; // "Extract" the Python string into a Rust &str.

    // Try to convert the data to ArrayBase<f64, _>.
    // Then use the ndarray_npy crate to write the array out to a numpy file.
    if dtype_name == "float64" {
        // The data is already f64, so no need to do any conversion.
        // "Extract" PyReadonlyArray from the Python object.
        // Call as_array() to convert to an ArrayView<f64, _>, a read-only
        // Rust array view that points to the array data in Python's memory.
        // Use the ndarray_npy crate to write the array to a file.
        let data: PyReadonlyArray<f64, IxDyn> = data.extract()?;
        write_npy(&data.as_array(), outfile)?;
    } else if dtype_name == "float32" {
        // Extract a PyReadonlyArray<f32, _> and get a read-only Rust array view
        // that points to the array data in Python's memory.  Then map
        // to an Array<f64, _> on Rusts's memory heap.
        let data32: PyReadonlyArray<f32, IxDyn> = data.extract()?;
        let data64: Array<f64, IxDyn> = data32.as_array().map(|&el| el as f64);
        // The program is about to end anyway, but as an instructive example,
        // now that the data is in Rust's memory, we can drop all references to
        // the Python data.  Note that none of the memory is actually released
        // to Python's garbage collector until we drop the GILGuard.
        drop(data32);
        drop(data);
        drop(py);
        drop(gil_guard);
        // Finally, after dropping gil_guard, the memory is released!
        // Write array data from Rust's memory heap to the file.
        write_npy(&data64, outfile)?;
    } else {
        // You can edit this program to support converting more data types to
        // f64, but for now we'll just raise an error for unsupported types.
        Err(DTypeError {
            name: Some(dtype_name.to_string()),
            dtype: dtype.into_py(py),
        })?;
    }

    // All done. Return success.
    Ok(())

    // If the GIL wasn't already released above via drop(gil_guard) it will be
    // released here when the stack unwinds and the GILGuard is dropped.
}
