//! The "gotchas" of unexpectedly long lifetimes
//! when managing Python's memory with PyO3.
//! See <https://pyo3.rs>.
//!
//! Per [this workaround](https://pyo3.rs/v0.14.2/faq.html#i-cant-run-cargo-test-im-having-linker-issues-like-symbol-not-found-or-undefined-reference-to-_pyexc_systemerror),
//! run this example with:
//!
//! ```
//! cargo run --no-default-features --bin memory
//! ```

use pyo3::types::PyString;
use pyo3::{Py, PyResult, Python};

fn main() -> PyResult<()> {
    // Initialize the Python interpreter.
    pyo3::prepare_freethreaded_python();

    // Example 1: GIL-bound references have the same lifetime as the GIL.
    // (Or more precisely, the GILPool, if we use unsafe code as below.)
    // Their memory is not released until the GIL is dropped.
    Python::with_gil(|py| -> PyResult<()> {
        for _ in 0..10 {
            // The lifetime of `hello` is bound to the lifetime
            // of the GIL, `py`, and will outlive this loop.
            let hello: &PyString = py.eval("\"Hello World!\"", None, None)?.extract()?;
            println!("Python says: {}", hello.to_str()?);
            // The Rust variable `hello` is dropped here,
            // but its Python memory isn't released yet.
        }
        // At this point we have 10 copies of hello allocated on Python's heap.
        Ok(())
    })?; // Here the GIL is dropped and all 10 copies of hello are released.

    // Example 1 Redux: We can acquire and release the GIL with each iteration
    // of the loop.
    for _ in 0..10 {
        Python::with_gil(|py| -> PyResult<()> {
            let hello: &PyString = py.eval("\"Hello World!\"", None, None)?.extract()?;
            println!("Python says: {}", hello.to_str()?);
            Ok(())
        })?; // GIL is dropped here and Python memory for `hello` is released.
    }

    // Exaple 1 Redux: Acquiring and releasing the GIL with each iteration might
    // not be practical.  We can achieve the same effect by separating the
    // GILPool from the GIL, but this is unsafe so we will have to be careful
    // not to trigger undefined behavior.
    Python::with_gil(|py| -> PyResult<()> {
        for _ in 0..10 {
            // Create a new GILPool.  This is unsafe, be sure you read the docs
            // and understand what you are doing!
            let pool = unsafe { py.new_pool() };
            // Shadow the original `py` with a shorter-lifetime token from the
            // new pool.  This will help prevent us from accidentally creating
            // references using the original `py`, which would be one of the
            // many ways to get undefined behavior from this code.
            let py = pool.python();
            // The lifetime of `hello` is bound to the lifetime
            // of the GILPool, and will *not* outlive this loop.
            let hello: &PyString = py.eval("\"Hello World!\"", None, None)?.extract()?;
            println!("Python says: {}", hello.to_str()?);
            // The GILPool `pool` is dropped at the end of this iteration,
            // releasing the Python memory for `hello`.
        }
        // All the `hello` variables have already been released at this point.
        Ok(())
    })?; // The GIL is dropped.

    // Example 2: GIL-independent references are released immediately on drop if
    // we are holding the GIL, otherwise they are released when the GIL is
    // reacquired.
    //
    // First let's temporarily acquire the GIL and obtain a GIL-independent
    // reference to Python's heap.  By GIL-independent we mean that the Python
    // memory pointed to by `hello_py` will outlive this `with_gil()` closure.
    let hello: Py<PyString> =
        Python::with_gil(|py| py.eval("\"Hello World!\"", None, None)?.extract())?;
    // Some time later in our program, let's temporarily reacquire the GIL and
    // do something with the Python memory.
    println!(
        "Python says: {}",
        hello.as_ref(Python::acquire_gil().python()).to_str()?
    );
    // Now let's try to drop `hello`.  We can drop the `hello` variable in
    // Rust, but its Python memory isn't released yet.  Why not?  Because we're
    // not currently holding the GIL, so we can't do anything to Python's
    // memory right now.
    drop(hello);
    // Briefly acquire the GIL again.
    // Acquiring the GIL will automatically release `hello`'s memory.
    // Since we don't need the GIL for anything else, we'll drop it right after
    // we acquire it.
    drop(Python::acquire_gil());

    // Example 2 Redux: If we drop the `Py<T>` smart pointer *while* we are
    // holding the GIL then the memory will be released immediately.
    let hello: Py<PyString> =
        Python::with_gil(|py| py.eval("\"Hello World!\"", None, None)?.extract())?;
    // Using `Py<Pystring>::into_ref()` consumes `self`, effectively dropping
    // `hello` while the GIL is acquired.
    println!(
        "Python says: {}",
        hello.into_ref(Python::acquire_gil().python()).to_str()?
    );
    // `hello` is dropped and its Python memory has been released.

    Ok(())
}
