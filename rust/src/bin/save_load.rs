//! Serialize and deserialize an ndarray array to an npy file using ndarray-npy.
//!
//! Note that Matlab does not support the numpy *.npy file format natively.
// You must use <https://github.com/kwikteam/npy-matlab> and then:
//!
//! ```matlab
//! addpath npy-matlab
//! % now we can read and write numpy files
//! mat = [1,2; 3,4]
//! writeNPY(mat, 'mat.npy');
//! mat = readNPY('mat.npy')
//! ```
//!
//! As you know, Matlab has its own "matfile" file format *.mat.  Unfortunately,
//! the matfile format is actually several different file formats (depending on
//! the matfile "version") smushed into one.  The file format is poorly
//! documented and, in some cases, requires a license-encumbered API to access.
//! The matfile format is also less efficient than numpy for storing numeric
//! arrays in the majority of cases, so using numpy is often better anyway.
//!
//! There is a [matfile crate](https://github.com/dthul/matfile) with some
//! limited support for reading numeric arrays from the most common matfile
//! "versions."
//!
//! You can also use the much-more-complicated-than-it-needs-to-be filesystem-
//! in-a-file [hdf5](https://github.com/aldanor/hdf5-rust) format to read and
//! write so called "version 7.3" matfiles, although again Matlab's
//! documentation is sorely lacking.
//!
//! Unfortunately, ndarray does not have a native concept of higher-order data
//! structures such as tables and cell arrays.  Presently there is no obvious
//! way to share such data with Matlab.  The Apache Arrow ecosystem shows much
//! promise, and ndarray will probably integrate with it soon:
//! <https://github.com/rust-ndarray/ndarray/issues/771>

extern crate blas_src;
use ndarray::*;
use ndarray_npy::{ReadNpyExt, WriteNpyExt};
use std::fs::File;

fn main() {
    // Make a 2x2 matrix of double-precision floating point numbers.
    //
    // ```matlab
    // mat = [1,2; 3,4]
    // ```
    let mat: Array<f64, Dim<[Ix; 2]>> = array![[1., 2.], [3., 4.]];
    println!("Original mat =\n{:?}", mat);

    // Serialize to npy file.
    // Note use of a scoped code block `{ }` to ensure `writer` is flushed when
    // it is dropped at the end of the code block, and before we try to read the
    // numpy file in the next code block.
    //
    // ```matlab
    // writeNPY(mat, 'mat.npy');
    // ```
    {
        let writer = File::create("mat.npy").unwrap();
        mat.write_npy(writer).unwrap();
    }

    // Deserialize from npy file.
    //
    // ```matlab
    // mat = readNPY('mat.npy')
    // ```
    let mat: Array<f64, Dim<[Ix; 2]>>;
    {
        let reader = File::open("mat.npy").unwrap();
        mat = Array::<f64, Dim<[Ix; 2]>>::read_npy(reader).unwrap();
    }
    println!("Deserialized mat =\n{:?}", mat);
}
