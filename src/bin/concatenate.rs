//! Making arrays bigger.
//!
//! Currently this requires copying into a newly allocated array.
//! Starting with ndarray 0.15 we can reallocate an existing array with:
//! [`ArrayBase::append()`](https://docs.rs/ndarray/0.15.3/ndarray/struct.ArrayBase.html#method.append)

extern crate blas_src;
use ndarray::{array, concatenate, stack, Array, Axis};

fn main() {
    // Generate some matrices to work with.
    //
    // ```matlab
    // mat1 = [1,2,3; 4,5,6]
    // mat2 = [7, 8, 9]
    // mat3 = [10; 11]
    // ```
    let mat1 = array![[1., 2., 3.], [4., 5., 6.]];
    let mat2 = array![[7., 8., 9.]];
    let mat3 = array![[10.], [11.]];
    println!("Array 1:\n{:?}", mat1);
    println!("Array 2:\n{:?}", mat1);
    println!("Array 3:\n{:?}\n", mat2);

    // In Matlab we can make a matrix bigger simply by assigning to it.  When
    // the matrix is enlarged, new elements are initialized to zero before
    // assignment.
    //
    // ```
    // mat = [1,2,3]; size(mat) % 1 3
    // mat(4,4) = 1
    // % 1     2     3     0
    // % 0     0     0     0
    // % 0     0     0     0
    // % 0     0     0     1
    // size(mat) % 4 4
    // ```
    //
    // In Rust (and Python's) ndarray*, we cannot change the memory allocation
    // after the array is created.  That is to say, we can neither shrink nor
    // grow the array.  However, we can make a new array by concatenating two
    // existing arrays.  Note that this does *not* mutate either of the existing
    // arrays.  The concatenate operation will fail if the arrays are of
    // incompatible shapes.
    //
    // *In ndarray 0.15 `ArrayBase::append()` can enlarge the memory allocation
    // of an existing array.
    //
    // ```matlab
    // mat = [mat1; mat2] % concatenate rows
    // mat = [mat1, mat3] % concatenate columns
    // ```
    let mat = concatenate(Axis(0), &[mat1.view(), mat2.view()]).unwrap();
    println!("Concatenate arrays 1 and 2 (rows):\n{:?}", mat);
    // There's also a macro for this if typing `view()` and `unwrap()` is making
    // your fingers tired.
    let mat = concatenate![Axis(1), mat1, mat3];
    println!("Concatenate arrays 1 and 3 (columns):\n{:?}", mat);
    // Studying the above closely, we see that `concatenate` is taking read-only
    // view of the arrays.  That means, best case scenario, at least one copy of
    // mat, mat1, and mat2 (or mat3) exists in memory at the same time.  This is
    // potentially quite wasteful if we are working with big arrays!
    //
    // If we know the size of the final array we want in advance, then it's
    // most efficient to allocate that array at the start and assign values into
    // it as we go.
    //
    // If we just don't know the size in advance then, if we concatenate along
    // the (row) major dimension of memory layout, we can be more memory
    // efficient by converting to and from `Vec`.  Note that this won't work to
    // append columns.
    //
    // *Again, in ndarray 0.15 this is no longer necessary since we can just
    // use `ArrayBase::append()`.`
    //
    // First we convert to raw vectors.  This does not invole any reallocation.
    let mut vec1 = mat1.into_raw_vec();
    let mut vec2 = mat2.into_raw_vec();
    // Now we append to vec1 by moving elements out of vec2.
    // This *will* require a reallocation of vec1 and a memcpy() of vec2, and it
    // *may* require a memcpy() of vec1 if it is reallocated into a new starting
    // address, but if we are lucky then the memory block for vec1 is simply`
    // extended.
    vec1.append(&mut vec2);
    // Drop the empty shell of vec2.
    std::mem::drop(vec2);
    // Convert back to a matrix of the appropriate shape.
    let mat = Array::from_shape_vec((3, 3), vec1).unwrap();
    println!(
        "Memory-efficient concatenation of arrays 1 and 2 (rows):\n{:?}\n",
        mat
    );
    // Unlike as with `concatenate()`, which makes copies, `mat1` and `mat2` no
    // longer exist now because they were (efficiently) consumed/moved by the
    // above procedure.

    // In Matlab we can also "stack" vectors to make a 2d matrix, or 2d matrices
    // to make a 3d matrix.
    //
    // ```
    // mat = [1,2,3]
    // mat = [mat; mat] % stack 2 row vectors into 2d matrix
    // mat(:,:,2) = mat % stack two 2d matrices into 3d matrix
    // ```
    let mat = array![1, 2, 3]; // 1d array
    println!("1d array:\n{:?}", mat);
    // We can stack using the ndarray::stack() function, similar to concatenate.
    let mat = stack(Axis(0), &[mat.view(), mat.view()]).unwrap();
    println!("Stacked to make a 2d array:\n{:?}", mat);
    // Or to save a little bit of typing, with the macro ndarray::stack!().
    let mat = stack!(Axis(0), mat, mat);
    println!("Stacked to make a 3d array:\n{:?}", mat);
}
