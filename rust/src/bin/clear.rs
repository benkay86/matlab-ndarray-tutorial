//! Clearing memory.  In Rust this is referred to as "dropping" a variable.

extern crate blas_src;
use ndarray::{array, Array, Dim, Ix};
use std::mem::drop;

// Functions are covered in greater detail later.
// This function takes a reference to an array and computes a new array by
// adding 1 to each element of the first array.  The function-local variable
// `mat2` is dropped when the function returns.
fn add1(mat1: &Array<f64, Dim<[Ix; 1]>>) {
    let mat2 = mat1 + 1.;
    println!("mat2 inside add1():\n{:?}", mat2);
    // mat2 dropped here
    // mat1 still exists
}

// This function takes ownership of an array, i.e. it is moved into the
// function.  When the function returns the variable is cleared.
fn oppenheimer_destroyer_of_matrices(mat: Array<f64, Dim<[Ix; 1]>>) {
    println!("Behold, I will destroy this matrix:\n{:?}", mat);
    // mat dropped here
}

fn main() {
    // Create some array as in <matrix_creation.rs>.
    //
    // ```matlab
    // mat = [1,2,3]
    // ```
    let mat = array![1., 2., 3.];
    println!("Some array:\n{:?}\n", mat);

    // Clear the array, freeing its memory.
    // The `drop()` comes from `std::mem::drop()`, see `use` statements at top
    // of file.
    //
    // ```matlab
    // clear mat
    // ```
    drop(mat);
    // Uncomment the following line to get a compiler error.
    // println!("Can't print the array after it's dropped:\n{:?}", mat);

    // In Matlab, all local variables are cleared when a function returns.
    // Functions and math in ndarray will be covered in later examples.
    //
    // ```matlab
    // mat1 = [1,2,3]
    // add1(mat1);
    // mat1 % unchanged, still exists
    // % mat2 does not exist after add1() returns
    // function add1(mat1)
    //     mat2 = mat1 + 1;
    // end
    // ```
    let mat1 = array![1., 2., 3.];
    println!("mat1 before add1():\n{:?}", mat1);
    add1(&mat1);
    println!("mat1 after add1():\n{:?}\n", mat1);
    // Uncomment the following line to get a compiler error.
    // println!("mat2 doesn't exist after add1() returns:\n{:?}", mat2);

    // Matlab doesn't let you move variables into a function, but Rust does.
    // If a variable is moved into a function then it will no longer exist after
    // the function returns.  This is how `std::mem::drop()` works.
    oppenheimer_destroyer_of_matrices(mat1);
    print!("\n");
    // Uncomment the following line to get a compiler error.
    // println!("Oppenheimer destroyed mat1:\n{:?}", mat1);

    // Local variables are also destroyed at the end of a scope defined by a
    // code block (i.e. between { and }).  This is useful for cleaning up
    // temporary variables without explicitly writing a function.
    let mat1 = array![1., 2., 3.];
    let mat2 = {
        let mat3 = array![4., 6., 6.];
        &mat1 + &mat3
        // mat3 dropped at end of scope
    };
    println!("mat1:\n{:?}", mat1);
    println!("mat2:\n{:?}\n", mat2);
    // Uncomment the following line to get a compiler error.
    // println!("mat3 doesn't exist outside block:\n{:?}", mat3);

    // Beware of variable shadowing.  When you assign to a variable with Matlab
    // the original variable is overwritten or cleared.  But when you "assign"
    // to a variable using `let` in Rust, the original variable is merely
    // shadowed by the new variable name.  The original variable still exists in
    // memory until it is dropped at the end of the code block or function.
    let mat = array![1., 2., 3.];
    println!("mat:\n{:?}", mat);
    let original_mat = &mat; // store reference to original mat
    let mat = array![4., 5., 6.]; // shadow original with a new mat
    println!("new mat:\n{:?}", mat);
    println!(
        "Original mat still exists, but its name is shadowed:\n{:?}\n",
        original_mat
    );

    // Closures are a lot like Rust functions and have similar propreties when
    // it comes to dropping variables.  This has led to the ever-popular toilet
    // closure so-called because |_|{} looks kind of like a toilet.  It is
    // basically the same thing as `drop()`, but perhaps more cryptic.
    let mat1 = array![1., 2., 3.];
    (|_| {})(mat1);
    // Uncomment the following line to get a compiler error.
    // println!("mat1 was flushed down the toilet:\n{:?}", mat1);

    // However, a slightly different style of toilet closure can be useful to
    // avoid unintentional shadowing of variable names.  First let's avoid
    // shadowing using an explicit `drop()`.
    let mat = array![1., 2., 3.];
    println!("mat:\n{:?}", mat);
    let new_mat = &mat + 1.;
    drop(mat); // explicitly drop mat to avoid shadowing
    let mat = new_mat;
    println!("mat after adding 1:\n{:?}", mat);
    // And now with the toilet closure idiom.
    let mat = (|mat| &mat + 1.)(mat);
    println!("mat after adding 1 twice:\n{:?}", mat);
    // Even more concisely using the `move` keyword to capture `mat` from the
    // the current scope by explicitly moving it into the closure.
    let mat = (move || &mat + 1.)();
    println!("mat after adding 1 three times:\n{:?}", mat);
    // Note that ndarray performs addition in-place by default.  So we could
    // actually have done the following without shadowing because the `+`
    // operator will consume `mat` if we take it by value (e.g. leave off them
    // reference operator `&`).  But that wouldn't have made for a good examples
    // of blocks and toilet closures ;-)
    //
    // let original_mat = &mat; // compiler error, mat will not be shadowed
    let mat = mat + 1.;
    println!("mat after adding 1 four times:\n{:?}", mat);
}
