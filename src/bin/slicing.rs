//! Slicing an array without reallocating.
//!
//! You can't do this explicitly in in Matlab, but sometimes it happens behind
//! the scenes.
//!
//! Slices and views make a shallow copy of the underlying array with a
//! different starting offset, stride, etc.  The shallow copy points back to
//! data in the original array so as to avoid unnecessary memory use.
//!
//! You can convert this shallow copy to a deep copy using `to_owned()`.
//! This is, confusingly, very different from `into_owned()`, which will take
//! ownership of a shallow copy but not necessarily reallocate into a contiguous
//! memory layout.

extern crate blas_src;
use ndarray::{array, Axis, s};

fn main() {
    // Generate a matrix for us to slice in different ways.
    //
    // ```matlab
    // mat = [1,11,111,1111;2,22,222,2222;3,33,333,3333;4,44,444,4444]
    // ```
    let mat = array![[1., 11., 111., 1111.],
                     [2., 22., 222., 2222.],
                     [3., 33., 333., 3333.],
                     [4., 44., 444., 4444.]];
    println!("Full 4x4 matrix:\n{:?}\n", mat);

    // Slice a view of the upper left 2x3 matrix.
    //
    // ```matlab
    // view = mat(1:2, 1:3)
    // ```
    //
    // Look at the output of println!() and you will see that while the shape of
    // the view is 2x3, the stride is 4x1 just like the original array.  All the
    // data is in the same place, we are just "viewing" it differently!
    //
    // Recall that ndarray counts from zero whereas Matlab counts from 1.
    // Recall also that Rust ranges do *not* include the end of the range.
    // The ..2 is equivalent to 0..2, "Rows 0 up to but not including 2."
    // The ..3 is equivalent to 0..3, "Columns 0 up to but not including 3."
    let view = mat.slice(s![..2, ..3]);
    println!("Upper left 2x3 view:\n{:?}", view);
    // If it's easier for you to think in inclusive ranges, you can use this
    // alternate syntax.  The ..1 means, "Rows 0 up to and including 1."
    let view = mat.slice(s![..=1, ..=2]);
    println!("Upper left 2x3 view:\n{:?}\n", view);

    // Generate a view of odd numbered columns.
    //
    // ```matlab
    // view = mat(:, 2:2:end)
    // ```
    //
    // Again note that ndarray starts counting at 0 and Matlab counts from 1.
    // So, in Matlab, this would actually be the even columns.
    // The first .. keeps all of the rows in each column.
    // The second 1..;2 says, "Start at column 1 and go through the last column;
    // take every 2nd column."
    let view = mat.slice(s![.., 1..;2]);
    println!("Odd columns\n{:?}\n", view);

    // Middle 2 rows and last 3 columns.
    //
    // ```matlab
    // view = mat(2:3, -3:end)
    // ```
    //
    // The 1..3 selects from the second row up to (but not including) row 3.
    // The -3.. selects from the third column from last up through the last.
    let view = mat.slice(s![1..3, -3..]);
    println!("Middle two rows, last 3 columns:\n{:?}\n", view);

    // Get the 2nd column.
    //
    // ```
    // view = mat(:, 2)
    // ```
    //
    // The syntax for indexing a single row or column is a little goofy.  If we
    // slice like this then the axis is *removed* completely, leaving us with a
    // 1-dimensional array.  See also ndarray::ArrayBase::index_axis().
    let view = mat.slice(s![.., 1]);
    println!("Second column as 1-dimensional array:\n{:?}", view);
    // Or equivalently:
    let view = mat.column(1);
    println!("Second column as 1-dimensional array using column():\n{:?}", view);
    // And more generally (for arrays with more than 2 axes):
    let view = mat.index_axis(Axis(1), 1);
    println!("Second column as 1-dimensional array using index_axis():\n{:?}", view);
    // If we want two simply *collapse* the axis to get a 2-dimensional column
    // vector then we must use the range syntax 1..2.  This starts at the second
    // column and goes up to but not including the third column -- in other
    // words, just the second column.  But by specifying it as a range we cue
    // ndarray not to remove the axis.
    let view = mat.slice(s![.., 1..2]);
    println!("Second column as 2-dimensional column vector:\n{:?}", view);
    // Or equivalently:
    let mut view = mat.view(); // generate a full view into the array
    view.collapse_axis(Axis(1), 1); // modify the view in-place
    println!("Second column as 2-dimensional column vector:\n{:?}\n", view);

    // We can convert the view into an owned array by cloning the data.
    // This deep copy also makes the layout contiguous in memory.
    // Note the change in stride and layout.
    println!("Is view in standard layout? {:?}", view.is_standard_layout());
    let submat = view.to_owned();
    println!("Clone of view into new array:\n{:?}", submat);
    println!("Is submat in standard layout? {:?}", submat.is_standard_layout());
}
