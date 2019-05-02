// Hyperbolic.swift
//
// Copyright (c) 2014–2015 Mattt Thompson (http://mattt.me)
//      Modifications 2016 Ludovic Arnold (http://ludovicarnold.com)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

import Accelerate


public enum MatrixAxies {
    case row
    case column
}

/**
 A Surge Matrix with elements of type T.
 
 Remark: (deferred copy) Methods documented with 'deferred copy' will only incur
 the cost of a copy if the concerned Array is mutated.  Array is a struct  (wich
 means  it  has copy semantics)  but a copy  is only  performed  if an  array is
 changed.  This behaviour  is indicated  with the words 'deferred copy'  in each
 method's documentation. If a  method returns an array with  'deferred copy', it
 can be safely modified, but it will only  incur  the cost of a copy if and when
 the concerned  array is mutated.  By contrast, methods  documented  with 'copy'
 will immediatly incur the cost of a copy.
 
 - parameter T: the type of the matrix elements (e.g. Float or Double).
 - note: the matrix is in row major (a.k.a. C) order.
*/
public struct Matrix<T> where T: FloatingPoint, T: ExpressibleByFloatLiteral {

    public let rows: Int
    public let columns: Int
    public let size: Int
    var grid: [T]
    
    /**
     Create a new matrix initialized with the given contents (deferred copy).
     - parameter rows: the number of rows
     - parameter columns: the number of columns
     - parameter grid: the matrix contents.
     */
    public init(rows: Int, columns: Int, grid: [T]) {
        self.rows = rows
        self.columns = columns
        self.size = rows * columns
        self.grid = grid
    }
    
    /**
     Create a new matrix filled with repeatedValue.
     - parameter rows: the number of rows.
     - parameter columns: the number of columns.
     - parameter repeatedValue: the initial value for all matrix elements.
    */
    public init(rows: Int, columns: Int, repeatedValue: T) {
        let grid = [T](repeating: repeatedValue, count: rows * columns)
        self.init(rows: rows, columns: columns, grid: grid)
    }
    
    /**
     Create a new matrix initialized with the given contents (copy).
     - parameter contents: the matrix contents.
    */
    public init(_ contents: [[T]]) {
        let m: Int = contents.count
        let n: Int = contents[0].count
        let repeatedValue: T = 0.0

        self.init(rows: m, columns: n, repeatedValue: repeatedValue)

        for (i, row) in contents.enumerated() {
            grid.replaceSubrange(i * n..<i * n + Swift.min(m,row.count), with: row)
        }
    }
    
    /**
     - returns: This matrix as a 1d array (deferred copy).
    */
    public func ravel() -> [T] {
        return self.grid
    }
    
    /**
     Reshaped view of this Matrix (deferred copy).
     - parameter rows: new number of rows.
     - parameter columns: new number of columns.
     - returns: a Matrix(rows, columns) with the same contents as self.
    */
    public func reshape(_ rows: Int, _ columns: Int) -> Matrix<T> {
        precondition( size == rows * columns, "cannot reshape Matrix(\(self.rows),\(self.columns)) to shape(\(rows),\(columns))")
        return Matrix<T>(rows: rows, columns: columns, grid: self.grid)
    }
    
    
    /**
     Get or set a single element.
     - parameter row: the element's row
     - parameter column: the element's column
    */
    public subscript(row: Int, column: Int) -> T {
        get {
            assert(indexIsValidForRow(row, column: column))
            return grid[(row * columns) + column]
        }

        set {
            assert(indexIsValidForRow(row, column: column))
            grid[(row * columns) + column] = newValue
        }
    }
    
    /**
     Get (deferred copy) or set (copy) a row .
     - parameter row: the row index.
    */
    public subscript(row row: Int) -> [T] {
        get {
            assert(row < rows)
            let startIndex = row * columns
            let endIndex = row * columns + columns
            return Array(grid[startIndex..<endIndex])
        }
        
        set {
            assert(row < rows)
            assert(newValue.count == columns)
            let startIndex = row * columns
            let endIndex = row * columns + columns
            grid.replaceSubrange(startIndex..<endIndex, with: newValue)
        }
    }
    
    /**
     Get (copy) or set (copy) a column.
     - parameter column: the column index.
    */
    public subscript(column column: Int) -> [T] {
        get {
            assert(column < columns)
            return (stride(from: column, to: size, by: columns)).map({ self.grid[$0] })
        }
        
        set {
            assert(column < columns)
            assert(newValue.count == rows)
            newValue.enumerated().forEach({ self.grid[$0 * columns + column] = $1 })
        }
    }
    
    /**
     Get the matrix as a 2D Array (deferred copy).
    */
    public func asArray() -> [[T]] {
        var result = [[T]]()
        for i in 0..<rows {
            result.append(self[row: i])
        }
        return result
    }
    
    /**
     Check if (row,column) is not out of bounds for this matrix.
     - parameter row: the row indice
     - parameter column: the column indice
     - returns: true if (row, column) are valid indices, false otherwise
    */
    fileprivate func indexIsValidForRow(_ row: Int, column: Int) -> Bool {
        return row >= 0 && row < rows && column >= 0 && column < columns
    }
    
}

// MARK: - Printable

extension Matrix: CustomStringConvertible {
    public var description: String {
        var description = ""

        for i in 0..<rows {
            let contents = (0..<columns).map{"\(self[i, $0])"}.joined(separator: "\t")

            switch (i, rows) {
            case (0, 1):
                description += "(\t\(contents)\t)"
            case (0, _):
                description += "⎛\t\(contents)\t⎞"
            case (rows - 1, _):
                description += "⎝\t\(contents)\t⎠"
            default:
                description += "⎜\t\(contents)\t⎥"
            }

            description += "\n"
        }

        return description
    }
}

// MARK: - SequenceType

extension Matrix: Sequence {
    public func makeIterator() -> AnyIterator<ArraySlice<T>> {
        let endIndex = rows * columns
        var nextRowStartIndex = 0

        return AnyIterator {
            if nextRowStartIndex == endIndex {
                return nil
            }

            let currentRowStartIndex = nextRowStartIndex
            nextRowStartIndex += self.columns

            return self.grid[currentRowStartIndex..<nextRowStartIndex]
        }
    }
}

extension Matrix: Equatable {}
public func ==<T> (lhs: Matrix<T>, rhs: Matrix<T>) -> Bool {
    return lhs.rows == rhs.rows && lhs.columns == rhs.columns && lhs.grid == rhs.grid
}


// MARK: -

/**
 Matrix-Matrix addition.
 - parameter x: a Matrix
 - parameter y: a Matrix
 - returns: x + y  (element-wise, newly allocated)
 */
public func add(_ x: Matrix<Float>, _ y: Matrix<Float>) -> Matrix<Float> {
    precondition(x.rows == y.rows && x.columns == y.columns, "Matrix(\(x.rows),\(x.columns)) and Matrix(\(y.rows),\(y.columns)) not compatible with element-wise addition")

    var results = y
    cblas_saxpy(Int32(x.grid.count), 1.0, x.grid, 1, &(results.grid), 1)

    return results
}

/**
 Matrix-Matrix addition.
 - parameter x: a Matrix
 - parameter y: a Matrix
 - returns: x + y  (element-wise, newly allocated)
 */
public func add(_ x: Matrix<Double>, _ y: Matrix<Double>) -> Matrix<Double> {
    precondition(x.rows == y.rows && x.columns == y.columns, "Matrix(\(x.rows),\(x.columns)) and Matrix(\(y.rows),\(y.columns)) not compatible with element-wise addition")

    var results = y
    cblas_daxpy(Int32(x.grid.count), 1.0, x.grid, 1, &(results.grid), 1)

    return results
}

/**
 Matrix-Matrix subtraction.
 - parameter x: a Matrix
 - parameter y: a Matrix
 - returns: x - y  (element-wise, newly allocated)
 */
public func sub(_ x: Matrix<Float>, _ y: Matrix<Float>) -> Matrix<Float> {
    precondition(x.rows == y.rows && x.columns == y.columns, "Matrix(\(x.rows),\(x.columns)) and Matrix(\(y.rows),\(y.columns)) not compatible with element-wise subtraction")
    
    var results = x
    cblas_saxpy(Int32(y.grid.count), -1.0, y.grid, 1, &(results.grid), 1)
    
    return results
}

/**
 Matrix-Matrix subtraction.
 - parameter x: a Matrix
 - parameter y: a Matrix
 - returns: x - y  (element-wise, newly allocated)
 */
public func sub(_ x: Matrix<Double>, _ y: Matrix<Double>) -> Matrix<Double> {
    precondition(x.rows == y.rows && x.columns == y.columns, "Matrix(\(x.rows),\(x.columns)) and Matrix(\(y.rows),\(y.columns)) not compatible with element-wise subtraction")
    
    var results = x
    cblas_daxpy(Int32(y.grid.count), -1.0, y.grid, 1, &(results.grid), 1)
    
    return results
}

/**
 Matrix scalar multiplication.
 - parameter alpha: a Float
 - parameter x: a Matrix
 - returns: x * alpha  (element-wise, newly allocated)
 */
public func mul(_ alpha: Float, _ x: Matrix<Float>) -> Matrix<Float> {
    var results = x
    cblas_sscal(Int32(x.grid.count), alpha, &(results.grid), 1)

    return results
}

/**
 Matrix scalar multiplication.
 - parameter alpha: a Double
 - parameter x: a Matrix
 - returns: x * alpha  (element-wise, newly allocated)
 */
public func mul(_ alpha: Double, _ x: Matrix<Double>) -> Matrix<Double> {
    var results = x
    cblas_dscal(Int32(x.grid.count), alpha, &(results.grid), 1)

    return results
}

/**
 Matrix-Matrix multiplication (NOT element-wise).
 - parameter x: a Matrix
 - parameter y: a Matrix
 - returns: x * y  (NOT element-wise, newly allocated)
 */
public func dot(_ x: Matrix<Float>, _ y: Matrix<Float>) -> Matrix<Float> {
    precondition(x.columns == y.rows, "Matrix(\(x.rows),\(x.columns)) and Matrix(\(y.rows),\(y.columns))  not compatible with multiplication")

    var results = Matrix<Float>(rows: x.rows, columns: y.columns, repeatedValue: 0.0)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(x.rows), Int32(y.columns), Int32(x.columns), 1.0, x.grid, Int32(x.columns), y.grid, Int32(y.columns), 0.0, &(results.grid), Int32(results.columns))

    return results
}

/**
 Matrix-Matrix multiplication (NOT element-wise).
 - parameter x: a Matrix
 - parameter y: a Matrix
 - returns: x * y  (NOT element-wise, newly allocated)
 */
public func dot(_ x: Matrix<Double>, _ y: Matrix<Double>) -> Matrix<Double> {
    precondition(x.columns == y.rows, "Matrix(\(x.rows),\(x.columns)) and Matrix(\(y.rows),\(y.columns)) not compatible with multiplication")

    var results = Matrix<Double>(rows: x.rows, columns: y.columns, repeatedValue: 0.0)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(x.rows), Int32(y.columns), Int32(x.columns), 1.0, x.grid, Int32(x.columns), y.grid, Int32(y.columns), 0.0, &(results.grid), Int32(results.columns))

    return results
}

/**
 Matrix-Vector multiplication (NOT element-wise).
 - parameter x: a Matrix
 - parameter y: an Array
 - returns: x * y  (NOT element-wise, newly allocated)
 */
public func dot(_ x: Matrix<Float>, _ y: [Float]) -> [Float] {
    precondition(x.columns == y.count, "Matrix(\(x.rows),\(x.columns)) and Vector(\(y.count)) not compatible with multiplication")
    
    var results = [Float](repeating: 0.0, count: x.rows)
    cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(x.rows), Int32(x.columns), 1.0, x.grid, Int32(x.columns), y, Int32(1), 0.0, &results, Int32(1))
    
    return results
}

/**
 Vector-Matrix multiplication (NOT element-wise).
 - parameter x: an Array
 - parameter y: a Matrix
 - returns: x * y  (NOT element-wise, newly allocated)
 */
public func dot(_ x: [Float], _ y: Matrix<Float>) -> [Float] {
    precondition(y.rows == x.count, "Vector(\(x.count)) and Matrix(\(y.rows),\(y.columns)) not compatible with multiplication")
    
    var results = [Float](repeating: 0.0, count: y.columns)
    cblas_sgemv(CblasRowMajor, CblasTrans, Int32(y.rows), Int32(y.columns), 1.0, y.grid, Int32(y.columns), x, Int32(1), 0.0, &results, Int32(1))
    
    return results
}

/**
 Matrix-Vector multiplication (NOT element-wise).
 - parameter x: a Matrix
 - parameter y: an Array
 - returns: x * y  (NOT element-wise, newly allocated)
 */
public func dot(_ x: Matrix<Double>, _ y: [Double]) -> [Double] {
    precondition(x.columns == y.count, "Matrix(\(x.rows),\(x.columns)) and Vector(\(y.count)) not compatible with multiplication")
    
    var results = [Double](repeating: 0.0, count: x.rows)
    cblas_dgemv(CblasRowMajor, CblasNoTrans, Int32(x.rows), Int32(x.columns), // order, trans, m, n
        1.0, x.grid, Int32(x.columns), // alpha, A, lda
        y, Int32(1), // X, incX
        0.0, &results, Int32(1)) // beta, Y, incY
    
    return results
}

/**
 Vector-Matrix multiplication (NOT element-wise).
 - parameter x: a Matrix
 - parameter y: an Array
 - returns: x * y  (NOT element-wise, newly allocated)
 */
public func dot(_ x: [Double], _ y: Matrix<Double>) -> [Double] {
    precondition(y.rows == x.count, "Vector(\(x.count)) and Matrix(\(y.rows),\(y.columns)) not compatible with multiplication")
    
    var results = [Double](repeating: 0.0, count: y.columns)
    cblas_dgemv(CblasRowMajor, CblasTrans, Int32(y.rows), Int32(y.columns), 1.0, y.grid, Int32(y.columns), x, Int32(1), 0.0, &results, Int32(1))
    
    return results
}

/**
 Matrix-Matrix multiplication (element-wise).
 - parameter x: a Matrix
 - parameter y: a Matrix
 - returns: x * y  (element-wise, newly allocated)
 */
public func mul(_ x: Matrix<Double>, _ y: Matrix<Double>) -> Matrix<Double> {
    precondition(x.rows == y.rows && x.columns == y.columns, "Matrix(\(x.rows),\(x.columns)) and Matrix(\(y.rows),\(y.columns)) not compatible with element-wise multiplication")
    var result = Matrix<Double>(rows: x.rows, columns: x.columns, repeatedValue: 0.0)
    result.grid = x.grid * y.grid
    return result
}

/**
 Matrix-Matrix multiplication (element-wise).
 - parameter x: a Matrix
 - parameter y: a Matrix
 - returns: x * y  (element-wise, newly allocated)
 */
public func mul(_ x: Matrix<Float>, _ y: Matrix<Float>) -> Matrix<Float> {
    precondition(x.rows == y.rows && x.columns == y.columns, "Matrix(\(x.rows),\(x.columns)) and Matrix(\(y.rows),\(y.columns)) not compatible with element-wise multiplication")
    var result = Matrix<Float>(rows: x.rows, columns: x.columns, repeatedValue: 0.0)
    result.grid = x.grid * y.grid
    return result
}

/**
 Matrix-Matrix division (NOT element-wise).
 - parameter x: a Matrix
 - parameter y: a Matrix
 - returns: x * 1/y  (NOT element-wise, newly allocated)
 */
public func div(_ x: Matrix<Double>, _ y: Matrix<Double>) -> Matrix<Double> {
    let yInv = inv(y)
    precondition(x.columns == yInv.rows, "Matrix(\(x.rows),\(x.columns)) and Matrix(\(y.rows),\(y.columns)) not compatible with division")
    return dot(x, yInv)
}

/**
 Matrix-Matrix division (NOT element-wise).
 - parameter x: a Matrix
 - parameter y: a Matrix
 - returns: x * 1/y  (NOT element-wise, newly allocated)
 */
public func div(_ x: Matrix<Float>, _ y: Matrix<Float>) -> Matrix<Float> {
    let yInv = inv(y)
    precondition(x.columns == yInv.rows, "Matrix(\(x.rows),\(x.columns)) and Matrix(\(y.rows),\(y.columns)) not compatible with division")
    return dot(x, yInv)
}

/**
 Matrix pow (element-wise).
 - parameter x: a Matrix
 - parameter y: a Double
 - returns: x^y  (element-wise, newly allocated)
 */
public func pow(_ x: Matrix<Double>, _ y: Double) -> Matrix<Double> {
    var result = Matrix<Double>(rows: x.rows, columns: x.columns, repeatedValue: 0.0)
    result.grid = pow(x.grid, y)
    return result
}

/**
 Matrix pow (element-wise).
 - parameter x: a Matrix
 - parameter y: a Double
 - returns: x^y  (element-wise, newly allocated)
 */
public func pow(_ x: Matrix<Float>, _ y: Float) -> Matrix<Float> {
    var result = Matrix<Float>(rows: x.rows, columns: x.columns, repeatedValue: 0.0)
    result.grid = pow(x.grid, y)
    return result
}

/**
 Matrix exponential (element-wise).
 - parameter x: a Matrix
 - returns: exp(x)  (element-wise, newly allocated)
 */
public func exp(_ x: Matrix<Double>) -> Matrix<Double> {
    var result = Matrix<Double>(rows: x.rows, columns: x.columns, repeatedValue: 0.0)
    result.grid = exp(x.grid)
    return result
}

/**
 Matrix exponential (element-wise).
 - parameter x: a Matrix
 - returns: exp(x)  (element-wise, newly allocated)
 */
public func exp(_ x: Matrix<Float>) -> Matrix<Float> {
    var result = Matrix<Float>(rows: x.rows, columns: x.columns, repeatedValue: 0.0)
    result.grid = exp(x.grid)
    return result
}

/**
 Matrix summation.
 - parameter x: a Matrix
 - returns: ∑(i=0..rows-1)∑(j=0..columns-1) x[i, j]
 */
public func sum(_ x: Matrix<Double>) -> Double {
    return sum(x.grid)
}

/**
 Matrix summation.
 - parameter x: a Matrix
 - parameter axies: .Column sums columns, .Row sums rows.
 - returns: the sum of columns, resp. rows
 */
public func sum(_ x: Matrix<Double>, axies: MatrixAxies) -> [Double] {
    switch axies {
    case .column:
        return (0..<x.columns).map({ sum(x[column: $0]) })
    case .row:
        return (0..<x.rows).map({ sum(x[row: $0]) })
    }
}

/**
 Matrix summation.
 - parameter x: a Matrix
 - returns: ∑(i=0..rows-1)∑(j=0..columns-1) x[i, j]
 */
public func sum(_ x: Matrix<Float>) -> Float {
    return sum(x.grid)
}

/**
 Matrix summation.
 - parameter x: a Matrix
 - parameter axies: .Column sums columns, .Row sums rows.
 - returns: the sum of columns, resp. rows
 */
public func sum(_ x: Matrix<Float>, axies: MatrixAxies) -> [Float] {
    switch axies {
    case .column:
        return (0..<x.columns).map({ sum(x[column: $0]) })
    case .row:
        return (0..<x.rows).map({ sum(x[row: $0]) })
    }
}

/**
 Matrix absolute summation.
 - parameter x: a Matrix
 - returns: ∑(i=0..rows-1)∑(j=0..columns-1) |x[i, j]|
 */
public func asum(_ x: Matrix<Double>) -> Double {
    return asum(x.grid)
}

/**
 Matrix absolute summation.
 - parameter x: a Matrix
 - parameter axies: .Column sums columns, .Row sums rows.
 - returns: the absolute sum of columns, resp. rows
 */
public func asum(_ x: Matrix<Double>, axies: MatrixAxies) -> [Double] {
    switch axies {
    case .column:
        return (0..<x.columns).map({ asum(x[column: $0]) })
    case .row:
        return (0..<x.rows).map({ asum(x[row: $0]) })
    }
}

/**
 Matrix absolute summation.
 - parameter x: a Matrix
 - returns: ∑(i=0..rows-1)∑(j=0..columns-1) |x[i, j]|
 */
public func asum(_ x: Matrix<Float>) -> Float {
    return asum(x.grid)
}

/**
 Matrix absolute summation.
 - parameter x: a Matrix
 - parameter axies: .Column sums columns, .Row sums rows.
 - returns: the absolute sum of columns, resp. rows
 */
public func asum(_ x: Matrix<Float>, axies: MatrixAxies) -> [Float] {
    switch axies {
    case .column:
        return (0..<x.columns).map({ asum(x[column: $0]) })
    case .row:
        return (0..<x.rows).map({ asum(x[row: $0]) })
    }
}

/**
 Matrix inversion (NOT element-wise).
 - parameter x: a Matrix
 - returns: 1/x (NOT element-wise, newly allocated)
 - remark: implementation asserts matrix is non-singular.
 */
public func inv(_ x : Matrix<Float>) -> Matrix<Float> {
    precondition(x.rows == x.columns, "Matrix(\(x.rows),\(x.columns)) is not square")

    var results = x

    var ipiv = [__CLPK_integer](repeating: 0, count: x.rows * x.rows)
    var lwork = __CLPK_integer(x.columns * x.columns)
    var work = [CFloat](repeating: 0.0, count: Int(lwork))
    var error: __CLPK_integer = 0
    // Solution for Swift 5 migration from https://stackoverflow.com/questions/47114737/overlapping-accesses-pointer
    // This is OK because dgetrf_ doesn't really mutate the pointer
    let nc = __CLPK_integer(x.columns)
    var nc1 = nc, nc2 = nc, nc3 = nc
    sgetrf_(&nc1, &nc2, &(results.grid), &nc3, &ipiv, &error)
    sgetri_(&nc1, &(results.grid), &nc2, &ipiv, &work, &lwork, &error)

    assert(error == 0, "Matrix not invertible")

    return results
}

/**
 Matrix inversion (NOT element-wise).
 - parameter x: a Matrix
 - returns: 1/x (NOT element-wise, newly allocated)
 - remark: implementation asserts matrix is non-singular.
 */
public func inv(_ x : Matrix<Double>) -> Matrix<Double> {
    precondition(x.rows == x.columns, "Matrix(\(x.rows),\(x.columns)) is not square")

    var results = x

    var ipiv = [__CLPK_integer](repeating: 0, count: x.rows * x.rows)
    var lwork = __CLPK_integer(x.columns * x.columns)
    var work = [CDouble](repeating: 0.0, count: Int(lwork))
    var error: __CLPK_integer = 0
    // Solution for Swift 5 migration from https://stackoverflow.com/questions/47114737/overlapping-accesses-pointer
    // This is OK because dgetrf_ doesn't really mutate the pointer
    let nc = __CLPK_integer(x.columns)
    var nc1 = nc, nc2 = nc, nc3 = nc
    dgetrf_(&nc1, &nc2, &(results.grid), &nc3, &ipiv, &error)
    dgetri_(&nc1, &(results.grid), &nc2, &ipiv, &work, &lwork, &error)

    assert(error == 0, "Matrix not invertible")

    return results
}

/**
 Matrix transpose (copy).
 - parameter x: a Matrix
 - returns: x' (newly allocated)
 */
public func transpose(_ x: Matrix<Float>) -> Matrix<Float> {
    var results = Matrix<Float>(rows: x.columns, columns: x.rows, repeatedValue: 0.0)
    vDSP_mtrans(x.grid, 1, &(results.grid), 1, vDSP_Length(results.rows), vDSP_Length(results.columns))

    return results
}

/**
 Matrix transpose (copy).
 - parameter x: a Matrix
 - returns: x' (newly allocated)
 */
public func transpose(_ x: Matrix<Double>) -> Matrix<Double> {
    var results = Matrix<Double>(rows: x.columns, columns: x.rows, repeatedValue: 0.0)
    vDSP_mtransD(x.grid, 1, &(results.grid), 1, vDSP_Length(results.rows), vDSP_Length(results.columns))

    return results
}

// MARK: - Operators

/**
 Matrix-Matrix addition.
*/
public func + (lhs: Matrix<Float>, rhs: Matrix<Float>) -> Matrix<Float> {
    return add(lhs, rhs)
}

/**
 Matrix-Matrix addition.
 */
public func + (lhs: Matrix<Double>, rhs: Matrix<Double>) -> Matrix<Double> {
    return add(lhs, rhs)
}

/**
 Matrix-Matrix subtraction
 */
public func - (lhs: Matrix<Float>, rhs: Matrix<Float>) -> Matrix<Float> {
    return sub(lhs, rhs)
}

/**
 Matrix-Matrix subtraction
 */
public func - (lhs: Matrix<Double>, rhs: Matrix<Double>) -> Matrix<Double> {
    return sub(lhs, rhs)
}

/**
 Matrix-Matrix multiplication
 */
public func * (lhs: Matrix<Float>, rhs: Matrix<Float>) -> Matrix<Float> {
    return mul(lhs, rhs)
}

/**
 Matrix-Matrix multiplication
 */
public func * (lhs: Matrix<Double>, rhs: Matrix<Double>) -> Matrix<Double> {
    return mul(lhs, rhs)
}

/**
 Matrix left scalar multiplication
*/
public func * (lhs: Float, rhs: Matrix<Float>) -> Matrix<Float> {
    return mul(lhs, rhs)
}

/**
 Matrix right scalar multiplication
*/
public func * (lhs: Matrix<Float>, rhs: Float) -> Matrix<Float> {
    return mul(rhs, lhs)
}

/**
 Matrix left scalar multiplication
*/
public func * (lhs: Double, rhs: Matrix<Double>) -> Matrix<Double> {
    return mul(lhs, rhs)
}

/**
 Matrix right scalar multiplication
*/
public func * (lhs: Matrix<Double>, rhs: Double) -> Matrix<Double> {
    return mul(rhs, lhs)
}


/**
 Matrix-Matrix multiplication
*/
public func • (lhs: Matrix<Float>, rhs: Matrix<Float>) -> Matrix<Float> {
    return dot(lhs, rhs)
}

/**
 Matrix-Vector multiplication
 */
public func • (lhs: Matrix<Float>, rhs: [Float]) -> [Float] {
    return dot(lhs, rhs)
}

/**
 Vector-Matrix multiplication
 */
public func • (lhs: [Float], rhs: Matrix<Float>) -> [Float] {
    return dot(lhs, rhs)
}

/**
 Matrix-Matrix multiplication
*/
public func • (lhs: Matrix<Double>, rhs: Matrix<Double>) -> Matrix<Double> {
    return dot(lhs, rhs)
}

/**
 Matrix-Vector multiplication
 */
public func • (lhs: Matrix<Double>, rhs: [Double]) -> [Double] {
    return dot(lhs, rhs)
}

/**
 Vector-Matrix multiplication
 */
public func • (lhs: [Double], rhs: Matrix<Double>) -> [Double] {
    return dot(lhs, rhs)
}

infix operator ÷ : MultiplicationPrecedence
/**
 matrix-matrix division
 */
public func ÷ (lhs: Matrix<Double>, rhs: Matrix<Double>) -> Matrix<Double> {
    return div(lhs, rhs)
}

/**
 matrix-matrix division
 */
public func ÷ (lhs: Matrix<Float>, rhs: Matrix<Float>) -> Matrix<Float> {
    return div(lhs, rhs)
}

/**
 matrix right scalar division
 */
public func / (lhs: Matrix<Double>, rhs: Double) -> Matrix<Double> {
    var result = Matrix<Double>(rows: lhs.rows, columns: lhs.columns, repeatedValue: 0.0)
    result.grid = lhs.grid / rhs;
    return result;
}

/**
 matrix right scalar division
 */
public func / (lhs: Matrix<Float>, rhs: Float) -> Matrix<Float> {
    var result = Matrix<Float>(rows: lhs.rows, columns: lhs.columns, repeatedValue: 0.0)
    result.grid = lhs.grid / rhs;
    return result;
}

postfix operator ′
/**
 Matrix transposition
*/
public postfix func ′ (value: Matrix<Float>) -> Matrix<Float> {
    return transpose(value)
}

/**
 Matrix transposition
*/
public postfix func ′ (value: Matrix<Double>) -> Matrix<Double> {
    return transpose(value)
}
