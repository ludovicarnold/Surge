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
    case Row
    case Column
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
public struct Matrix<T where T: FloatingPointType, T: FloatLiteralConvertible> {
    public typealias Element = T

    public let rows: Int
    public let columns: Int
    public let size: Int
    var grid: [Element]
    
    /**
     Create a new matrix initialized with the given contents (deferred copy).
     - parameter rows: the number of rows
     - parameter columns: the number of columns
     - parameter grid: the matrix contents.
     */
    public init(rows: Int, columns: Int, grid: [Element]) {
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
    public init(rows: Int, columns: Int, repeatedValue: Element) {
        let grid = [Element](count: rows * columns, repeatedValue: repeatedValue)
        self.init(rows: rows, columns: columns, grid: grid)
    }
    
    /**
     Create a new matrix initialized with the given contents (copy).
     - parameter contents: the matrix contents.
    */
    public init(_ contents: [[Element]]) {
        let m: Int = contents.count
        let n: Int = contents[0].count
        let repeatedValue: Element = 0.0

        self.init(rows: m, columns: n, repeatedValue: repeatedValue)

        for (i, row) in contents.enumerate() {
            grid.replaceRange(i*n..<i*n+min(m, row.count), with: row)
        }
    }
    
    /**
     - returns: This matrix as a 1d array (deferred copy).
    */
    public func ravel() -> [Element] {
        return self.grid
    }
    
    /**
     Reshaped view of this Matrix (deferred copy).
     - parameter rows: new number of rows.
     - parameter columns: new number of columns.
     - returns: a Matrix(rows, columns) with the same contents as self.
    */
    public func reshape(rows: Int, _ columns: Int) -> Matrix<Element> {
        precondition( size == rows * columns, "cannot reshape Matrix(\(self.rows),\(self.columns)) to shape(\(rows),\(columns))")
        return Matrix<T>(rows: rows, columns: columns, grid: self.grid)
    }
    
    
    /**
     Get or set a single element.
     - parameter row: the element's row
     - parameter column: the element's column
    */
    public subscript(row: Int, column: Int) -> Element {
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
    public subscript(row: Int) -> [Element] {
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
            grid.replaceRange(startIndex..<endIndex, with: newValue)
        }
    }
    
    /**
     Get (copy) or set (copy) a column.
     - parameter column: the column index.
    */
    public subscript(column column: Int) -> [Element] {
        get {
            assert(column < columns)
            return (column.stride(to: size, by: columns)).map({ self.grid[$0] })
        }
        
        set {
            assert(column < columns)
            assert(newValue.count == rows)
            newValue.enumerate().forEach({ self.grid[$0 * columns + column] = $1 })
        }
    }
    
    /**
     Get the matrix as a 2D Array (deferred copy).
    */
    public func asArray() -> [[Element]] {
        var result = [[Element]]()
        for i in 0..<rows {
            result.append(self[i])
        }
        return result
    }
    
    /**
     Check if (row,column) is not out of bounds for this matrix.
     - parameter row: the row indice
     - parameter column: the column indice
     - returns: true if (row, column) are valid indices, false otherwise
    */
    private func indexIsValidForRow(row: Int, column: Int) -> Bool {
        return row >= 0 && row < rows && column >= 0 && column < columns
    }
    
}

// MARK: - Printable

extension Matrix: CustomStringConvertible {
    public var description: String {
        var description = ""

        for i in 0..<rows {
            let contents = (0..<columns).map{"\(self[i, $0])"}.joinWithSeparator("\t")

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

extension Matrix: SequenceType {
    public func generate() -> AnyGenerator<ArraySlice<Element>> {
        let endIndex = rows * columns
        var nextRowStartIndex = 0

        return AnyGenerator {
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
 Matrix addition.
 - parameter x: a Matrix
 - parameter y: a Matrix
 - returns: x + y  (element-wise, newly allocated)
 */
public func add(x: Matrix<Float>, y: Matrix<Float>) -> Matrix<Float> {
    precondition(x.rows == y.rows && x.columns == y.columns, "Matrix dimensions not compatible with addition")

    var results = y
    cblas_saxpy(Int32(x.grid.count), 1.0, x.grid, 1, &(results.grid), 1)

    return results
}

/**
 Matrix addition.
 - parameter x: a Matrix
 - parameter y: a Matrix
 - returns: x + y  (element-wise, newly allocated)
 */
public func add(x: Matrix<Double>, y: Matrix<Double>) -> Matrix<Double> {
    precondition(x.rows == y.rows && x.columns == y.columns, "Matrix dimensions not compatible with addition")

    var results = y
    cblas_daxpy(Int32(x.grid.count), 1.0, x.grid, 1, &(results.grid), 1)

    return results
}

/**
 Matrix subtraction.
 - parameter x: a Matrix
 - parameter y: a Matrix
 - returns: x - y  (element-wise, newly allocated)
 */
public func sub(x: Matrix<Float>, y: Matrix<Float>) -> Matrix<Float> {
    precondition(x.rows == y.rows && x.columns == y.columns, "Matrix dimensions not compatible with subtraction")
    
    var results = x
    cblas_saxpy(Int32(y.grid.count), -1.0, y.grid, 1, &(results.grid), 1)
    
    return results
}

/**
 Matrix subtraction.
 - parameter x: a Matrix
 - parameter y: a Matrix
 - returns: x - y  (element-wise, newly allocated)
 */
public func sub(x: Matrix<Double>, y: Matrix<Double>) -> Matrix<Double> {
    precondition(x.rows == y.rows && x.columns == y.columns, "Matrix dimensions not compatible with subtraction")
    
    var results = x
    cblas_daxpy(Int32(y.grid.count), -1.0, y.grid, 1, &(results.grid), 1)
    
    return results
}

/**
 Matrix multiplication by a scalar.
 - parameter alpha: a Float
 - parameter x: a Matrix
 - returns: x * alpha  (element-wise, newly allocated)
 */
public func mul(alpha: Float, x: Matrix<Float>) -> Matrix<Float> {
    var results = x
    cblas_sscal(Int32(x.grid.count), alpha, &(results.grid), 1)

    return results
}

/**
 Matrix multiplication by a scalar.
 - parameter alpha: a Double
 - parameter x: a Matrix
 - returns: x * alpha  (element-wise, newly allocated)
 */
public func mul(alpha: Double, x: Matrix<Double>) -> Matrix<Double> {
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
public func mul(x: Matrix<Float>, y: Matrix<Float>) -> Matrix<Float> {
    precondition(x.columns == y.rows, "Matrix dimensions not compatible with multiplication")

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
public func mul(x: Matrix<Double>, y: Matrix<Double>) -> Matrix<Double> {
    precondition(x.columns == y.rows, "Matrix dimensions not compatible with multiplication")

    var results = Matrix<Double>(rows: x.rows, columns: y.columns, repeatedValue: 0.0)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(x.rows), Int32(y.columns), Int32(x.columns), 1.0, x.grid, Int32(x.columns), y.grid, Int32(y.columns), 0.0, &(results.grid), Int32(results.columns))

    return results
}

/**
 Matrix-Matrix multiplication (element-wise).
 - parameter x: a Matrix
 - parameter y: a Matrix
 - returns: x * y  (element-wise, newly allocated)
 */
public func elmul(x: Matrix<Double>, y: Matrix<Double>) -> Matrix<Double> {
    precondition(x.rows == y.rows && x.columns == y.columns, "Matrix must have the same dimensions")
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
public func elmul(x: Matrix<Float>, y: Matrix<Float>) -> Matrix<Float> {
    precondition(x.rows == y.rows && x.columns == y.columns, "Matrix must have the same dimensions")
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
public func div(x: Matrix<Double>, y: Matrix<Double>) -> Matrix<Double> {
    let yInv = inv(y)
    precondition(x.columns == yInv.rows, "Matrix dimensions not compatible")
    return mul(x, y: yInv)
}

/**
 Matrix-Matrix division (NOT element-wise).
 - parameter x: a Matrix
 - parameter y: a Matrix
 - returns: x * 1/y  (NOT element-wise, newly allocated)
 */
public func div(x: Matrix<Float>, y: Matrix<Float>) -> Matrix<Float> {
    let yInv = inv(y)
    precondition(x.columns == yInv.rows, "Matrix dimensions not compatible")
    return mul(x, y: yInv)
}

/**
 Matrix pow (element-wise).
 - parameter x: a Matrix
 - parameter y: a Double
 - returns: x^y  (element-wise, newly allocated)
 */
public func pow(x: Matrix<Double>, _ y: Double) -> Matrix<Double> {
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
public func pow(x: Matrix<Float>, _ y: Float) -> Matrix<Float> {
    var result = Matrix<Float>(rows: x.rows, columns: x.columns, repeatedValue: 0.0)
    result.grid = pow(x.grid, y)
    return result
}

/**
 Matrix exponential (element-wise).
 - parameter x: a Matrix
 - returns: exp(x)  (element-wise, newly allocated)
 */
public func exp(x: Matrix<Double>) -> Matrix<Double> {
    var result = Matrix<Double>(rows: x.rows, columns: x.columns, repeatedValue: 0.0)
    result.grid = exp(x.grid)
    return result
}

/**
 Matrix exponential (element-wise).
 - parameter x: a Matrix
 - returns: exp(x)  (element-wise, newly allocated)
 */
public func exp(x: Matrix<Float>) -> Matrix<Float> {
    var result = Matrix<Float>(rows: x.rows, columns: x.columns, repeatedValue: 0.0)
    result.grid = exp(x.grid)
    return result
}

/**
 Matrix summation.
 - parameter x: a Matrix
 - parameter axies: .Column sums along columns, .Row sums along rows.
 - returns: the sum of columns, resp. rows as a column matrix,
   resp. a row matrix (element-wise, newly allocated)
 */
public func sum(x: Matrix<Double>, axies: MatrixAxies = .Column) -> Matrix<Double> {
    
    switch axies {
    case .Column:
        var result = Matrix<Double>(rows: 1, columns: x.columns, repeatedValue: 0.0)
        for i in 0..<x.columns {
            result.grid[i] = sum(x[column: i])
        }
        return result
        
    case .Row:
        var result = Matrix<Double>(rows: x.rows, columns: 1, repeatedValue: 0.0)
        for i in 0..<x.rows {
            result.grid[i] = sum(x[i])
        }
        return result
    }
}

/**
 Matrix summation.
 - parameter x: a Matrix
 - parameter axies: .Column sums along columns, .Row sums along rows.
 - returns: the sum of columns, resp. rows as a column matrix,
 resp. a row matrix (element-wise, newly allocated)
 */
public func sum(x: Matrix<Float>, axies: MatrixAxies = .Column) -> Matrix<Float> {
    
    switch axies {
    case .Column:
        var result = Matrix<Float>(rows: 1, columns: x.columns, repeatedValue: 0.0)
        for i in 0..<x.columns {
            result.grid[i] = sum(x[column: i])
        }
        return result
        
    case .Row:
        var result = Matrix<Float>(rows: x.rows, columns: 1, repeatedValue: 0.0)
        for i in 0..<x.rows {
            result.grid[i] = sum(x[i])
        }
        return result
    }
}

/**
 Matrix absolute summation.
 - parameter x: a Matrix
 - parameter axies: .Column sums along columns, .Row sums along rows.
 - returns: the absolute sum of columns, resp. rows as a column matrix,
 resp. a row matrix (element-wise, newly allocated)
 */
public func asum(x: Matrix<Double>, axies: MatrixAxies = .Column) -> Matrix<Double> {
    
    switch axies {
    case .Column:
        var result = Matrix<Double>(rows: 1, columns: x.columns, repeatedValue: 0.0)
        for i in 0..<x.columns {
            result.grid[i] = asum(x[column: i])
        }
        return result
        
    case .Row:
        var result = Matrix<Double>(rows: x.rows, columns: 1, repeatedValue: 0.0)
        for i in 0..<x.rows {
            result.grid[i] = asum(x[i])
        }
        return result
    }
}

/**
 Matrix absolute summation.
 - parameter x: a Matrix
 - parameter axies: .Column sums along columns, .Row sums along rows.
 - returns: the absolute sum of columns, resp. rows as a column matrix,
 resp. a row matrix (element-wise, newly allocated)
 */
public func asum(x: Matrix<Float>, axies: MatrixAxies = .Column) -> Matrix<Float> {
    
    switch axies {
    case .Column:
        var result = Matrix<Float>(rows: 1, columns: x.columns, repeatedValue: 0.0)
        for i in 0..<x.columns {
            result.grid[i] = asum(x[column: i])
        }
        return result
        
    case .Row:
        var result = Matrix<Float>(rows: x.rows, columns: 1, repeatedValue: 0.0)
        for i in 0..<x.rows {
            result.grid[i] = asum(x[i])
        }
        return result
    }
}

/**
 Matrix inversion (NOT element-wise).
 - parameter x: a Matrix
 - returns: 1/x (NOT element-wise, newly allocated)
 - remark: implementation asserts matrix is non-singular.
 */
public func inv(x : Matrix<Float>) -> Matrix<Float> {
    precondition(x.rows == x.columns, "Matrix must be square")

    var results = x

    var ipiv = [__CLPK_integer](count: x.rows * x.rows, repeatedValue: 0)
    var lwork = __CLPK_integer(x.columns * x.columns)
    var work = [CFloat](count: Int(lwork), repeatedValue: 0.0)
    var error: __CLPK_integer = 0
    var nc = __CLPK_integer(x.columns)

    sgetrf_(&nc, &nc, &(results.grid), &nc, &ipiv, &error)
    sgetri_(&nc, &(results.grid), &nc, &ipiv, &work, &lwork, &error)

    assert(error == 0, "Matrix not invertible")

    return results
}

/**
 Matrix inversion (NOT element-wise).
 - parameter x: a Matrix
 - returns: 1/x (NOT element-wise, newly allocated)
 - remark: implementation asserts matrix is non-singular.
 */
public func inv(x : Matrix<Double>) -> Matrix<Double> {
    precondition(x.rows == x.columns, "Matrix must be square")

    var results = x

    var ipiv = [__CLPK_integer](count: x.rows * x.rows, repeatedValue: 0)
    var lwork = __CLPK_integer(x.columns * x.columns)
    var work = [CDouble](count: Int(lwork), repeatedValue: 0.0)
    var error: __CLPK_integer = 0
    var nc = __CLPK_integer(x.columns)

    dgetrf_(&nc, &nc, &(results.grid), &nc, &ipiv, &error)
    dgetri_(&nc, &(results.grid), &nc, &ipiv, &work, &lwork, &error)

    assert(error == 0, "Matrix not invertible")

    return results
}

/**
 Matrix transpose (copy).
 - parameter x: a Matrix
 - returns: x' (newly allocated)
 */
public func transpose(x: Matrix<Float>) -> Matrix<Float> {
    var results = Matrix<Float>(rows: x.columns, columns: x.rows, repeatedValue: 0.0)
    vDSP_mtrans(x.grid, 1, &(results.grid), 1, vDSP_Length(results.rows), vDSP_Length(results.columns))

    return results
}

/**
 Matrix transpose (copy).
 - parameter x: a Matrix
 - returns: x' (newly allocated)
 */
public func transpose(x: Matrix<Double>) -> Matrix<Double> {
    var results = Matrix<Double>(rows: x.columns, columns: x.rows, repeatedValue: 0.0)
    vDSP_mtransD(x.grid, 1, &(results.grid), 1, vDSP_Length(results.rows), vDSP_Length(results.columns))

    return results
}

// MARK: - Operators

public func + (lhs: Matrix<Float>, rhs: Matrix<Float>) -> Matrix<Float> {
    return add(lhs, y: rhs)
}

public func + (lhs: Matrix<Double>, rhs: Matrix<Double>) -> Matrix<Double> {
    return add(lhs, y: rhs)
}

public func - (lhs: Matrix<Float>, rhs: Matrix<Float>) -> Matrix<Float> {
    return sub(lhs, y: rhs)
}

public func - (lhs: Matrix<Double>, rhs: Matrix<Double>) -> Matrix<Double> {
    return sub(lhs, y: rhs)
}

public func * (lhs: Float, rhs: Matrix<Float>) -> Matrix<Float> {
    return mul(lhs, x: rhs)
}

public func * (lhs: Double, rhs: Matrix<Double>) -> Matrix<Double> {
    return mul(lhs, x: rhs)
}

public func * (lhs: Matrix<Float>, rhs: Matrix<Float>) -> Matrix<Float> {
    return mul(lhs, y: rhs)
}

public func * (lhs: Matrix<Double>, rhs: Matrix<Double>) -> Matrix<Double> {
    return mul(lhs, y: rhs)
}

public func / (lhs: Matrix<Double>, rhs: Matrix<Double>) -> Matrix<Double> {
    return div(lhs, y: rhs)
}

public func / (lhs: Matrix<Float>, rhs: Matrix<Float>) -> Matrix<Float> {
    return div(lhs, y: rhs)
}

public func / (lhs: Matrix<Double>, rhs: Double) -> Matrix<Double> {
    var result = Matrix<Double>(rows: lhs.rows, columns: lhs.columns, repeatedValue: 0.0)
    result.grid = lhs.grid / rhs;
    return result;
}

public func / (lhs: Matrix<Float>, rhs: Float) -> Matrix<Float> {
    var result = Matrix<Float>(rows: lhs.rows, columns: lhs.columns, repeatedValue: 0.0)
    result.grid = lhs.grid / rhs;
    return result;
}

postfix operator ′ {}
public postfix func ′ (value: Matrix<Float>) -> Matrix<Float> {
    return transpose(value)
}

public postfix func ′ (value: Matrix<Double>) -> Matrix<Double> {
    return transpose(value)
}
