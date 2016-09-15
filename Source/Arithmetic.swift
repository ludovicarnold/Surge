// Arithmetic.swift
//
// Copyright (c) 2014–2015 Mattt Thompson (http://mattt.me)
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

// MARK: Sum

/**
 Sum elements of an Array.
 - parameter x: the array to sum over.
 - returns: ∑(i=0..count-1) x[i]
 */
public func sum(_ x: [Float]) -> Float {
    var result: Float = 0.0
    vDSP_sve(x, 1, &result, vDSP_Length(x.count))

    return result
}

/**
 Sum elements of an Array.
 - parameter x: the array to sum over.
 - returns: ∑(i=0..count-1) x[i]
 */
public func sum(_ x: [Double]) -> Double {
    var result: Double = 0.0
    vDSP_sveD(x, 1, &result, vDSP_Length(x.count))

    return result
}

// MARK: Sum of Absolute Values

/**
 Sum the absolute values of an Array.
 - parameter x: the array to sum over.
 - returns: ∑(i=0..count-1) |x[i]|
*/
public func asum(_ x: [Float]) -> Float {
    return cblas_sasum(Int32(x.count), x, 1)
}

/**
 Sum the absolute values of an Array.
 - parameter x: the array to sum over.
 - returns: ∑(i=0..count-1) |x[i]|
 */
public func asum(_ x: [Double]) -> Double {
    return cblas_dasum(Int32(x.count), x, 1)
}

// MARK: Maximum

/**
 The maximum value of an array.
 - parameter x: the input array.
 - returns: M | ∃i, x[i] = M, ∀i, x[i] ≤ M.
 */
public func max(_ x: [Float]) -> Float {
    var result: Float = 0.0
    vDSP_maxv(x, 1, &result, vDSP_Length(x.count))

    return result
}

/**
 The maximum value of an array.
 - parameter x: the input array.
 - returns: M | ∃i, x[i] = M, ∀i, x[i] ≤ M.
 */
public func max(_ x: [Double]) -> Double {
    var result: Double = 0.0
    vDSP_maxvD(x, 1, &result, vDSP_Length(x.count))

    return result
}

/**
 The index of the maximum value of an array.
 - parameter x: the input array.
 - returns: k | ∀i, x[i] ≤ x[k].
 */
public func argmax(_ x: [Float]) -> Int {
    var maxv: Float = 0.0
    var maxvi: UInt = 0
    vDSP_maxvi(x, 1, &maxv, &maxvi, vDSP_Length(x.count))
    
    return Int(maxvi)
}

/**
 The index of the maximum value of an array.
 - parameter x: the input array.
 - returns: k | ∀i, x[i] ≤ x[k].
 */
public func argmax(_ x: [Double]) -> Int {
    var maxv: Double = 0.0
    var maxvi: UInt = 0
    vDSP_maxviD(x, 1, &maxv, &maxvi, vDSP_Length(x.count))
    
    return Int(maxvi)
}

// MARK: Minimum

/**
 The minimum value of an array.
 - parameter x: the input array.
 - returns: m | ∃i, x[i] = m, ∀i, x[i] ≥ m.
 */
public func min(_ x: [Float]) -> Float {
    var result: Float = 0.0
    vDSP_minv(x, 1, &result, vDSP_Length(x.count))

    return result
}

/**
 The minimum value of an array.
 - parameter x: the input array.
 - returns: m | ∃i, x[i] = m, ∀i, x[i] ≥ m.
 */
public func min(_ x: [Double]) -> Double {
    var result: Double = 0.0
    vDSP_minvD(x, 1, &result, vDSP_Length(x.count))

    return result
}

/**
 The index of the minimum value of an array.
 - parameter x: the input array.
 - returns: k | ∀i, x[i] ≥ x[k].
 */
public func argmin(_ x: [Float]) -> Int {
    var minv: Float = 0.0
    var minvi: UInt = 0
    vDSP_minvi(x, 1, &minv, &minvi, vDSP_Length(x.count))
    
    return Int(minvi)
}

/**
 The index of the minimum value of an array.
 - parameter x: the input array.
 - returns: k | ∀i, x[i] ≥ x[k].
 */
public func argmin(_ x: [Double]) -> Int {
    var minv: Double = 0.0
    var minvi: UInt = 0
    vDSP_minviD(x, 1, &minv, &minvi, vDSP_Length(x.count))
    
    return Int(minvi)
}

// MARK: Mean

/**
 The mean value of an array.
 - parameter x: the input array.
 - returns: (1/count) ∑(i=0..count-1) x[i]
 */
public func mean(_ x: [Float]) -> Float {
    var result: Float = 0.0
    vDSP_meanv(x, 1, &result, vDSP_Length(x.count))

    return result
}

/**
 The mean value of an array.
 - parameter x: the input array.
 - returns: (1/count) ∑(i=0..count-1) x[i]
 */
public func mean(_ x: [Double]) -> Double {
    var result: Double = 0.0
    vDSP_meanvD(x, 1, &result, vDSP_Length(x.count))

    return result
}

// MARK: Mean Magnitude

/**
 The mean magnitude of an array.
 - parameter x: the input array.
 - returns: (1/count) ∑(i=0..count-1) |x[i]|
 */
public func meamg(_ x: [Float]) -> Float {
    var result: Float = 0.0
    vDSP_meamgv(x, 1, &result, vDSP_Length(x.count))

    return result
}

/**
 The mean magnitude of an array.
 - parameter x: the input array.
 - returns: (1/count) ∑(i=0..count-1) |x[i]|
 */
public func meamg(_ x: [Double]) -> Double {
    var result: Double = 0.0
    vDSP_meamgvD(x, 1, &result, vDSP_Length(x.count))

    return result
}

// MARK: Mean Square Value

/**
 The mean squared  value of an array.
 - parameter x: the input array.
 - returns: (1/count) ∑(i=0..count-1) x[i]²
 */
public func measq(_ x: [Float]) -> Float {
    var result: Float = 0.0
    vDSP_measqv(x, 1, &result, vDSP_Length(x.count))

    return result
}

/**
 The mean squared  value of an array.
 - parameter x: the input array.
 - returns: (1/count) ∑(i=0..count-1) x[i]²
 */
public func measq(_ x: [Double]) -> Double {
    var result: Double = 0.0
    vDSP_measqvD(x, 1, &result, vDSP_Length(x.count))

    return result
}

// MARK: Add

/**
 Element-wise array addition.
 - parameter x: an Array
 - parameter y: an Array
 - precondition: x.count == y.count
 - returns: newly allocated x + y
 */
public func add(_ x: [Float], _ y: [Float]) -> [Float] {
    precondition(x.count == y.count, "Vector(\(x.count)) and Vector(\(y.count)) not compatible with element-wise addition")
    var results = [Float](y)
    cblas_saxpy(Int32(x.count), 1.0, x, 1, &results, 1)

    return results
}

/**
 Element-wise array addition.
 - parameter x: an Array
 - parameter y: an Array
 - precondition: x.count == y.count
 - returns: newly allocated x + y
 */
public func add(_ x: [Double], _ y: [Double]) -> [Double] {
    precondition(x.count == y.count, "Vector(\(x.count)) and Vector(\(y.count)) not compatible with element-wise addition")
    var results = [Double](y)
    cblas_daxpy(Int32(x.count), 1.0, x, 1, &results, 1)

    return results
}

// MARK: Subtraction

/**
 Element-wise array subtraction.
 - parameter x: an Array
 - parameter y: an Array
 - precondition: x.count == y.count
 - returns: newly allocated x - y
 */
public func sub(_ x: [Float], _ y: [Float]) -> [Float] {
    precondition(x.count == y.count, "Vector(\(x.count)) and Vector(\(y.count)) not compatible with element-wise subtraction")
    var results = [Float](y)
    catlas_saxpby(Int32(x.count), 1.0, x, 1, -1, &results, 1)
    
    return results
}

/**
 Element-wise array subtraction.
 - parameter x: an Array
 - parameter y: an Array
 - precondition: x.count == y.count
 - returns: newly allocated x - y
 */
public func sub(_ x: [Double], _ y: [Double]) -> [Double] {
    precondition(x.count == y.count, "Vector(\(x.count)) and Vector(\(y.count)) not compatible with element-wise subtraction")
    var results = [Double](y)
    catlas_daxpby(Int32(x.count), 1.0, x, 1, -1, &results, 1)
    
    return results
}

// MARK: Multiply

/**
 Element-wise array multiplication.
 - parameter x: an Array
 - parameter y: an Array
 - precondition: x.count == y.count
 - returns: newly allocated x * y
 */
public func mul(_ x: [Float], _ y: [Float]) -> [Float] {
    precondition(x.count == y.count, "Vector(\(x.count)) and Vector(\(y.count)) not compatible with element-wise multiplication")
    var results = [Float](repeating: 0.0, count: x.count)
    vDSP_vmul(x, 1, y, 1, &results, 1, vDSP_Length(x.count))

    return results
}

/**
 Element-wise array multiplication.
 - parameter x: an Array
 - parameter y: an Array
 - precondition: x.count == y.count
 - returns: newly allocated x * y
 */
public func mul(_ x: [Double], _ y: [Double]) -> [Double] {
    precondition(x.count == y.count, "Vector(\(x.count)) and Vector(\(y.count)) not compatible with element-wise multiplication")
    var results = [Double](repeating: 0.0, count: x.count)
    vDSP_vmulD(x, 1, y, 1, &results, 1, vDSP_Length(x.count))

    return results
}

// MARK: Divide

/**
 Element-wise array division.
 - parameter x: an Array
 - parameter y: an Array
 - precondition: x.count == y.count
 - returns: newly allocated x / y
 */
public func div(_ x: [Float], _ y: [Float]) -> [Float] {
    precondition(x.count == y.count, "Vector(\(x.count)) and Vector(\(y.count)) not compatible with element-wise division")
    var results = [Float](repeating: 0.0, count: x.count)
    vvdivf(&results, x, y, [Int32(x.count)])

    return results
}

/**
 Element-wise array division.
 - parameter x: an Array
 - parameter y: an Array
 - precondition: x.count == y.count
 - returns: newly allocated x / y
 */
public func div(_ x: [Double], _ y: [Double]) -> [Double] {
    precondition(x.count == y.count, "Vector(\(x.count)) and Vector(\(y.count)) not compatible with element-wise division")
    var results = [Double](repeating: 0.0, count: x.count)
    vvdiv(&results, x, y, [Int32(x.count)])

    return results
}

public func mod(_ x: [Float], _ y: [Float]) -> [Float] {
    precondition(x.count == y.count, "Vector(\(x.count)) and Vector(\(y.count)) not compatible with element-wise modulo")
    var results = [Float](repeating: 0.0, count: x.count)
    vvfmodf(&results, x, y, [Int32(x.count)])

    return results
}

public func mod(_ x: [Double], _ y: [Double]) -> [Double] {
    precondition(x.count == y.count, "Vector(\(x.count)) and Vector(\(y.count)) not compatible with element-wise modulo")
    var results = [Double](repeating: 0.0, count: x.count)
    vvfmod(&results, x, y, [Int32(x.count)])

    return results
}

// MARK: Remainder

public func remainder(_ x: [Float], _ y: [Float]) -> [Float] {
    precondition(x.count == y.count, "Vector(\(x.count)) and Vector(\(y.count)) not compatible with element-wise remainder")
    var results = [Float](repeating: 0.0, count: x.count)
    vvremainderf(&results, x, y, [Int32(x.count)])

    return results
}

public func remainder(_ x: [Double], _ y: [Double]) -> [Double] {
    precondition(x.count == y.count, "Vector(\(x.count)) and Vector(\(y.count)) not compatible with element-wise remainder")
    var results = [Double](repeating: 0.0, count: x.count)
    vvremainder(&results, x, y, [Int32(x.count)])

    return results
}

// MARK: Square Root

/**
 Element-wise square root.
 - parameter x: an Array
 - returns: newly allocated √x
 */
public func sqrt(_ x: [Float]) -> [Float] {
    var results = [Float](repeating: 0.0, count: x.count)
    vvsqrtf(&results, x, [Int32(x.count)])

    return results
}

/**
 Element-wise square root.
    - parameter x: an Array
    - returns: newly allocated √x
*/
public func sqrt(_ x: [Double]) -> [Double] {
    var results = [Double](repeating: 0.0, count: x.count)
    vvsqrt(&results, x, [Int32(x.count)])

    return results
}

// MARK: Dot Product

/**
 Dot product of x and y.
 - parameter x: an Array
 - parameter y: an Array
 - precondition: x.count == y.count
 - returns: ∑(i=0..count-1) x[i].y[i]
*/
public func dot(_ x: [Float], _ y: [Float]) -> Float {
    precondition(x.count == y.count, "Vector(\(x.count)) and Vector(\(y.count)) not compatible with dot product")

    var result: Float = 0.0
    vDSP_dotpr(x, 1, y, 1, &result, vDSP_Length(x.count))

    return result
}

/**
 Dot product of x and y.
 - parameter x: an Array
 - parameter y: an Array
 - precondition: x.count == y.count
 - returns: ∑(i=0..count-1) x[i].y[i]
 */
public func dot(_ x: [Double], _ y: [Double]) -> Double {
    precondition(x.count == y.count, "Vector(\(x.count)) and Vector(\(y.count)) not compatible with dot product")

    var result: Double = 0.0
    vDSP_dotprD(x, 1, y, 1, &result, vDSP_Length(x.count))

    return result
}

// MARK: - Distance

/**
 Euclidean distance between x and y.
 - parameter x: an Array
 - parameter y: an Array
 - precondition: x.count == y.count
 - returns: ∑(i=0..count-1) (x[i]-y[i])²
 */
public func dist(_ x: [Float], _ y: [Float]) -> Float {
    precondition(x.count == y.count, "Vector(\(x.count)) and Vector(\(y.count)) not compatible with Euclidean distance")
    let sub = x - y
    var squared = [Float](repeating: 0.0, count: x.count)
    vDSP_vsq(sub, 1, &squared, 1, vDSP_Length(x.count))
    
    return sqrt(sum(squared))
}

/**
 Euclidean distance between x and y.
 - parameter x: an Array
 - parameter y: an Array
 - precondition: x.count == y.count
 - returns: ∑(i=0..count-1) (x[i]-y[i])²
 */
public func dist(_ x: [Double], _ y: [Double]) -> Double {
    precondition(x.count == y.count, "Vector(\(x.count)) and Vector(\(y.count)) not compatible with Euclidean distance")
    let sub = x - y
    var squared = [Double](repeating: 0.0, count: x.count)
    vDSP_vsqD(sub, 1, &squared, 1, vDSP_Length(x.count))
    
    return sqrt(sum(squared))
}

// MARK: - Operators

/**
 Array concatenation (default + operator behavior).
*/
public func concat(_ args: [Float]...) -> [Float] {
    var result = [Float]()
    for arg in args {
        result.append(contentsOf: arg)
    }
    assert(result.count == args.map({ $0.count }).reduce(0, +) )
    return result
}

/**
 Array concatenation (default + operator behavior).
 */
public func concat(_ args: [Double]...) -> [Double] {
    var result = [Double]()
    for arg in args {
        result.append(contentsOf: arg)
    }
    assert(result.count == args.map({ $0.count }).reduce(0, +) )
    return result
}

/**
 Add lhs and rhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: an Array.
 - returns: newly allocated lhs + rhs
 */
public func + (lhs: [Float], rhs: [Float]) -> [Float] {
    return add(lhs, rhs)
}

/**
 Add rhs to lhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: an Array.
 */
public func += (lhs: inout [Float], rhs: [Float]) {
    lhs = add(lhs, rhs)
}

/**
 Add lhs and rhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: an Array.
 - returns: newly allocated lhs + rhs
 */
public func + (lhs: [Double], rhs: [Double]) -> [Double] {
    return add(lhs, rhs)
}

/**
 Add rhs to lhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: an Array.
 */
public func += (lhs: inout [Double], rhs: [Double]) {
    lhs = add(lhs, rhs)
}

/**
 Add lhs and rhs (element-wise).
 - parameter lhs: a Float.
 - parameter rhs: an Array.
 - returns: newly allocated lhs + rhs
 */
public func + (lhs: Float, rhs: [Float]) -> [Float] {
    return add([Float](repeating: lhs, count: rhs.count), rhs)
}

/**
 Add lhs and rhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: a Float.
 - returns: newly allocated lhs + rhs
 */
public func + (lhs: [Float], rhs: Float) -> [Float] {
    return add(lhs, [Float](repeating: rhs, count: lhs.count))
}

/**
 Add rhs to lhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: a Float.
 */
public func += (lhs: inout [Float], rhs: Float) {
    lhs = add(lhs, [Float](repeating: rhs, count: lhs.count))
}

/**
 Add lhs and rhs (element-wise).
 - parameter lhs: a Double.
 - parameter rhs: an Array.
 - returns: newly allocated lhs + rhs
 */
public func + (lhs: Double, rhs: [Double]) -> [Double] {
    return add([Double](repeating: lhs, count: rhs.count), rhs)
}

/**
 Add lhs and rhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: a Double.
 - returns: newly allocated lhs + rhs
 */
public func + (lhs: [Double], rhs: Double) -> [Double] {
    return add(lhs, [Double](repeating: rhs, count: lhs.count))
}

/**
 Add rhs to lhs (element-wise).
 - parameter lhs: an Array.
 */
public func += (lhs: inout [Double], rhs: Double) {
    lhs = add(lhs, [Double](repeating: rhs, count: lhs.count))
}

/**
 Subtract rhs from lhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: an Array.
 - returns: newly allocated lhs - rhs
 */
public func - (lhs: [Float], rhs: [Float]) -> [Float] {
    return sub(lhs, rhs)
}

/**
 Subtract rhs from lhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: an Array.
 */
public func -= (lhs: inout [Float], rhs: [Float]) {
    lhs = sub(lhs, rhs)
}

/**
 Subtract rhs from lhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: an Array.
 - returns: newly allocated lhs - rhs
 */
public func - (lhs: [Double], rhs: [Double]) -> [Double] {
    return sub(lhs, rhs)
}

/**
 Subtract rhs from lhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: an Array.
 */
public func -= (lhs: inout [Double], rhs: [Double]) {
    lhs = sub(lhs, rhs)
}

/**
 Subtract rhs from lhs (element-wise).
 - parameter lhs: a Float.
 - parameter rhs: an Array.
 - returns: newly allocated lhs - rhs
 */
public func - (lhs: Float, rhs: [Float]) -> [Float] {
    return sub([Float](repeating: lhs, count: rhs.count), rhs)
}

/**
 Subtract rhs from lhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: a Float.
 - returns: newly allocated lhs - rhs
 */
public func - (lhs: [Float], rhs: Float) -> [Float] {
    return sub(lhs, [Float](repeating: rhs, count: lhs.count))
}

/**
 Subtract rhs from lhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: a Float.
 */
public func -= (lhs: inout [Float], rhs: Float) {
    lhs = sub(lhs, [Float](repeating: rhs, count: lhs.count))
}

/**
 Subtract rhs from lhs (element-wise).
 - parameter lhs: a Double.
 - parameter rhs: an Array.
 - returns: newly allocated lhs - rhs
 */
public func - (lhs: Double, rhs: [Double]) -> [Double] {
    return sub([Double](repeating: lhs, count: rhs.count), rhs)
}

/**
 Subtract rhs from lhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: a Double.
 - returns: newly allocated lhs - rhs
 */
public func - (lhs: [Double], rhs: Double) -> [Double] {
    return sub(lhs, [Double](repeating: rhs, count: lhs.count))
}

/**
 Subtract rhs from lhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: a Double.
 */
public func -= (lhs: inout [Double], rhs: Double) {
    lhs = sub(lhs, [Double](repeating: rhs, count: lhs.count))
}

/**
 Divide lhs by rhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: an Array.
 - returns: newly allocated lhs / rhs
 */
public func / (lhs: [Float], rhs: [Float]) -> [Float] {
    return div(lhs, rhs)
}

/**
 Divide lhs by rhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: an Array.
 */
public func /= (lhs: inout [Float], rhs: [Float]) {
    lhs = div(lhs, rhs)
}

/**
 Divide lhs by rhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: an Array.
 - returns: newly allocated lhs / rhs
 */
public func / (lhs: [Double], rhs: [Double]) -> [Double] {
    return div(lhs, rhs)
}

/**
 Divide lhs by rhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: an Array.
 */
public func /= (lhs: inout [Double], rhs: [Double]) {
    lhs = div(lhs, rhs)
}

/**
 Divide lhs by rhs (element-wise).
 - parameter lhs: a Float.
 - parameter rhs: an Array.
 - returns: newly allocated lhs / rhs
 */
public func / (lhs: Float, rhs: [Float]) -> [Float] {
    return div([Float](repeating: lhs, count: rhs.count), rhs)
}

/**
 Divide lhs by rhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: a Float.
 - returns: newly allocated lhs / rhs
 */
public func / (lhs: [Float], rhs: Float) -> [Float] {
    return div(lhs, [Float](repeating: rhs, count: lhs.count))
}

/**
 Divide lhs by rhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: a Float.
 */
public func /= (lhs: inout [Float], rhs: Float) {
    lhs = div(lhs, [Float](repeating: rhs, count: lhs.count))
}

/**
 Divide lhs by rhs (element-wise).
 - parameter lhs: a Double.
 - parameter rhs: an Array.
 - returns: newly allocated lhs / rhs
 */
public func / (lhs: Double, rhs: [Double]) -> [Double] {
    return div([Double](repeating: lhs, count: rhs.count), rhs)
}

/**
 Divide lhs by rhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: a Double.
 - returns: newly allocated lhs / rhs
 */
public func / (lhs: [Double], rhs: Double) -> [Double] {
    return div(lhs, [Double](repeating: rhs, count: lhs.count))
}

/**
 Divide lhs by rhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: a Double.
 */
public func /= (lhs: inout [Double], rhs: Double) {
    lhs = div(lhs, [Double](repeating: rhs, count: lhs.count))
}

/**
 Multiply lhs by rhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: an Array.
 - returns: newly allocated lhs * rhs
 */
public func * (lhs: [Float], rhs: [Float]) -> [Float] {
    return mul(lhs, rhs)
}

/**
 Multiply lhs by rhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: an Array.
 */
public func *= (lhs: inout [Float], rhs: [Float]) {
    lhs = mul(lhs, rhs)
}

/**
 Multiply lhs by rhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: an Array.
 - returns: newly allocated lhs * rhs
 */
public func * (lhs: [Double], rhs: [Double]) -> [Double] {
    return mul(lhs, rhs)
}

/**
 Multiply lhs by rhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: an Array.
 */
public func *= (lhs: inout [Double], rhs: [Double]) {
    lhs = mul(lhs, rhs)
}

/**
 Multiply lhs by rhs (element-wise).
 - parameter lhs: a Float.
 - parameter rhs: an Array.
 - returns: newly allocated lhs * rhs
 */
public func * (lhs: Float, rhs: [Float]) -> [Float] {
    return mul([Float](repeating: lhs, count: rhs.count), rhs)
}

/**
 Multiply lhs by rhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: a Float.
 - returns: newly allocated lhs * rhs
 */
public func * (lhs: [Float], rhs: Float) -> [Float] {
    return mul(lhs, [Float](repeating: rhs, count: lhs.count))
}

/**
 Multiply lhs by rhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: a Float.
 */
public func *= (lhs: inout [Float], rhs: Float) {
    lhs = mul(lhs, [Float](repeating: rhs, count: lhs.count))
}

/**
 Multiply lhs by rhs (element-wise).
 - parameter lhs: a Double.
 - parameter rhs: an Array.
 - returns: newly allocated lhs * rhs
 */
public func * (lhs: Double, rhs: [Double]) -> [Double] {
    return mul([Double](repeating: lhs, count: rhs.count), rhs)
}

/**
 Multiply lhs by rhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: a Double.
 - returns: newly allocated lhs * rhs
 */
public func * (lhs: [Double], rhs: Double) -> [Double] {
    return mul(lhs, [Double](repeating: rhs, count: lhs.count))
}

/**
 Multiply lhs by rhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: a Double.
 */
public func *= (lhs: inout [Double], rhs: Double) {
    lhs = mul(lhs, [Double](repeating: rhs, count: lhs.count))
}

public func % (lhs: [Float], rhs: [Float]) -> [Float] {
    return mod(lhs, rhs)
}

public func %= (lhs: inout [Float], rhs: [Float]) {
    lhs = mod(lhs, rhs)
}

public func % (lhs: [Double], rhs: [Double]) -> [Double] {
    return mod(lhs, rhs)
}

public func %= (lhs: inout [Double], rhs: [Double]) {
    lhs = mod(lhs, rhs)
}

public func % (lhs: Float, rhs: [Float]) -> [Float] {
    return mod([Float](repeating: lhs, count: rhs.count), rhs)
}

public func % (lhs: [Float], rhs: Float) -> [Float] {
    return mod(lhs, [Float](repeating: rhs, count: lhs.count))
}

public func %= (lhs: inout [Float], rhs: Float) {
    lhs = mod(lhs, [Float](repeating: rhs, count: lhs.count))
}

public func % (lhs: Double, rhs: [Double]) -> [Double] {
    return mod([Double](repeating: lhs, count: rhs.count), rhs)
}

public func % (lhs: [Double], rhs: Double) -> [Double] {
    return mod(lhs, [Double](repeating: rhs, count: lhs.count))
}

public func %= (lhs: inout [Double], rhs: Double) {
    lhs = mod(lhs, [Double](repeating: rhs, count: lhs.count))
}

/**
 Dot product of lhs and rhs.
 - parameter lhs: an Array
 - parameter rhs: an Array
 - precondition: lhs.count == rhs.count
 - returns: sum( lhs[0] * rhs[0] + ... + lhs[count-1] * rhs[count-1])
 */
infix operator • : MultiplicationPrecedence
public func • (lhs: [Double], rhs: [Double]) -> Double {
    return dot(lhs, rhs)
}

/**
 Dot product of lhs and rhs.
 - parameter lhs: an Array
 - parameter rhs: an Array
 - precondition: lhs.count == rhs.count
 - returns: sum( lhs[0] * rhs[0] + ... + lhs[count-1] * rhs[count-1])
 */
public func • (lhs: [Float], rhs: [Float]) -> Float {
    return dot(lhs, rhs)
}


