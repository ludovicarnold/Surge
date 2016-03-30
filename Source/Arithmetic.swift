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
public func sum(x: [Float]) -> Float {
    var result: Float = 0.0
    vDSP_sve(x, 1, &result, vDSP_Length(x.count))

    return result
}

/**
 Sum elements of an Array.
 - parameter x: the array to sum over.
 - returns: ∑(i=0..count-1) x[i]
 */
public func sum(x: [Double]) -> Double {
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
public func asum(x: [Float]) -> Float {
    return cblas_sasum(Int32(x.count), x, 1)
}

/**
 Sum the absolute values of an Array.
 - parameter x: the array to sum over.
 - returns: ∑(i=0..count-1) |x[i]|
 */
public func asum(x: [Double]) -> Double {
    return cblas_dasum(Int32(x.count), x, 1)
}

// MARK: Maximum

/**
 The maximum value of an array.
 - parameter x: the input array.
 - returns: M | ∃i, x[i] = M, ∀i, x[i] ≤ M.
 */
public func max(x: [Float]) -> Float {
    var result: Float = 0.0
    vDSP_maxv(x, 1, &result, vDSP_Length(x.count))

    return result
}

/**
 The maximum value of an array.
 - parameter x: the input array.
 - returns: M | ∃i, x[i] = M, ∀i, x[i] ≤ M.
 */
public func max(x: [Double]) -> Double {
    var result: Double = 0.0
    vDSP_maxvD(x, 1, &result, vDSP_Length(x.count))

    return result
}

// MARK: Minimum

/**
 The minimum value of an array.
 - parameter x: the input array.
 - returns: m | ∃i, x[i] = m, ∀i, x[i] ≥ m.
 */
public func min(x: [Float]) -> Float {
    var result: Float = 0.0
    vDSP_minv(x, 1, &result, vDSP_Length(x.count))

    return result
}

/**
 The minimum value of an array.
 - parameter x: the input array.
 - returns: m | ∃i, x[i] = m, ∀i, x[i] ≥ m.
 */
public func min(x: [Double]) -> Double {
    var result: Double = 0.0
    vDSP_minvD(x, 1, &result, vDSP_Length(x.count))

    return result
}

// MARK: Mean

/**
 The mean value of an array.
 - parameter x: the input array.
 - returns: (1/count) ∑(i=0..count-1) x[i]
 */
public func mean(x: [Float]) -> Float {
    var result: Float = 0.0
    vDSP_meanv(x, 1, &result, vDSP_Length(x.count))

    return result
}

/**
 The mean value of an array.
 - parameter x: the input array.
 - returns: (1/count) ∑(i=0..count-1) x[i]
 */
public func mean(x: [Double]) -> Double {
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
public func meamg(x: [Float]) -> Float {
    var result: Float = 0.0
    vDSP_meamgv(x, 1, &result, vDSP_Length(x.count))

    return result
}

/**
 The mean magnitude of an array.
 - parameter x: the input array.
 - returns: (1/count) ∑(i=0..count-1) |x[i]|
 */
public func meamg(x: [Double]) -> Double {
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
public func measq(x: [Float]) -> Float {
    var result: Float = 0.0
    vDSP_measqv(x, 1, &result, vDSP_Length(x.count))

    return result
}

/**
 The mean squared  value of an array.
 - parameter x: the input array.
 - returns: (1/count) ∑(i=0..count-1) x[i]²
 */
public func measq(x: [Double]) -> Double {
    var result: Double = 0.0
    vDSP_measqvD(x, 1, &result, vDSP_Length(x.count))

    return result
}

// MARK: Add

/**
 Element-wise array addition.
 - parameter x: an Array
 - parameter y: an Array
 - returns: newly allocated x + y
 */
public func add(x: [Float], y: [Float]) -> [Float] {
    var results = [Float](y)
    cblas_saxpy(Int32(x.count), 1.0, x, 1, &results, 1)

    return results
}

/**
 Element-wise array addition.
 - parameter x: an Array
 - parameter y: an Array
 - returns: newly allocated x + y
 */
public func add(x: [Double], y: [Double]) -> [Double] {
    var results = [Double](y)
    cblas_daxpy(Int32(x.count), 1.0, x, 1, &results, 1)

    return results
}

// MARK: Subtraction

/**
 Element-wise array subtraction.
 - parameter x: an Array
 - parameter y: an Array
 - returns: newly allocated x - y
 */
public func sub(x: [Float], y: [Float]) -> [Float] {
    var results = [Float](y)
    catlas_saxpby(Int32(x.count), 1.0, x, 1, -1, &results, 1)
    
    return results
}

/**
 Element-wise array subtraction.
 - parameter x: an Array
 - parameter y: an Array
 - returns: newly allocated x - y
 */
public func sub(x: [Double], y: [Double]) -> [Double] {
    var results = [Double](y)
    catlas_daxpby(Int32(x.count), 1.0, x, 1, -1, &results, 1)
    
    return results
}

// MARK: Multiply

/**
 Element-wise array multiplication.
 - parameter x: an Array
 - parameter y: an Array
 - returns: newly allocated x * y
 */
public func mul(x: [Float], y: [Float]) -> [Float] {
    var results = [Float](count: x.count, repeatedValue: 0.0)
    vDSP_vmul(x, 1, y, 1, &results, 1, vDSP_Length(x.count))

    return results
}

/**
 Element-wise array multiplication.
 - parameter x: an Array
 - parameter y: an Array
 - returns: newly allocated x * y
 */
public func mul(x: [Double], y: [Double]) -> [Double] {
    var results = [Double](count: x.count, repeatedValue: 0.0)
    vDSP_vmulD(x, 1, y, 1, &results, 1, vDSP_Length(x.count))

    return results
}

// MARK: Divide

/**
 Element-wise array division.
 - parameter x: an Array
 - parameter y: an Array
 - returns: newly allocated x / y
 */
public func div(x: [Float], y: [Float]) -> [Float] {
    var results = [Float](count: x.count, repeatedValue: 0.0)
    vvdivf(&results, x, y, [Int32(x.count)])

    return results
}

/**
 Element-wise array division.
 - parameter x: an Array
 - parameter y: an Array
 - returns: newly allocated x / y
 */
public func div(x: [Double], y: [Double]) -> [Double] {
    var results = [Double](count: x.count, repeatedValue: 0.0)
    vvdiv(&results, x, y, [Int32(x.count)])

    return results
}

public func mod(x: [Float], y: [Float]) -> [Float] {
    var results = [Float](count: x.count, repeatedValue: 0.0)
    vvfmodf(&results, x, y, [Int32(x.count)])

    return results
}

public func mod(x: [Double], y: [Double]) -> [Double] {
    var results = [Double](count: x.count, repeatedValue: 0.0)
    vvfmod(&results, x, y, [Int32(x.count)])

    return results
}

// MARK: Remainder

public func remainder(x: [Float], y: [Float]) -> [Float] {
    var results = [Float](count: x.count, repeatedValue: 0.0)
    vvremainderf(&results, x, y, [Int32(x.count)])

    return results
}

public func remainder(x: [Double], y: [Double]) -> [Double] {
    var results = [Double](count: x.count, repeatedValue: 0.0)
    vvremainder(&results, x, y, [Int32(x.count)])

    return results
}

// MARK: Square Root

/**
 Element-wise square root.
 - parameter x: an Array
 - returns: newly allocated √x
 */
public func sqrt(x: [Float]) -> [Float] {
    var results = [Float](count: x.count, repeatedValue: 0.0)
    vvsqrtf(&results, x, [Int32(x.count)])

    return results
}

/**
 Element-wise square root.
    - parameter x: an Array
    - returns: newly allocated √x
*/
public func sqrt(x: [Double]) -> [Double] {
    var results = [Double](count: x.count, repeatedValue: 0.0)
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
public func dot(x: [Float], y: [Float]) -> Float {
    precondition(x.count == y.count, "Vectors must have equal count")

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
public func dot(x: [Double], y: [Double]) -> Double {
    precondition(x.count == y.count, "Vectors must have equal count")

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
public func dist(x: [Float], y: [Float]) -> Float {
    precondition(x.count == y.count, "Vectors must have equal count")
    let sub = x - y
    var squared = [Float](count: x.count, repeatedValue: 0.0)
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
public func dist(x: [Double], y: [Double]) -> Double {
    precondition(x.count == y.count, "Vectors must have equal count")
    let sub = x - y
    var squared = [Double](count: x.count, repeatedValue: 0.0)
    vDSP_vsqD(sub, 1, &squared, 1, vDSP_Length(x.count))
    
    return sqrt(sum(squared))
}

// MARK: - Operators

/**
 Add lhs and rhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: an Array.
 - returns: newly allocated lhs + rhs
 */
public func + (lhs: [Float], rhs: [Float]) -> [Float] {
    return add(lhs, y: rhs)
}

/**
 Add lhs and rhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: an Array.
 - returns: newly allocated lhs + rhs
 */
public func + (lhs: [Double], rhs: [Double]) -> [Double] {
    return add(lhs, y: rhs)
}

/**
 Add lhs and rhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: a Float.
 - returns: newly allocated lhs + rhs
 */
public func + (lhs: [Float], rhs: Float) -> [Float] {
    return add(lhs, y: [Float](count: lhs.count, repeatedValue: rhs))
}

/**
 Add lhs and rhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: a Double.
 - returns: newly allocated lhs + rhs
 */
public func + (lhs: [Double], rhs: Double) -> [Double] {
    return add(lhs, y: [Double](count: lhs.count, repeatedValue: rhs))
}

/**
 Subtract rhs from lhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: an Array.
 - returns: newly allocated lhs - rhs
 */
public func - (lhs: [Float], rhs: [Float]) -> [Float] {
    return sub(lhs, y: rhs)
}

/**
 Subtract rhs from lhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: an Array.
 - returns: newly allocated lhs - rhs
 */
public func - (lhs: [Double], rhs: [Double]) -> [Double] {
    return sub(lhs, y: rhs)
}

/**
 Subtract rhs from lhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: a Float.
 - returns: newly allocated lhs - rhs
 */
public func - (lhs: [Float], rhs: Float) -> [Float] {
    return sub(lhs, y: [Float](count: lhs.count, repeatedValue: rhs))
}

/**
 Subtract rhs from lhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: a Double.
 - returns: newly allocated lhs - rhs
 */
public func - (lhs: [Double], rhs: Double) -> [Double] {
    return sub(lhs, y: [Double](count: lhs.count, repeatedValue: rhs))
}

/**
 Divide lhs by rhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: an Array.
 - returns: newly allocated lhs / rhs
 */
public func / (lhs: [Float], rhs: [Float]) -> [Float] {
    return div(lhs, y: rhs)
}

/**
 Divide lhs by rhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: an Array.
 - returns: newly allocated lhs / rhs
 */
public func / (lhs: [Double], rhs: [Double]) -> [Double] {
    return div(lhs, y: rhs)
}

/**
 Divide lhs by rhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: a Float.
 - returns: newly allocated lhs / rhs
 */
public func / (lhs: [Float], rhs: Float) -> [Float] {
    return div(lhs, y: [Float](count: lhs.count, repeatedValue: rhs))
}

/**
 Divide lhs by rhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: a Double.
 - returns: newly allocated lhs / rhs
 */
public func / (lhs: [Double], rhs: Double) -> [Double] {
    return div(lhs, y: [Double](count: lhs.count, repeatedValue: rhs))
}

/**
 Multiply lhs by rhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: an Array.
 - returns: newly allocated lhs * rhs
 */
public func * (lhs: [Float], rhs: [Float]) -> [Float] {
    return mul(lhs, y: rhs)
}

/**
 Multiply lhs by rhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: an Array.
 - returns: newly allocated lhs * rhs
 */
public func * (lhs: [Double], rhs: [Double]) -> [Double] {
    return mul(lhs, y: rhs)
}

/**
 Multiply lhs by rhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: a Float.
 - returns: newly allocated lhs * rhs
 */
public func * (lhs: [Float], rhs: Float) -> [Float] {
    return mul(lhs, y: [Float](count: lhs.count, repeatedValue: rhs))
}

/**
 Multiply lhs by rhs (element-wise).
 - parameter lhs: an Array.
 - parameter rhs: a Double.
 - returns: newly allocated lhs * rhs
 */
public func * (lhs: [Double], rhs: Double) -> [Double] {
    return mul(lhs, y: [Double](count: lhs.count, repeatedValue: rhs))
}

public func % (lhs: [Float], rhs: [Float]) -> [Float] {
    return mod(lhs, y: rhs)
}

public func % (lhs: [Double], rhs: [Double]) -> [Double] {
    return mod(lhs, y: rhs)
}

public func % (lhs: [Float], rhs: Float) -> [Float] {
    return mod(lhs, y: [Float](count: lhs.count, repeatedValue: rhs))
}

public func % (lhs: [Double], rhs: Double) -> [Double] {
    return mod(lhs, y: [Double](count: lhs.count, repeatedValue: rhs))
}

/**
 Dot product of lhs and rhs.
 - parameter lhs: an Array
 - parameter rhs: an Array
 - precondition: lhs.count == rhs.count
 - returns: sum( lhs[0] * rhs[0] + ... + lhs[count-1] * rhs[count-1])
 */
infix operator • {}
public func • (lhs: [Double], rhs: [Double]) -> Double {
    return dot(lhs, y: rhs)
}

/**
 Dot product of lhs and rhs.
 - parameter lhs: an Array
 - parameter rhs: an Array
 - precondition: lhs.count == rhs.count
 - returns: sum( lhs[0] * rhs[0] + ... + lhs[count-1] * rhs[count-1])
 */
public func • (lhs: [Float], rhs: [Float]) -> Float {
    return dot(lhs, y: rhs)
}
