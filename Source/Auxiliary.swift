// Auxilliary.swift
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

// MARK: Absolute Value

/**
 Absolute value of an Array.
 - returns: |x| (element-wise,  newly allocated).
*/
public func abs(_ x: [Double]) -> [Double] {
    var results = [Double](repeating: 0.0, count: x.count)
    vvfabs(&results, x, [Int32(x.count)])

    return results
}

/**
 Absolute value of an Array.
 - returns: |x| (element-wise,  newly allocated).
 */
public func abs(_ x: [Float]) -> [Float] {
    var results = [Float](repeating: 0.0, count: x.count)
    vvfabsf(&results, x, [Int32(x.count)])

    return results
}

// MARK: Ceiling

/**
 Ceiling of an Array.
 - returns: ceil(x) (element-wise,  newly allocated).
 */
public func ceil(_ x: [Float]) -> [Float] {
    var results = [Float](repeating: 0.0, count: x.count)
    vvceilf(&results, x, [Int32(x.count)])

    return results
}

/**
 Ceiling of an Array.
 - returns: ceil(x) (element-wise,  newly allocated)
 */
public func ceil(_ x: [Double]) -> [Double] {
    var results = [Double](repeating: 0.0, count: x.count)
    vvceil(&results, x, [Int32(x.count)])

    return results
}

// MARK: Clip

/**
 Clip values of an Array between low and high (element-wise,  newly allocated).
 - returns: y = clip(x) such that:
     * y[i] == x[i] if low <= x[i] <= high
     * y[i] == low  if x[i] < low
     * y[i] == high if x[i] > high
 */
public func clip(_ x: [Float], low: Float, high: Float) -> [Float] {
    var results = [Float](repeating: 0.0, count: x.count), y = low, z = high
    vDSP_vclip(x, 1, &y, &z, &results, 1, vDSP_Length(x.count))

    return results
}

/**
 Clip values of an Array between low and high (element-wise,  newly allocated).
 - returns: y = clip(x) such that:
     * y[i] == x[i] if low <= x[i] <= high
     * y[i] == low  if x[i] < low
     * y[i] == high if x[i] > high
 */
public func clip(_ x: [Double], low: Double, high: Double) -> [Double] {
    var results = [Double](repeating: 0.0, count: x.count), y = low, z = high
    vDSP_vclipD(x, 1, &y, &z, &results, 1, vDSP_Length(x.count))

    return results
}

// MARK: Copy Sign

/**
 An array with given magnitude and sign (element-wise,  newly allocated).
 */
public func copysign(_ sign: [Float], magnitude: [Float]) -> [Float] {
    var results = [Float](repeating: 0.0, count: sign.count)
    vvcopysignf(&results, magnitude, sign, [Int32(sign.count)])

    return results
}

/**
 An array with given magnitude and sign (element-wise,  newly allocated).
 */
public func copysign(_ sign: [Double], magnitude: [Double]) -> [Double] {
    var results = [Double](repeating: 0.0, count: sign.count)
    vvcopysign(&results, magnitude, sign, [Int32(sign.count)])

    return results
}

// MARK: Floor

/**
 Floor of an Array.
 - returns: floor(x) (element-wise,  newly allocated).
 */
public func floor(_ x: [Float]) -> [Float] {
    var results = [Float](repeating: 0.0, count: x.count)
    vvfloorf(&results, x, [Int32(x.count)])

    return results
}

/**
 Floor of an Array.
 - returns: floor(x) (element-wise,  newly allocated).
 */
public func floor(_ x: [Double]) -> [Double] {
    var results = [Double](repeating: 0.0, count: x.count)
    vvfloor(&results, x, [Int32(x.count)])

    return results
}

// MARK: Negate

/**
 Opposite of an Array.
 - returns: -x (element-wise,  newly allocated).
 */
public func neg(_ x: [Float]) -> [Float] {
    var results = [Float](repeating: 0.0, count: x.count)
    vDSP_vneg(x, 1, &results, 1, vDSP_Length(x.count))

    return results
}

/**
 Opposite of an Array.
 - returns: -x (element-wise,  newly allocated).
 */
public func neg(_ x: [Double]) -> [Double] {
    var results = [Double](repeating: 0.0, count: x.count)
    vDSP_vnegD(x, 1, &results, 1, vDSP_Length(x.count))

    return results
}

// MARK: Reciprocal

/**
 Reciprocal of an Array.
 - returns: 1/x (element-wise,  newly allocated).
 */
public func rec(_ x: [Float]) -> [Float] {
    var results = [Float](repeating: 0.0, count: x.count)
    vvrecf(&results, x, [Int32(x.count)])

    return results
}

/**
 Reciprocal of an Array.
 - returns: 1/x (element-wise,  newly allocated).
 */
public func rec(_ x: [Double]) -> [Double] {
    var results = [Double](repeating: 0.0, count: x.count)
    vvrec(&results, x, [Int32(x.count)])

    return results
}

// MARK: Round

/**
 Round an Array to the nearest integer.
  - returns: round(x) (element-wise,  newly allocated).
 */
public func round(_ x: [Float]) -> [Float] {
    var results = [Float](repeating: 0.0, count: x.count)
    vvnintf(&results, x, [Int32(x.count)])

    return results
}

/**
 Round an Array to the nearest integer.
 - returns: round(x) (element-wise,  newly allocated).
 */
public func round(_ x: [Double]) -> [Double] {
    var results = [Double](repeating: 0.0, count: x.count)
    vvnint(&results, x, [Int32(x.count)])

    return results
}

// MARK: Threshold

/**
 Maximum of x and low (element-wise,  newly allocated).
 */
public func threshold(_ x: [Float], low: Float) -> [Float] {
    var results = [Float](repeating: 0.0, count: x.count), y = low
    vDSP_vthr(x, 1, &y, &results, 1, vDSP_Length(x.count))

    return results
}

/**
 Maximum of x and low (element-wise,  newly allocated).
 */
public func threshold(_ x: [Double], low: Double) -> [Double] {
    var results = [Double](repeating: 0.0, count: x.count), y = low
    vDSP_vthrD(x, 1, &y, &results, 1, vDSP_Length(x.count))

    return results
}

// MARK: Truncate

/**
 Integer trucation of an Array (element-wise, newly allocated).
 */
public func trunc(_ x: [Float]) -> [Float] {
    var results = [Float](repeating: 0.0, count: x.count)
    vvintf(&results, x, [Int32(x.count)])

    return results
}

/**
 Integer trucation of an Array (element-wise,  newly allocated).
 */
public func trunc(_ x: [Double]) -> [Double] {
    var results = [Double](repeating: 0.0, count: x.count)
    vvint(&results, x, [Int32(x.count)])

    return results
}

prefix operator -
/**
 Unary minus.
 */
public prefix func - (value: [Float]) -> [Float] {
    return neg(value)
}

/**
 Unary minus.
 */
public prefix func - (value: [Double]) -> [Double] {
    return neg(value)
}
