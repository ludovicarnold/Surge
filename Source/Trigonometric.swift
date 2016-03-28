// Trigonometric.swift
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

// MARK: Sine-Cosine

/**
 (Sine, Cosine).
 - returns: (sin(x), cos(x)) (element-wise,  newly allocated).
 */
public func sincos(x: [Float]) -> (sin: [Float], cos: [Float]) {
    var sin = [Float](count: x.count, repeatedValue: 0.0)
    var cos = [Float](count: x.count, repeatedValue: 0.0)
    vvsincosf(&sin, &cos, x, [Int32(x.count)])

    return (sin, cos)
}

/**
 (Sine, Cosine).
 - returns: (sin(x), cos(x)) (element-wise,  newly allocated).
 */
public func sincos(x: [Double]) -> (sin: [Double], cos: [Double]) {
    var sin = [Double](count: x.count, repeatedValue: 0.0)
    var cos = [Double](count: x.count, repeatedValue: 0.0)
    vvsincos(&sin, &cos, x, [Int32(x.count)])

    return (sin, cos)
}

// MARK: Sine

/**
 Sine.
 - returns: sin(x) (element-wise,  newly allocated).
 */
public func sin(x: [Float]) -> [Float] {
    var results = [Float](count: x.count, repeatedValue: 0.0)
    vvsinf(&results, x, [Int32(x.count)])

    return results
}

/**
 Sine.
 - returns: sin(x) (element-wise,  newly allocated).
 */
public func sin(x: [Double]) -> [Double] {
    var results = [Double](count: x.count, repeatedValue: 0.0)
    vvsin(&results, x, [Int32(x.count)])

    return results
}

// MARK: Cosine

/**
 Cosine.
 - returns: cos(x) (element-wise,  newly allocated).
 */
public func cos(x: [Float]) -> [Float] {
    var results = [Float](count: x.count, repeatedValue: 0.0)
    vvcosf(&results, x, [Int32(x.count)])

    return results
}

/**
 Cosine.
 - returns: cos(x) (element-wise,  newly allocated).
 */
public func cos(x: [Double]) -> [Double] {
    var results = [Double](count: x.count, repeatedValue: 0.0)
    vvcos(&results, x, [Int32(x.count)])

    return results
}

// MARK: Tangent

/**
 Tangent.
 - returns: tan(x) (element-wise,  newly allocated).
 */
public func tan(x: [Float]) -> [Float] {
    var results = [Float](count: x.count, repeatedValue: 0.0)
    vvtanf(&results, x, [Int32(x.count)])

    return results
}

/**
 Tangent.
 - returns: tan(x) (element-wise,  newly allocated).
 */
public func tan(x: [Double]) -> [Double] {
    var results = [Double](count: x.count, repeatedValue: 0.0)
    vvtan(&results, x, [Int32(x.count)])

    return results
}

// MARK: Arcsine

/**
 Arcsine.
 - returns: asin(x) (element-wise,  newly allocated).
 */
public func asin(x: [Float]) -> [Float] {
    var results = [Float](count: x.count, repeatedValue: 0.0)
    vvasinf(&results, x, [Int32(x.count)])

    return results
}

/**
 Arcsine.
 - returns: asin(x) (element-wise,  newly allocated).
 */
public func asin(x: [Double]) -> [Double] {
    var results = [Double](count: x.count, repeatedValue: 0.0)
    vvasin(&results, x, [Int32(x.count)])

    return results
}

// MARK: Arccosine

/**
 Arccosine.
 - returns: acos(x) (element-wise,  newly allocated).
 */
public func acos(x: [Float]) -> [Float] {
    var results = [Float](count: x.count, repeatedValue: 0.0)
    vvacosf(&results, x, [Int32(x.count)])

    return results
}

/**
 Arccosine.
 - returns: acos(x) (element-wise,  newly allocated).
 */
public func acos(x: [Double]) -> [Double] {
    var results = [Double](count: x.count, repeatedValue: 0.0)
    vvacos(&results, x, [Int32(x.count)])

    return results
}

// MARK: Arctangent

/**
 Arctangent.
 - returns: atan(x) (element-wise,  newly allocated).
 */
public func atan(x: [Float]) -> [Float] {
    var results = [Float](count: x.count, repeatedValue: 0.0)
    vvatanf(&results, x, [Int32(x.count)])

    return results
}

/**
 Arctangent.
 - returns: atan(x) (element-wise,  newly allocated).
 */
public func atan(x: [Double]) -> [Double] {
    var results = [Double](count: x.count, repeatedValue: 0.0)
    vvatan(&results, x, [Int32(x.count)])

    return results
}

// MARK: -

// MARK: Radians to Degrees

/**
 Radians to Degrees (element-wise,  newly allocated).
 */
func rad2deg(x: [Float]) -> [Float] {
    var results = [Float](count: x.count, repeatedValue: 0.0)
    let divisor = [Float](count: x.count, repeatedValue: Float(M_PI / 180.0))
    vvdivf(&results, x, divisor, [Int32(x.count)])

    return results
}

/**
 Radians to Degrees (element-wise,  newly allocated).
 */
func rad2deg(x: [Double]) -> [Double] {
    var results = [Double](count: x.count, repeatedValue: 0.0)
    let divisor = [Double](count: x.count, repeatedValue: M_PI / 180.0)
    vvdiv(&results, x, divisor, [Int32(x.count)])

    return results
}

// MARK: Degrees to Radians

/**
 Degrees to Radians (element-wise,  newly allocated).
 */
func deg2rad(x: [Float]) -> [Float] {
    var results = [Float](count: x.count, repeatedValue: 0.0)
    let divisor = [Float](count: x.count, repeatedValue: Float(180.0 / M_PI))
    vvdivf(&results, x, divisor, [Int32(x.count)])

    return results
}

/**
 Degrees to Radians (element-wise,  newly allocated).
 */
func deg2rad(x: [Double]) -> [Double] {
    var results = [Double](count: x.count, repeatedValue: 0.0)
    let divisor = [Double](count: x.count, repeatedValue: 180.0 / M_PI)
    vvdiv(&results, x, divisor, [Int32(x.count)])

    return results
}
