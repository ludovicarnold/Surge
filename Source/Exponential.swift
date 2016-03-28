// Exponential.swift
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

// MARK: Exponentiation

/**
 Exponential.
 - returns: exp(x) (element-wise,  newly allocated).
 */
public func exp(x: [Float]) -> [Float] {
    var results = [Float](count: x.count, repeatedValue: 0.0)
    vvexpf(&results, x, [Int32(x.count)])

    return results
}

/**
 Exponential.
 - returns: exp(x) (element-wise,  newly allocated).
 */
public func exp(x: [Double]) -> [Double] {
    var results = [Double](count: x.count, repeatedValue: 0.0)
    vvexp(&results, x, [Int32(x.count)])

    return results
}

// MARK: Square Exponentiation

/**
 Square exponentiation.
 - returns: 2^x (element-wise,  newly allocated).
*/
public func exp2(x: [Float]) -> [Float] {
    var results = [Float](count: x.count, repeatedValue: 0.0)
    vvexp2f(&results, x, [Int32(x.count)])

    return results
}

/**
 Square exponentiation.
 - returns: 2^x (element-wise,  newly allocated).
 */
public func exp2(x: [Double]) -> [Double] {
    var results = [Double](count: x.count, repeatedValue: 0.0)
    vvexp2(&results, x, [Int32(x.count)])

    return results
}

// MARK: Natural Logarithm

/**
 Natural logarigthm.
 - returns: ln(x) (element-wise,  newly allocated).
*/
public func log(x: [Float]) -> [Float] {
    var results = [Float](x)
    vvlogf(&results, x, [Int32(x.count)])

    return results
}

/**
 Natural logarigthm.
 - returns: ln(x) (element-wise,  newly allocated).
 */
public func log(x: [Double]) -> [Double] {
    var results = [Double](x)
    vvlog(&results, x, [Int32(x.count)])

    return results
}

// MARK: Base-2 Logarithm

/**
 Base-2 logarithm.
 - returns: log2(x) (element-wise,  newly allocated).
*/
public func log2(x: [Float]) -> [Float] {
    var results = [Float](x)
    vvlog2f(&results, x, [Int32(x.count)])

    return results
}

/**
 Base-2 logarithm.
 - returns: log2(x) (element-wise,  newly allocated).
 */
public func log2(x: [Double]) -> [Double] {
    var results = [Double](x)
    vvlog2(&results, x, [Int32(x.count)])

    return results
}

// MARK: Base-10 Logarithm

/**
 Base-10 logarithm.
 - returns: log10(x) (element-wise,  newly allocated).
 */
public func log10(x: [Float]) -> [Float] {
    var results = [Float](x)
    vvlog10f(&results, x, [Int32(x.count)])

    return results
}

/**
 Base-10 logarithm.
 - returns: log10(x) (element-wise,  newly allocated).
 */
public func log10(x: [Double]) -> [Double] {
    var results = [Double](x)
    vvlog10(&results, x, [Int32(x.count)])

    return results
}

// MARK: Logarithmic Exponentiation

public func logb(x: [Float]) -> [Float] {
    var results = [Float](x)
    vvlogbf(&results, x, [Int32(x.count)])

    return results
}

public func logb(x: [Double]) -> [Double] {
    var results = [Double](x)
    vvlogb(&results, x, [Int32(x.count)])

    return results
}
