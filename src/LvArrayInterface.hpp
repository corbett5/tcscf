#pragma once

#include "Array.hpp"
#include "ArrayOfArrays.hpp"
#include "ChaiBuffer.hpp"
#include "tensorOps.hpp"
#include "memcpy.hpp"
#include "output.hpp"

#define CHECK_BOUNDS_1( array, size0 ) \
  LVARRAY_ERROR_IF_NE( array.size( 0 ), size0 )

#define CHECK_BOUNDS_2( array, size0, size1 ) \
  LVARRAY_ERROR_IF_NE( array.size( 0 ), size0 ); \
  LVARRAY_ERROR_IF_NE( array.size( 1 ), size1 )

#define CHECK_BOUNDS_3( array, size0, size1, size2 ) \
  LVARRAY_ERROR_IF_NE( array.size( 0 ), size0 ); \
  LVARRAY_ERROR_IF_NE( array.size( 1 ), size1 ); \
  LVARRAY_ERROR_IF_NE( array.size( 2 ), size2 )

#define CHECK_BOUNDS_6( array, size0, size1, size2, size3, size4, size5 ) \
  LVARRAY_ERROR_IF_NE( array.size( 0 ), size0 ); \
  LVARRAY_ERROR_IF_NE( array.size( 1 ), size1 ); \
  LVARRAY_ERROR_IF_NE( array.size( 2 ), size2 ); \
  LVARRAY_ERROR_IF_NE( array.size( 3 ), size3 ); \
  LVARRAY_ERROR_IF_NE( array.size( 4 ), size4 ); \
  LVARRAY_ERROR_IF_NE( array.size( 5 ), size5 )

namespace tcscf
{

using IndexType = std::ptrdiff_t;

/// Similar to std::array< T, N >, but works on device.
template< typename T, camp::idx_t N >
using CArray = LvArray::typeManipulation::CArray< T, N >;

/// A slice of arbitrary dimension.
template< typename T, int N, int USD = N - 1 >
using ArraySlice = LvArray::ArraySlice< T, N, USD, IndexType >;

/// A view of arbitrary dimension.
template< typename T, int N, int USD >
using ArrayView = LvArray::ArrayView< T, N, USD, IndexType, LvArray::ChaiBuffer >;

/// An array of arbitrary dimension.
template< typename T, int N, typename PERMUTATION >
using Array = LvArray::Array< T, N, PERMUTATION, IndexType, LvArray::ChaiBuffer >;

/// A one dimensional slice.
template< typename T, int USD=0 >
using ArraySlice1d = ArraySlice< T, 1, USD >;

/// A one dimensional view.
template< typename T >
using ArrayView1d = ArrayView< T, 1, 0 >;

/// A one dimensional array.
template< typename T >
using Array1d = Array< T, 1, RAJA::PERM_I >;

/// A one dimensional slice.
template< typename T, int USD=1 >
using ArraySlice2d = ArraySlice< T, 2, USD >;

/// A two dimensional view.
template< typename T, int USD=1 >
using ArrayView2d = ArrayView< T, 2, USD >;

/// A two dimensional array.
template< typename T, typename PERM=RAJA::PERM_IJ >
using Array2d = Array< T, 2, PERM >;

/// A three dimensional view.
template< typename T, int USD=2 >
using ArrayView3d = ArrayView< T, 3, USD >;

/// A three dimensional array.
template< typename T, typename PERM=RAJA::PERM_IJK >
using Array3d = Array< T, 3, PERM >;

/// A four dimensional array.
template< typename T, typename PERM=RAJA::PERM_IJKL >
using Array4d = Array< T, 4, PERM >;

/// A four dimensional view.
template< typename T, int USD=3 >
using ArrayView4d = ArrayView< T, 4, USD >;

/// A six dimensional array.
template< typename T, typename PERM=camp::make_idx_seq_t< 6 > >
using Array6d = Array< T, 6, PERM >;

/// A six dimensional view.
template< typename T, int USD=5 >
using ArrayView6d = ArrayView< T, 6, 5 >;

template< typename T >
using ArrayOfArrays = LvArray::ArrayOfArrays< T, IndexType, LvArray::ChaiBuffer >;

template< typename T >
using ArrayOfArraysView = LvArray::ArrayOfArraysView< T, IndexType, false, LvArray::ChaiBuffer >;

} // namespace tcscf
