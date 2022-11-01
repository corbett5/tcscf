#pragma once

#include "LvArrayInterface.hpp"
#include "mathFunctions.hpp"

namespace tcscf::orthogonalization
{

namespace internal
{

/**
 */
template< typename T, int USD >
T innerProduct( ArraySlice1d< T const, USD > const & lhs, ArraySlice1d< T const, USD > const & rhs )
{
  LVARRAY_ASSERT_EQ( lhs.size(), rhs.size() );

  T dot = 0;

  for( IndexType i = 0; i < lhs.size(); ++i )
  {
    dot = dot + conj( lhs[ i ] ) * rhs[ i ];
  }

  return dot;
}

/**
 */
template< typename T, int USD >
RealType< T > norm( ArraySlice1d< T const, USD > const & v )
{
  RealType< T > dot = 0;

  for( IndexType i = 0; i < v.size(); ++i )
  {
    dot = dot + std::norm( v[ i ] );
  }

  return std::sqrt( dot );
}

/**
 */
template< typename T, int USD >
void scale( ArraySlice1d< T, USD > const & slice, T const scale )
{
  for( auto & value : slice )
  {
    value *= scale;
  }
}

/**
 */
template< typename T, int USD >
void scaledAdd(
  ArraySlice1d< T, USD > const & lhs,
  T const scale,
  ArraySlice1d< T const, USD > const & rhs )
{
  LVARRAY_ASSERT_EQ( lhs.size(), rhs.size() );

  for( IndexType i = 0; i < lhs.size(); ++i )
  {
    lhs[ i ] = lhs[ i ] + scale * rhs[ i ];
  }
}

} // namespace internal

/**
 */
template< typename T >
void modifiedGramSchmidt( std::vector< ArraySlice1d< T > > const & vectors )
{
  internal::scale( vectors[ 0 ], 1 / internal::norm( vectors[ 0 ].toSliceConst() ) );

  for( std::size_t i = 1; i < vectors.size(); ++i )
  {
    // TODO: for performance do this loop in a batch.
    for( std::size_t j = i; j < vectors.size(); ++j )
    {
      T const scale = internal::innerProduct( vectors[ i - 1 ].toSliceConst(), vectors[ j ].toSliceConst() );
      internal::scaledAdd( vectors[ j ], -scale, vectors[ i - 1 ].toSliceConst() );
    }

    internal::scale( vectors[ i ], 1 / internal::norm( vectors[ i ].toSliceConst() ) );
  }
}

} // namespace tcscf::orthogonalization
