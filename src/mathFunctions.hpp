#pragma once

#include "LvArrayInterface.hpp"
#include "caliperInterface.hpp"

#include <cmath>

namespace tcscf
{

namespace internal
{

template< typename >
struct IsComplex
{
  static constexpr bool value = false;
};

template< typename T >
struct IsComplex< std::complex< T > >
{
  static constexpr bool value = true;
};

} // namespace internal

///////////////////////////////////////////////////////////////////////////////////////////////////
// Constants
///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename T >
constexpr T pi = T( 3.1415926535897932385L );

template< typename T >
constexpr T ln2 = T( 0.693147180559945309L );

template< typename T >
constexpr std::complex< T > I = std::complex< T >( 0, 1 );

///////////////////////////////////////////////////////////////////////////////////////////////////
// Math helpers
///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename T >
constexpr bool isComplex = internal::IsComplex< T >::value;

template< typename T >
constexpr std::enable_if_t< !isComplex< T >, T >
conj( T const & value )
{ return value; }

template< typename T >
constexpr std::enable_if_t< isComplex< T >, T >
conj( T const & value )
{ return std::conj( value ); }

template< typename T >
std::complex< T > operator*( int const x, std::complex< T > const & z )
{ return { z.real() * x, z.imag() * x }; }

template< typename T >
std::complex< T > operator*( std::complex< T > const & z, int const x )
{ return { z.real() * x, z.imag() * x }; }

template< typename T >
std::complex< T > operator/( std::complex< T > const & z, int const x )
{ return { z.real() / x, z.imag() / x }; }

/**
 * 
 */
constexpr std::int64_t truncatedFactorial( int const max, int const min )
{
  std::int64_t result = 1;
  for( int i = min + 1; i <= max; ++i )
  {
    result *= i;
  }

  return result;
}

/**
 * 
 */
template< typename T >
constexpr std::complex< T > cExp( T const theta )
{
  T s, c;
  LvArray::math::sincos( theta, s, c );
  return { c, s };
}

/**
 * 
 */
template< typename T >
T sphericalHarmonicMagnitude( int const l, int const m, T const theta )
{
  return std::sph_legendre( l, std::abs( m ), theta );
}

/**
 * 
 */
template< typename T >
std::complex< T > sphericalHarmonic( int const l, int const m, T const theta, T const phi )
{
  return sphericalHarmonicMagnitude( l, m, theta ) * cExp( m * phi );
}

/**
 * 
 */
template< typename T >
T assocLaguerre( int const n, int const k, T const x )
{
  return std::assoc_laguerre( n, k, x );
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Coordinate transformations
///////////////////////////////////////////////////////////////////////////////////////////////////

/**
 *
 */
template< typename T >
constexpr CArray< T, 3 > sphericalToCartesian( T const r, T const theta, T const phi )
{
  T const x = r * std::cos( phi ) * std::sin( theta );
  T const y = r * std::sin( phi ) * std::sin( theta );
  T const z = r * std::cos( theta );

  return { x, y, z };
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Matrix stuff
///////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * 
 */
template< typename T, typename LAMBDA >
void fillOneElectronHermitianMatrix(
  ArrayView2d< T > const & matrix,
  LAMBDA && lambda )
{
  TCSCF_MARK_FUNCTION;

  LVARRAY_ERROR_IF_NE( matrix.size( 0 ), matrix.size( 1 ) );

  using ResultType = decltype( lambda( 0, 0 ) );

  for( int a = 0; a < matrix.size( 0 ); ++a )
  {
    for( int b = a; b < matrix.size( 0 ); ++b )
    {
      ResultType const value = lambda( a, b );
      matrix( a, b ) += value;

      if( a != b )
      {
        matrix( b, a ) += conj( value );
      }
    }
  }
}

/**
 * 
 */
template< typename T, typename LAMBDA >
void fillTwoElectronSymmetricHermitianArray(
  ArrayView4d< T > const & array,
  bool const realBasisFunctions,
  LAMBDA && lambda )
{
  TCSCF_MARK_FUNCTION;

  LVARRAY_ERROR_IF_NE( array.size( 0 ), array.size( 1 ) );
  LVARRAY_ERROR_IF_NE( array.size( 0 ), array.size( 2 ) );
  LVARRAY_ERROR_IF_NE( array.size( 0 ), array.size( 3 ) );

  int const N = array.size( 0 );

  for( int a = 0; a < N; ++a )
  {
    for( int b = a; b < N; ++b )
    {
      for( int c = a; c < N; ++c )
      {
        int const dStart = realBasisFunctions ? b : a;
        for( int d = dStart; d < N; ++d )
        {
          T const value{ lambda( a, b, c, d ) };

          array( a, b, c, d ) += value;

          bool const useSymmetry = a != b;
          bool const useHermitian = a != c && a != d;
          bool const useReal = realBasisFunctions && d != b;

          if( useSymmetry )
          {
            array( b, a, d, c ) += value;
          }

          if( useHermitian || (realBasisFunctions && a != c) )
          {
            array( c, d, a, b ) += conj( value );
          }

          if( useHermitian && useSymmetry )
          { 
            array( d, c, b, a ) += conj( value );
          }

          if( useReal )
          {
            array( a, d, c, b ) += value;
          }

          if( useReal && useSymmetry )
          {
            array( d, a, b, c ) += value;
          }

          if( useReal && useHermitian )
          {
            array( c, b, a, d ) += value;
          }

          if( useReal && useHermitian && useSymmetry )
          {
            array( b, c, d, a ) += value;
          }
        }
      }
    }
  }
}


} // namespace tcscf