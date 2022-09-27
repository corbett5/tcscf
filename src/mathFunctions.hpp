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

/**
 * 
 */
template< typename T >
std::pair< double, double > meanAndStd( ArrayView1d< T const > const & values )
{
  double mean = 0;
  for( auto value : values )
  {
    mean += value;
  }

  mean /= values.size();

  double var = 0;
  for( auto value : values )
  {
    var += std::pow( value - mean, 2 );
  }

  var /= values.size();

  return { mean, std::pow( var, 0.5 ) };
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Geometry
///////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * 
 */
template< typename REAL >
constexpr REAL calculateR12(
  REAL const r1,
  REAL const theta1,
  REAL const phi1,
  REAL const r2,
  REAL const theta2,
  REAL const phi2 )
{
  REAL r12 = std::pow( r1, 2 ) + std::pow( r2, 2 );
  r12 -= 2 * r1 * r2 * (std::sin( theta1 ) * std::sin( theta2 ) * std::cos( phi1  - phi2 ) + std::cos( theta1 ) * std::cos( theta2 ));
  return std::sqrt( std::abs( r12 ) );
}

template< typename REAL >
struct Cartesian;

template< typename REAL >
struct Spherical
{
  constexpr operator Cartesian< REAL >() const
  {
    REAL sinPhi, cosPhi;
    LvArray::math::sincos( _phi, sinPhi, cosPhi );

    REAL sinTheta, cosTheta;
    LvArray::math::sincos( _theta, sinTheta, cosTheta );

    return { _r * cosPhi * sinTheta, _r * sinPhi * sinTheta, _r * cosTheta };
  }

  constexpr REAL x() const
  { return _r * std::cos( _phi ) * std::sin( _theta ); }

  constexpr REAL y() const
  { return _r * std::sin( _phi ) * std::sin( _theta ); }

  constexpr REAL z() const
  { return _r * std::cos( _theta ); }

  constexpr REAL r() const
  { return _r; }

  constexpr REAL theta() const
  { return _theta; }

  constexpr REAL phi() const
  { return _phi; }

  REAL const _r;
  REAL const _theta;
  REAL const _phi;
};


template< typename REAL >
struct Cartesian
{
  constexpr REAL x() const
  { return _x; }

  constexpr REAL y() const
  { return _y; }

  constexpr REAL z() const
  { return _z; }

  constexpr REAL r() const
  { return std::hypot( _x, _y, _z ); }

  constexpr REAL theta() const
  { return std::acos( _z / r() ); }

  constexpr REAL phi() const
  { return std::atan2( _y, _x ); }

  constexpr Cartesian operator+( Cartesian const & other ) const
  {
    return { _x + other._x, _y + other._y, _z + other._z };
  }

  constexpr void scaledAdd( REAL const alpha, Cartesian const other )
  {
    _x = _x + alpha * other._x;
    _y = _y + alpha * other._y;
    _z = _z + alpha * other._z;
  }

  REAL _x;
  REAL _y;
  REAL _z;
};


template< typename V1, typename V2 >
auto dot( V1 & v1, V2 const & v2 )
{
  return v1.x() * v2.x() + v1.y() * v2.y() + v1.z() * v2.z();
}

template< typename REAL >
REAL dot( Spherical< REAL > const & v1, Spherical< REAL > const & v2 )
{
  REAL sinTheta1, cosTheta1;
  LvArray::math::sincos( v1.theta(), sinTheta1, cosTheta1 );

  REAL sinTheta2, cosTheta2;
  LvArray::math::sincos( v2.theta(), sinTheta2, cosTheta2 );

  return v1.r() * v2.r() * ( sinTheta1 * sinTheta2 * std::cos( v1.phi() - v2.phi() ) + cosTheta1 * cosTheta2 );
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