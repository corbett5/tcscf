#pragma once

// #define STD_SPECIAL_FUNCTIONS

#if defined( STD_SPECIAL_FUNCTIONS )
  #include <cmath>
#else
  #include <boost/math/special_functions/spherical_harmonic.hpp>
  #include <boost/math/special_functions/laguerre.hpp>
#endif

#include <boost/math/quadrature/tanh_sinh.hpp>
#include <boost/math/quadrature/exp_sinh.hpp>

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
#if defined( STD_SPECIAL_FUNCTIONS )
  return sphericalHarmonicMagnitude( l, m, theta ) * std::exp( I< T > * m * phi );
#else
  return boost::math::spherical_harmonic( l, m, theta, phi );
#endif
}

/**
 * 
 */
template< typename T >
T assocLaguerre( int const n, int const k, T const x )
{
#if defined( STD_SPECIAL_FUNCTIONS )
  return std::assoc_laguerre( n, k, x );
#else
  return boost::math::laguerre( n, k, x );
#endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Boost quadrature wrappers
///////////////////////////////////////////////////////////////////////////////////////////////////

/**
 *
 */
template< typename REAL, typename F >
auto integrate0toInf( F const & f )
{
  return boost::math::quadrature::exp_sinh< REAL >{}.integrate( f );
}

/**
 * 
 */
template< typename REAL, typename F >
auto sphericalCoordinates3DIntegral( F const & f )
{
  boost::math::quadrature::tanh_sinh< REAL > angleIntegrator;

  auto integralOverR = [&] ( REAL const r )
  {
    auto integralOverTheta = [&] ( REAL const theta )
    {
      auto integrand = [&] ( REAL const phi )
      {
        return f( r, theta, phi ) * std::pow( r, 2 ) * std::sin( theta );
      };

      return angleIntegrator.integrate( integrand, 0, 2 * pi< REAL > );
    };

    return angleIntegrator.integrate( integralOverTheta, 0, pi< REAL > );
  };

  return boost::math::quadrature::exp_sinh<REAL>{}.integrate( integralOverR );
}

} // namespace tcscf