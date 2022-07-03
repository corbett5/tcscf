#pragma once

#include <boost/math/special_functions/spherical_harmonic.hpp>
#include <boost/math/special_functions/laguerre.hpp>

#include <boost/math/quadrature/tanh_sinh.hpp>
#include <boost/math/quadrature/exp_sinh.hpp>

namespace tcscf
{

///////////////////////////////////////////////////////////////////////////////////////////////////
// Math helpers
///////////////////////////////////////////////////////////////////////////////////////////////////

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
std::complex< T > sphericalHarmonic( int const l, int const m, T const theta, T const phi )
{
  return boost::math::spherical_harmonic( l, m, theta, phi );
}

/**
 * 
 */
template< typename T >
T assocLaguerre( int const n, int const k, T const x )
{
  return boost::math::laguerre( n, k, x );
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

      return angleIntegrator.integrate( integrand, 0, 2 * M_PI );
    };

    return angleIntegrator.integrate( integralOverTheta, 0, M_PI );
  };

  return boost::math::quadrature::exp_sinh<REAL>{}.integrate(integralOverR);
}

} // namespace tcscf