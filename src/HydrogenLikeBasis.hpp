#pragma once

#include "AtomicBasis.hpp"

namespace tcscf
{

/**
 * 
 */
template< typename REAL >
struct HydrogenLikeBasisFunction
{
  using Real = REAL;
  using Complex = std::complex< Real >;

  static constexpr bool isBasisReal = false;
  static constexpr bool isOrthonormal = true;

  /**
   * 
   */
  HydrogenLikeBasisFunction( int const Z, int const nP, int const lP, int const mP ):
    n{ nP },
    l{ lP },
    m{ mP },
    beta{ Z / Real( n ) },
    normalization{ -std::pow( 2 * beta, l + 1.5 ) / std::sqrt( 2 * n * truncatedFactorial( n + l, n - l - 1 ) ) }
  {
    LVARRAY_ERROR_IF_LT( Z, 0 );
    LVARRAY_ERROR_IF_LT( n, 0 );
    LVARRAY_ERROR_IF_GE( l, n );
    LVARRAY_ERROR_IF_GT( std::abs( m ), l );
  }

  /**
   * 
   */
  Complex operator()( Real const r, Real const theta, Real const phi ) const
  { return radialComponent( r ) * sphericalHarmonic( l, m, theta, phi ); }

  /**
   * 
   */
  Real radialComponent(Real const r) const
  {
    return normalization * std::pow( r, l ) * boost::math::laguerre( n - l - 1 , 2 * l + 1, 2 * beta * r ) * std::exp( -beta * r );
  }

  int const n;
  int const l;
  int const m;
  Real const beta;
  Real const normalization;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Specializations
///////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * 
 */
template< typename REAL >
std::complex< REAL > coreMatrixElement(
  int const Z,
  HydrogenLikeBasisFunction< REAL > const & b1,
  HydrogenLikeBasisFunction< REAL > const & b2 )
{
  if( b1.n != b2.n ) return 0;
  if( b1.l != b2.l ) return 0;
  if( b1.m != b2.m ) return 0;

  return - std::pow( Z, 2 ) / (REAL( 2 ) * std::pow( b1.n, 2 ));
}

} // namespace tcscf