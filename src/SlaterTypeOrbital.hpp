#pragma once

#include "AtomicBasis.hpp"

#include "integration/quadrature.hpp"
#include "integration/changeOfVariables.hpp"

namespace tcscf
{

/**
 * 
 */
template< typename REAL >
struct SlaterTypeOrbital
{
  using Real = REAL;
  using Complex = std::complex< Real >;

  static constexpr bool isBasisReal = false;
  static constexpr bool isOrthonormal = false;

  /**
   * 
   */
  SlaterTypeOrbital( Real const alphaP, int const nP, int const lP, int const mP ):
    alpha{ alphaP },
    n{ nP },
    l{ lP },
    m{ mP },
    normalization{ std::pow( 2 * alpha, n ) * std::sqrt( 2 * alpha / factorial( 2 * n ) ) }
  {
    LVARRAY_ERROR_IF_LT( n, 1 );
    LVARRAY_ERROR_IF_LT( n, l );
    LVARRAY_ERROR_IF_LT( l, std::abs( m ) );
  }

  /**
   * 
   */
  Complex operator()( Spherical< Real > const & r ) const
  { return radialComponent( r.r() ) * sphericalHarmonic( l, m, r.theta(), r.phi() ); }

  /**
   * 
   */
  Real radialComponent( Real const r ) const
  { return normalization * std::pow( r, n - 1 ) * std::exp( -alpha * r ); }

  /**
   * 
   */
  Cartesian< Complex > gradient( Spherical< Real > const & r ) const
  {
    LVARRAY_ERROR( "Uh oh" << r.x() );
    return {};
  }

  Real const alpha;
  int const n;
  int const l;
  int const m;
  Real const normalization;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Specializations
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace internal
{

/**
 * 
 */
double overlapNonNormalized( int const sumOfN, double const sumOfAlpha )
{
  return std::tgamma( sumOfN + 1 ) / std::pow( sumOfAlpha, sumOfN + 1 );
}

}

/**
 */
template< typename REAL >
REAL overlap(
  SlaterTypeOrbital< REAL > const & b1,
  SlaterTypeOrbital< REAL > const & b2 )
{
  if( (b1.l != b2.l) || (b1.m != b2.m) )
  {
    return 0;
  }

  return b1.normalization * b2.normalization * internal::overlapNonNormalized( b1.n + b2.n, b1.alpha + b2.alpha );
}

/**
 */
template< typename REAL >
REAL coreMatrixElement(
  int const Z,
  SlaterTypeOrbital< REAL > const & b1,
  SlaterTypeOrbital< REAL > const & b2 )
{
  if( (b1.l != b2.l) || (b1.m != b2.m) )
  {
    return 0;
  }

  int const n = b2.n;
  int const l = b2.l;
  REAL const alpha = b2.alpha;

  int const nSum = b1.n + b2.n;
  REAL const alphaSum = b1.alpha + b2.alpha;

  return 0.5 * b1.normalization * b2.normalization * (
    (l * (l + 1) - n * (n - 1)) * internal::overlapNonNormalized( nSum - 2, alphaSum )
    + 2 * (n * alpha - Z) * internal::overlapNonNormalized( nSum - 1, alphaSum )
    - std::pow( alpha, 2 ) * internal::overlapNonNormalized( nSum, alphaSum ) );
}

} // namespace tcscf