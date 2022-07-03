#pragma once

#include "AtomicBasis.hpp"

namespace tcscf
{

/**
 * 
 */
template< typename REAL >
struct OchiBasisFunction
{
  using Real = REAL;
  using Complex = std::complex< Real >;

  static constexpr bool isBasisReal = false;
  static constexpr bool isOrthonormal = true;

  /**
   * 
   */
  OchiBasisFunction( Real const alphaP, int const nP, int const lP, int const mP ):
    alpha{ alphaP },
    n{ nP },
    l{ lP },
    m{ mP },
    normalization{ std::pow( 2 * alpha, l + 1.5 ) / std::sqrt( truncatedFactorial( n + 2 * l + 2, n ) ) }
  {}

  /**
   * 
   */
  Complex operator()( Real const r, Real const theta, Real const phi ) const
  { return fnl(r) / r * sphericalHarmonic( l, m, theta, phi ); }

  /**
   * 
   */
  Real fnl(Real const r) const
  {
    return normalization * std::pow(r, l + 1) * boost::math::laguerre(n, 2 * l + 2, 2 * alpha * r) * std::exp(-alpha * r);
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

/**
 * 
 */
template< typename REAL >
std::complex< REAL > coreMatrixElement(
  int const Z,
  OchiBasisFunction< REAL > const & b1,
  OchiBasisFunction< REAL > const & b2 )
{
  LVARRAY_ERROR_IF_GT( std::abs( b1.alpha - b2.alpha ), 0 );

  if(b1.l != b2.l) return 0;
  if(b1.m != b2.m) return 0;

  REAL const alpha = b1.alpha;
  int const n1 = b1.n;
  int const n2 = b2.n;
  int const l = b1.l;

  REAL const integral = integrate0toInf< REAL >(
    [=] ( REAL const r )
    {
      return b1.fnl( r ) * 1 / r * b2.fnl( r );
    }
  );

  REAL const coeff = alpha * (l + 1) * (1 + 2 * std::min( n1, n2 ) / REAL( 2 * l + 3 )) - Z;
  return coeff * integral - (n1 == n2) * std::pow( alpha, 2 ) / 2; 
}

} // namespace tcscf