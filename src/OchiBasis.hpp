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
  Real fnl( Real const r ) const
  { return radialComponent( r ) * r; }

  /**
   * 
   */
  Real radialComponent( Real const r ) const
  { return normalization * std::pow( r, l ) * assocLaguerre( n, 2 * l + 2, 2 * alpha * r ) * std::exp(-alpha * r); }

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
REAL coreMatrixElement(
  ArrayView2d< REAL const > const & quadratureGrid,
  int const Z,
  OchiBasisFunction< REAL > const & b1,
  OchiBasisFunction< REAL > const & b2 )
{
  TCSCF_MARK_FUNCTION;

  LVARRAY_ERROR_IF_GT( std::abs( b1.alpha - b2.alpha ), 0 );

  if(b1.l != b2.l) return 0;
  if(b1.m != b2.m) return 0;

  REAL const alpha = b1.alpha;
  int const n1 = b1.n;
  int const n2 = b2.n;
  int const l = b1.l;

  REAL const integral = integration::integrate< 1 >( quadratureGrid,
    [=] ( CArray< REAL, 1 > const r )
    {
      return b1.fnl( r[ 0 ] ) * 1 / r[ 0 ] * b2.fnl( r[ 0 ] );
    }
  );

  REAL const coeff = alpha * (l + 1) * (1 + 2 * std::min( n1, n2 ) / REAL( 2 * l + 3 )) - Z;
  return coeff * integral - (n1 == n2) * std::pow( alpha, 2 ) / 2; 
}

} // namespace tcscf