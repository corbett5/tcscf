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
  {
    LVARRAY_ERROR_IF_LT( n, 0 );
    LVARRAY_ERROR_IF_LT( l, m );
  }

  /**
   * 
   */
  Complex operator()( Real const r, Real const theta, Real const phi ) const
  { return radialComponent( r ) * sphericalHarmonic( l, m, theta, phi ); }

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

  /**
   * 
   */
  Spherical< Real > gradient( Spherical< Real > const & r )
  {
    Real const radial = radialComponent( r.r() );
    Real const Ylm = sphericalHarmonic( l, m, r.theta(), r.phi() );

    Real const dLaguerrePart = (n > 0) *
      -2 * alpha * normalization * std::pow( r, l ) * std::exp( -alpha * r.r() ) * assocLaguerre( n - 1, 2 * l + 3, 2 * alpha * r.r() );

    Real const Ylmp1 = (l != m) * std::sqrt((l - m) * (l + m + 1)) * cExp( -r.phi() ) * sphericalHarmonic( l, m + 1, r.theta(), r.phi() );
    
    Real const rHat = ((l / r.r() - alpha) * radial + dLaguerrePart) * Ylm;
    Real const thetaHat = (m * Ylm / std::tan( r.theta() ) + Ylmp1) * radial / r.r();
    Real const phiHat = I< Real > * m * Ylm * radial / (r.r() * std::sin( r.theta() ));

    return { rHat, thetaHat, phiHat };
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