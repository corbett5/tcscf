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
  Complex operator()( Spherical< Real > const & r ) const
  { return radialComponent( r.r() ) * sphericalHarmonic( l, m, r.theta(), r.phi() ); }

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
  Cartesian< Complex > gradient( Spherical< Real > const & r ) const
  {
    Real const radial = radialComponent( r.r() );
    Complex const Ylm = sphericalHarmonic( l, m, r.theta(), r.phi() );

    Real const dLaguerrePart = (n > 0) *
      -2 * alpha * normalization * std::pow( r.r(), l ) * std::exp( -alpha * r.r() ) * assocLaguerre( n - 1, 2 * l + 3, 2 * alpha * r.r() );

    Complex const Ylmp1 = (l != m) * std::sqrt((l - m) * (l + m + 1)) * cExp( -r.phi() ) * sphericalHarmonic( l, m + 1, r.theta(), r.phi() );
    
    Complex const rHat = ((l / r.r() - alpha) * radial + dLaguerrePart) * Ylm;
    Complex const thetaHat = (m * Ylm / std::tan( r.theta() ) + Ylmp1) * radial / r.r();
    Complex const phiHat = I< Real > * m * Ylm * radial / (r.r() * std::sin( r.theta() ));

    Cartesian< Complex > answer {};
    
    Cartesian< Real > rHatC
      { std::cos( r.phi() ) * std::sin( r.theta() ),
        std::sin( r.phi() ) * std::sin( r.theta() ),
        std::cos( r.theta() ) };

    answer._x += rHat * rHatC.x();
    answer._y += rHat * rHatC.y();
    answer._z += rHat * rHatC.z();

    Cartesian< Real > thetaHatC
    { std::cos( r.phi() ) * std::cos( r.theta() ),
      std::sin( r.phi() ) * std::cos( r.theta() ),
      -std::sin( r.theta() ) };

    answer._x += thetaHat * thetaHatC.x();
    answer._y += thetaHat * thetaHatC.y();
    answer._z += thetaHat * thetaHatC.z();

    Cartesian< Real > phiHatC
    { -std::sin( r.phi() ),
      std::cos( r.phi() ),
      0 };

    answer._x += phiHat * phiHatC.x();
    answer._y += phiHat * phiHatC.y();
    answer._z += phiHat * phiHatC.z();

    return answer;
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
  integration::QuadratureGrid< REAL > const & quadratureGrid,
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