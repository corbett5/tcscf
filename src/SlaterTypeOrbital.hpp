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
  void resetOrbitalExponent( Real const newAlpha )
  {
    alpha = newAlpha;
    normalization = std::pow( 2 * alpha, n ) * std::sqrt( 2 * alpha / factorial( 2 * n ) );
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
    Real const radial = radialComponent( r.r() );
    Complex const Ylm = sphericalHarmonic( l, m, r.theta(), r.phi() );

    Real const dRadialPart = ((n - 1) / r.r() - alpha) * radial;

    Complex const Ylmp1 = (l != m) * std::sqrt((l - m) * (l + m + 1)) * cExp( -r.phi() ) * sphericalHarmonic( l, m + 1, r.theta(), r.phi() );
    
    Complex const rHat = dRadialPart * Ylm;
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

  Real alpha;
  int n;
  int l;
  int m;
  Real normalization;
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

template< typename REAL >
REAL coreMatrixElement(
  integration::QuadratureGrid< REAL > const &,
  int const Z,
  SlaterTypeOrbital< REAL > const & b1,
  SlaterTypeOrbital< REAL > const & b2 )
{
  return coreMatrixElement( Z, b1, b2 );
}

} // namespace tcscf