#pragma once

#include "Lebedev.hpp"
#include "ChebyshevGauss.hpp"
#include "changeOfVariables.hpp"

// TODO move this out as well
#include "qmcWrapper.hpp"

#include "../RAJAInterface.hpp"

namespace tcscf::integration
{

/**
 * 
 */
template< typename REAL >
struct TreutlerAhlrichsLebedev
{
  using Real = REAL;
  static constexpr int NDIM = 3;

  /**
   * 
   */
  TreutlerAhlrichsLebedev( Real const epsilon, int const nRadialPoints, int const angularOrder ):
    m_radialGrid(
      createGrid( ChebyshevGauss< Real >( nRadialPoints ), changeOfVariables::TreutlerAhlrichsM4< Real >( 1, epsilon ) ) ),
    m_angularGrid( createGrid( Lebedev< Real >( angularOrder  ) ) )
  {}

  /**
   * 
   */
  constexpr IndexType numPoints() const
  { return m_radialGrid.size( 1 ) * m_angularGrid.size( 1 ); }

  Array2d< Real > const m_radialGrid;
  Array2d< Real > const m_angularGrid;
};

/**
 *
 */ 
template< typename REAL, typename F >
auto integrate(
  TreutlerAhlrichsLebedev< REAL > const & integrator,
  F && f ) -> decltype( f( CArray< REAL, 3 > {} ) * REAL {} )
{
  ArrayView2d< REAL const > const & radialGrid = integrator.m_radialGrid;
  ArrayView2d< REAL const > const & angularGrid = integrator.m_angularGrid;

  using AnswerType = decltype( f( CArray< REAL, 3 > {} ) * REAL {} );
  AnswerType answer = 0;
  for( IndexType i = 0; i < angularGrid.size( 1 ); ++i )
  {
    CArray< REAL, 3 > coords{ 0, angularGrid( 0, i ), angularGrid( 1, i ) };

    AnswerType tmp = 0;
    for( IndexType j = 0; j < radialGrid.size( 1 ); ++j )
    {
      coords[ 0 ] = radialGrid( 0, j );
      REAL const weight = radialGrid( 1, j );
      tmp = tmp + weight * f( coords );
    }

    answer = answer + tmp * angularGrid( 2, i );
  }

  return answer;
}

// TODO(corbet5): move this stuff out of this file
/**
 *
 */
template< typename T, typename ATOMIC_BASIS, typename F >
T evaluateR12Integral(
  ArrayView2d< T const > const & quadratureGrid,
  T const r1,
  Cartesian< T > const & r1C,
  ATOMIC_BASIS const & b2,
  ATOMIC_BASIS const & b4,
  F && f )
{
  return 2 * pi< T > * integrate< 2 >( quadratureGrid, [&] ( CArray< T, 2 > const & r12V )
  {
    Cartesian< T > const r12C = Spherical< T >{ r12V[ 0 ], r12V[ 1 ], T {} };
    
    Cartesian< T > const r2C = r1C - r12C;

    T const r2 = r2C.r();
    T const theta2 = std::acos( r2C.z() / (r2 + std::numeric_limits< T >::epsilon()) );
    // Here we set phi2 to be zero because of an assumed symmetry of the orbitals.

    T const r12 = r12V[ 0 ];

    T const r2Value = b2.radialComponent( r2 ) * b4.radialComponent( r2 );
    T const a2ValueMagnitude = sphericalHarmonicMagnitude( b2.l, b2.m, theta2 ) * sphericalHarmonicMagnitude( b4.l, b4.m, theta2 );

    return r2Value * a2ValueMagnitude * f( r1, r1C, r12, r12C, r2 );
  } );
}

/**
 * Note: here we are integrating over r1 and r2 using the logarithmic grid, but idk if this is the best in general. It doesn't seem to matter here.
 */
template< typename T, typename ATOMIC_BASIS, typename F >
Array4d< std::complex< T > > integrateAllR1R12(
  IndexType const nQMC1,
  IndexType const nQMC2,
  std::vector< ATOMIC_BASIS > const & basisFunctions,
  F && f )
{
  using PolicyType = ParallelHost;
  
  static_assert( std::is_same_v< T, typename ATOMIC_BASIS::Real > );

  QMCCache< 3, T > const qmcCacheR1( nQMC1 );
  QMCCache< 2, T > const qmcCacheR2( nQMC2 );

  constexpr changeOfVariables::TreutlerAhlrichsM4< T > rChange( 0, 0.9 );
  constexpr changeOfVariables::Linear< T > thetaChange{ pi< T > };
  constexpr changeOfVariables::Linear< T > phiChange{ 2 * pi< T > };

  Array2d< T > const r1Grid = integration::createGrid( qmcCacheR1,
    changeOfVariables::createMultiple< T, 3 >( rChange, thetaChange, phiChange ) );

  for( IndexType i = 0; i < r1Grid.size( 1 ); ++i )
  {
    T const jacobian = std::pow( r1Grid( 0, i ), 2 ) * std::sin( r1Grid( 1, i ) );
    r1Grid( 3, i ) *= jacobian;
  }

  Array2d< T > const r2Grid = integration::createGrid( qmcCacheR2,
    changeOfVariables::createMultiple< T, 2 >( rChange, thetaChange ) );
  
  for( IndexType i = 0; i < r2Grid.size( 1 ); ++i )
  {
    T const jacobian = std::pow( r2Grid( 0, i ), 2 ) * std::sin( r2Grid( 1, i ) );
    r2Grid( 2, i ) *= jacobian;
  }

  IndexType nBasis = basisFunctions.size();

  Array4d< std::complex< T > > answer( nBasis, nBasis, nBasis, nBasis );

  forAll< DefaultPolicy< PolicyType > >( r1Grid.size( 1 ),
    [&basisFunctions, f, r1Grid=r1Grid.toViewConst(), r2Grid=r2Grid.toViewConst(), nBasis, answer=answer.toView()] ( IndexType const idx )
    {
      Spherical< T > const r1S { r1Grid( 0, idx ), r1Grid( 1, idx ), r1Grid( 2, idx ) };
      T const r1Weight = r1Grid( 3, idx );

      Cartesian< T > const r1C = r1S;

      for( IndexType b2 = 0; b2 < nBasis; ++b2 )
      {
        for( IndexType b4 = b2; b4 < nBasis; ++b4 )
        // for( IndexType b4 = 0; b4 < nBasis; ++b4 )
        {
          T const innerIntegral = evaluateR12Integral(
            r2Grid,
            r1S.r(),
            r1C,
            basisFunctions[ b2 ],
            basisFunctions[ b4 ],
            f );
          
          for( IndexType b1 = 0; b1 < nBasis; ++b1 )
          {
            ATOMIC_BASIS const & bf1 = basisFunctions[ b1 ];
            for( IndexType b3 = 0; b3 < nBasis; ++b3 )
            {
              ATOMIC_BASIS const & bf3 = basisFunctions[ b3 ];

              T const r1Value = bf1.radialComponent( r1S.r() ) * bf3.radialComponent( r1S.r() );
              std::complex< T > const a1Value = conj( sphericalHarmonic( bf1.l, bf1.m, r1S.theta(), r1S.phi() ) ) * sphericalHarmonic( bf3.l, bf3.m, r1S.theta(), r1S.phi() );
              
              std::complex< T > const addition = innerIntegral * r1Weight * a1Value * r1Value;
              atomicAdd< PolicyType >( &answer( b1, b2, b3, b4 ), addition );
            }
          }
        }
      }
    }
  );

  for( IndexType b1 = 0; b1 < nBasis; ++b1 )
  {
    for( IndexType b2 = 0; b2 < nBasis; ++b2 )
    {
      for( IndexType b3 = 0; b3 < nBasis; ++b3 )
      {
        for( IndexType b4 = b2 + 1; b4 < nBasis; ++b4 )
        {
          answer( b1, b4, b3, b2 ) = answer( b1, b2, b3, b4 );
        }
      }
    }
  }

  return answer;
}

} // namespace tcscf::integration