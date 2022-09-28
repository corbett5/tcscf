#pragma once

#include "Lebedev.hpp"
#include "ChebyshevGauss.hpp"
#include "changeOfVariables.hpp"

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
template< typename T, typename F >
T evaluateR12Integral(
  ArrayView2d< T const > const & quadratureGrid,
  Spherical< T > const & r1,
  ArrayView2d< T const > const & r2BasisValues,
  IndexType const & b2,
  IndexType const & b4,
  F && f )
{
  IndexType idx = 0;
  return 2 * pi< T > * integrate< 2 >( quadratureGrid, [&] ( CArray< T, 2 > const & r2V )
  {    
    T const r2 = r2V[ 0 ];
    T const theta2 = r2V[ 1 ];

    T const r12 = calculateR12( r1.r(), r1.theta(), r1.phi(), r2, theta2, T {} );
    T const answer = r2BasisValues( idx, b2 ) * r2BasisValues( idx, b4 ) * f( r1.r(), r2, r12 );
    ++idx;
    return answer;
  } );
}

// template< typename T, typename ATOMIC_BASIS, typename F >
// T evaluateR12Integral(
//   ArrayView2d< T const > const & quadratureGrid,
//   T const r1,
//   Cartesian< T > const & r1C,
//   ATOMIC_BASIS const & b2,
//   ATOMIC_BASIS const & b4,
//   F && f )
// {
//   return 2 * pi< T > * integrate< 2 >( quadratureGrid, [&] ( CArray< T, 2 > const & r12V )
//   {
//     Cartesian< T > const r12C = Spherical< T >{ r12V[ 0 ], r12V[ 1 ], T {} };
    
//     Cartesian< T > const r2C = r1C + r12C;

//     T const r2 = r2C.r();
//     T const theta2 = std::acos( r2C.z() / (r2 + std::numeric_limits< T >::epsilon()) );
//     // Here we set phi2 to be zero because of an assumed symmetry of the orbitals.

//     T const r12 = r12V[ 0 ];

//     T const r2Value = b2.radialComponent( r2 ) * b4.radialComponent( r2 );
//     T const a2ValueMagnitude = sphericalHarmonicMagnitude( b2.l, b2.m, theta2 ) * sphericalHarmonicMagnitude( b4.l, b4.m, theta2 );

//     return r2Value * a2ValueMagnitude * f( r1, r1C, r12, r12C, r2 );
//   } );
// }

/**
 * Note: here we are integrating over r1 and r12 using the logarithmic grid, but idk if this is the best in general. It doesn't seem to matter here.
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

  IndexType const nBasis = basisFunctions.size();

  QMCCache< 3, T > const qmcCacheR1( nQMC1 );
  QMCCache< 2, T > const qmcCacheR2( nQMC2 );

  constexpr changeOfVariables::TreutlerAhlrichsM4< T > rChange( 0, 0.9 );
  constexpr changeOfVariables::Linear< T > thetaChange{ pi< T > };
  constexpr changeOfVariables::Linear< T > phiChange{ 2 * pi< T > };

  Array2d< T > const r1Grid = integration::createGrid( qmcCacheR1,
    changeOfVariables::createMultiple< T, 3 >( rChange, thetaChange, phiChange ) );

  Array2d< std::complex< T > > const r1BasisValues( r1Grid.size(), nBasis );
  forAll< DefaultPolicy< PolicyType > >( r1Grid.size( 1 ),
    [&basisFunctions, nBasis, r1Grid=r1Grid.toView(), r1BasisValues=r1BasisValues.toView()] ( IndexType const idx )
    {
      T const r = r1Grid( 0, idx );
      T const theta = r1Grid( 1, idx );
      T const phi = r1Grid( 2, idx );

      T const jacobian = std::pow( r, 2 ) * std::sin( theta );
      r1Grid( 3, idx ) *= jacobian;

      for( IndexType b = 0; b < nBasis; ++b )
      {
        r1BasisValues( idx, b ) = basisFunctions[ b ]( r, theta, phi );
      }
    }
  );

  Array2d< T > const r2Grid = integration::createGrid( qmcCacheR2,
    changeOfVariables::createMultiple< T, 2 >( rChange, thetaChange ) );
  
  Array2d< T > const r2BasisValues( r2Grid.size(), nBasis );
  forAll< DefaultPolicy< PolicyType > >( r2Grid.size( 1 ),
    [&basisFunctions, nBasis, r2Grid=r2Grid.toView(), r2BasisValues=r2BasisValues.toView()] ( IndexType const idx )
    {
      T const r = r2Grid( 0, idx );
      T const theta = r2Grid( 1, idx );

      T const jacobian = std::pow( r, 2 ) * std::sin( theta );
      r2Grid( 2, idx ) *= jacobian;

      for( IndexType b = 0; b < nBasis; ++b )
      {
        // Take the real component because phi2 = 0 so the imaginary component is zero anyways.
        r2BasisValues( idx, b ) = std::real( basisFunctions[ b ]( r, theta, T {} ) );
      }
    }
  );

  Array4d< std::complex< T > > const answer( nBasis, nBasis, nBasis, nBasis );

  forAll< DefaultPolicy< PolicyType > >( r1Grid.size( 1 ),
    [f, r1Grid=r1Grid.toViewConst(), r2Grid=r2Grid.toViewConst(),
     nBasis, r1BasisValues=r1BasisValues.toViewConst(), r2BasisValues=r2BasisValues.toViewConst(), answer=answer.toView()] ( IndexType const idx )
    {
      Spherical< T > const r1S { r1Grid( 0, idx ), r1Grid( 1, idx ), r1Grid( 2, idx ) };
      T const r1Weight = r1Grid( 3, idx );

      // Cartesian< T > const r1C = r1S;

      for( IndexType b2 = 0; b2 < nBasis; ++b2 )
      {
        // for( IndexType b4 = b2; b4 < nBasis; ++b4 )
        for( IndexType b4 = 0; b4 < nBasis; ++b4 )
        {
          T const innerIntegral = evaluateR12Integral(
            r2Grid,
            r1S,
            r2BasisValues,
            b2,
            b4,
            f );

          // T const innerIntegral = evaluateR12Integral(
          //   r2Grid,
          //   r1S.r(),
          //   r1C,
          //   basisFunctions[ b2 ],
          //   basisFunctions[ b4 ],
          //   f );
          
          for( IndexType b1 = 0; b1 < nBasis; ++b1 )
          {
            for( IndexType b3 = 0; b3 < nBasis; ++b3 )
            {
              std::complex< T > const addition = innerIntegral * r1Weight * r1BasisValues( idx, b1 ) * r1BasisValues( idx, b3 );
              atomicAdd< PolicyType >( &answer( b1, b2, b3, b4 ), addition );
            }
          }
        }
      }
    }
  );

  // for( IndexType b1 = 0; b1 < nBasis; ++b1 )
  // {
  //   for( IndexType b2 = 0; b2 < nBasis; ++b2 )
  //   {
  //     for( IndexType b3 = 0; b3 < nBasis; ++b3 )
  //     {
  //       for( IndexType b4 = b2 + 1; b4 < nBasis; ++b4 )
  //       {
  //         answer( b1, b4, b3, b2 ) = answer( b1, b2, b3, b4 );
  //       }
  //     }
  //   }
  // }

  return answer;
}

} // namespace tcscf::integration