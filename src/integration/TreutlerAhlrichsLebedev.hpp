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
      createGrid( ChebyshevGauss< Real >( nRadialPoints ), changeOfVariables::TreutlerAhlrichs< Real >( epsilon ) ) ),
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

/**
 * 
 */
template< typename T, typename ATOMIC_BASIS, typename F >
std::complex< T > integrateR1R2(
  TreutlerAhlrichsLebedev< T > const & integrator,
  ATOMIC_BASIS const & b1,
  ATOMIC_BASIS const & b2,
  ATOMIC_BASIS const & b3,
  ATOMIC_BASIS const & b4,
  F && f )
{
  using PolicyType = ParallelHost;

  static_assert( std::is_same_v< T, typename ATOMIC_BASIS::Real > );

  ArrayView2d< T const > const & radialGrid = integrator.m_radialGrid;
  ArrayView2d< T const > const & angularGrid = integrator.m_angularGrid;

  RAJA::ReduceSum< Reduce< PolicyType >, std::complex< T > > answer( 0 );
  
  forAll< DefaultPolicy< PolicyType > >( angularGrid.size( 1 ), [=] ( IndexType const a1Idx )
  {
    T const theta1 = angularGrid( 0, a1Idx );
    T const phi1 = angularGrid( 1, a1Idx );
    T const weightA1 = angularGrid( 2, a1Idx );

    for( IndexType r1Idx = 0; r1Idx < radialGrid.size( 1 ); ++r1Idx )
    {
      T const r1 = radialGrid( 0, r1Idx );
      T const weightR1 = radialGrid( 1, r1Idx );

      for( IndexType a2Idx = 0; a2Idx < angularGrid.size( 1 ); ++a2Idx )
      {
        T const theta2 = angularGrid( 0, a2Idx );
        T const phi2 = angularGrid( 1, a2Idx );
        T const weightA2 = angularGrid( 2, a2Idx );

        T r2Sum = 0;
        for( IndexType r2Idx = 0; r2Idx < radialGrid.size( 1 ); ++r2Idx )
        {
          T const r2 = radialGrid( 0, r2Idx );
          T const weightR2 = radialGrid( 1, r2Idx );

          CArray< T, 6 > const R1R2 { r1, theta1, phi1, r2, theta2, phi2 };
          r2Sum = r2Sum + weightR2 * b2.radialComponent( r2 ) * b4.radialComponent( r2 ) * f( R1R2 ) * std::pow( r2, 2 );
        }

        std::complex< T > const a1Value = conj( sphericalHarmonic( b1.l, b1.m, theta1, phi1 ) ) * sphericalHarmonic( b3.l, b3.m, theta1, phi1 );
        T const r1Value = b1.radialComponent( r1 ) * b3.radialComponent( r1 );
        std::complex< T > const a2Value = conj( sphericalHarmonic( b2.l, b2.m, theta2, phi2 ) ) * sphericalHarmonic( b4.l, b4.m, theta2, phi2 );

        answer += weightA2 * weightR1 * weightA1 * r2Sum * a1Value * r1Value * a2Value * std::pow( r1, 2 );
      }
    }
  } );

  return answer.get();
}

template< typename T, typename ATOMIC_BASIS, typename F >
T evaluateR12Integral(
  ArrayView2d< T const > const & radialGrid,
  ArrayView2d< T const > const & angularGrid,
  T const r1,
  CArray< T, 3 > const xyz1,
  ATOMIC_BASIS const & b2,
  ATOMIC_BASIS const & b4,
  F && f )
{
  T r12Integral = 0;
  for( IndexType a12Idx = 0; a12Idx < angularGrid.size( 1 ); ++a12Idx )
  {
    T const theta12 = angularGrid( 0, a12Idx );
    T const phi12 = angularGrid( 1, a12Idx );
    T const weightA12 = angularGrid( 2, a12Idx );

    for( IndexType r12Idx = 0; r12Idx < radialGrid.size( 1 ); ++r12Idx )
    {
      T const r12 = radialGrid( 0, r12Idx );
      T const weightR12 = radialGrid( 1, r12Idx );

      CArray< T, 3 > const xyz12 = sphericalToCartesian( r12, theta12, phi12 );
      
      T const x2 = xyz1[ 0 ] + xyz12[ 0 ];
      T const y2 = xyz1[ 1 ] + xyz12[ 1 ];
      T const z2 = xyz1[ 2 ] + xyz12[ 2 ];

      T const r2 = std::hypot( x2, y2, z2 );
      T const theta2 = std::acos( z2 / (r2 + std::numeric_limits< T >::epsilon()) );
      // Here we set phi2 to be zero because of an assumed symmetry of the orbitals.

      T const weight = weightR12 * weightA12;
      T const jacobian = std::pow( r12, 2 );
      
      T const r2Value = b2.radialComponent( r2 ) * b4.radialComponent( r2 );
      T const a2ValueMagnitude = sphericalHarmonicMagnitude( b2.l, b2.m, theta2 ) * sphericalHarmonicMagnitude( b4.l, b4.m, theta2 );

      r12Integral = r12Integral + weight * r2Value * a2ValueMagnitude * f( r1, r2, r12 ) * jacobian;
    }
  }

  return r12Integral;
}


/**
 *
 */ 
template< typename T, typename ATOMIC_BASIS, typename F >
std::complex< T > integrateR1R12(
  TreutlerAhlrichsLebedev< T > const & integrator,
  ATOMIC_BASIS const & b1,
  ATOMIC_BASIS const & b2,
  ATOMIC_BASIS const & b3,
  ATOMIC_BASIS const & b4,
  F && f )
{
  using PolicyType = ParallelHost;

  static_assert( std::is_same_v< T, typename ATOMIC_BASIS::Real > );

  ArrayView2d< T const > const & radialGrid = integrator.m_radialGrid;
  ArrayView2d< T const > const & angularGrid = integrator.m_angularGrid;

  RAJA::ReduceSum< Reduce< PolicyType >, std::complex< T > > answer( 0 );
  
  forAll< DefaultPolicy< PolicyType > >( angularGrid.size( 1 ), [=] ( IndexType const a1Idx )
  {
    T const theta1 = angularGrid( 0, a1Idx );
    T const phi1 = angularGrid( 1, a1Idx );
    T const weightA1 = angularGrid( 2, a1Idx );

    for( IndexType r1Idx = 0; r1Idx < radialGrid.size( 1 ); ++r1Idx )
    {
      T const r1 = radialGrid( 0, r1Idx );
      T const weightR1 = radialGrid( 1, r1Idx );

      CArray< T, 3 > const xyz1 = sphericalToCartesian( r1, theta1, phi1 );

      T innerIntegral = evaluateR12Integral( radialGrid, angularGrid, r1, xyz1, b2, b4, f );

      T const weight = weightR1 * weightA1;
      T const jacobian = std::pow( r1, 2 );

      T const r1Value = b1.radialComponent( r1 ) * b3.radialComponent( r1 );
      std::complex< T > const a1Value = conj( sphericalHarmonic( b1.l, b1.m, theta1, phi1 ) ) * sphericalHarmonic( b3.l, b3.m, theta1, phi1 );

      answer += innerIntegral * weight * a1Value * r1Value * jacobian;
    }
  } );

  return answer.get();
}

/**
 *
 */
template< typename T, typename ATOMIC_BASIS, typename F >
Array4d< std::complex< T > > integrateAllR1R12(
  TreutlerAhlrichsLebedev< T > const & integrator,
  std::vector< ATOMIC_BASIS > const & basisFunctions,
  F && f )
{
  using PolicyType = ParallelHost;
  
  static_assert( std::is_same_v< T, typename ATOMIC_BASIS::Real > );

  ArrayView2d< T const > const & radialGrid = integrator.m_radialGrid;
  ArrayView2d< T const > const & angularGrid = integrator.m_angularGrid;

  IndexType nBasis = basisFunctions.size();

  Array4d< std::complex< T > > answer( nBasis, nBasis, nBasis, nBasis );

  forAll< DefaultPolicy< PolicyType > >( angularGrid.size( 1 ),
    [&basisFunctions, f, radialGrid, angularGrid, nBasis, answer=answer.toView()] ( IndexType const a1Idx )
    {
      T const theta1 = angularGrid( 0, a1Idx );
      T const phi1 = angularGrid( 1, a1Idx );
      T const weightA1 = angularGrid( 2, a1Idx );

      for( IndexType r1Idx = 0; r1Idx < radialGrid.size( 1 ); ++r1Idx )
      {
        T const r1 = radialGrid( 0, r1Idx );
        T const weightR1 = radialGrid( 1, r1Idx );

        CArray< T, 3 > const xyz1 = sphericalToCartesian( r1, theta1, phi1 );

        for( IndexType b2 = 0; b2 < nBasis; ++b2 )
        {
          for( IndexType b4 = b2; b4 < nBasis; ++b4 )
          {
            T const innerIntegral = evaluateR12Integral(
              radialGrid,
              angularGrid,
              r1,
              xyz1,
              basisFunctions[ b2 ],
              basisFunctions[ b4 ],
              f );
            
            for( IndexType b1 = 0; b1 < nBasis; ++b1 )
            {
              ATOMIC_BASIS const & bf1 = basisFunctions[ b1 ];
              for( IndexType b3 = 0; b3 < nBasis; ++b3 )
              {
                ATOMIC_BASIS const & bf3 = basisFunctions[ b3 ];
                T const weight = weightR1 * weightA1;
                T const jacobian = std::pow( r1, 2 );

                T const r1Value = bf1.radialComponent( r1 ) * bf3.radialComponent( r1 );
                std::complex< T > const a1Value = conj( sphericalHarmonic( bf1.l, bf1.m, theta1, phi1 ) ) * sphericalHarmonic( bf3.l, bf3.m, theta1, phi1 );
                
                std::complex< T > const addition = innerIntegral * weight * a1Value * r1Value * jacobian;
                atomicAdd< PolicyType >( &answer( b1, b2, b3, b4 ), addition );
              }
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