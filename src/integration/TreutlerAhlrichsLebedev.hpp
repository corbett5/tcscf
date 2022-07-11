#pragma once

#include "Lebedev.hpp"
#include "ChebyshevGauss.hpp"
#include "changeOfVariables.hpp"

namespace tcscf::integration
{

namespace internal
{

template< typename REAL >
ArrayView2d< REAL const > getLebedevGrid( int const order );

} // namespace internal

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
    m_radialGrid2(
      createGrid( ChebyshevGauss< Real >( nRadialPoints ), changeOfVariables::TreutlerAhlrichs< Real >( epsilon ) ) ),
    m_angularGrid( createGrid( Lebedev< Real >( angularOrder  ) ) )
  {}

  /**
   * 
   */
  TreutlerAhlrichsLebedev( Real const epsilon, int const nRadialPoints, int const nRadialPoints2, int const angularOrder ):
    m_radialGrid(
      createGrid( ChebyshevGauss< Real >( nRadialPoints ), changeOfVariables::TreutlerAhlrichs< Real >( epsilon ) ) ),
    m_radialGrid2(
      createGrid( ChebyshevGauss< Real >( nRadialPoints2 ), changeOfVariables::TreutlerAhlrichs< Real >( epsilon ) ) ),
    m_angularGrid( createGrid( Lebedev< Real >( angularOrder  ) ) )
  {}

  /**
   * 
   */
  constexpr IndexType numPoints() const
  { return m_radialGrid.size( 1 ) * m_angularGrid.size( 1 ); }

  Array2d< Real > const m_radialGrid;
  Array2d< Real > const m_radialGrid2;
  Array2d< Real > const m_angularGrid;
};

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

} // namespace tcscf::integration