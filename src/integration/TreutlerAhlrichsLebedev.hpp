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

  QuadratureGrid< Real > const m_radialGrid;
  QuadratureGrid< Real > const m_angularGrid;
};

/**
 *
 */ 
template< typename REAL, typename F >
auto integrate(
  TreutlerAhlrichsLebedev< REAL > const & integrator,
  F && f ) -> decltype( f( CArray< REAL, 3 > {} ) * REAL {} )
{
  ArrayView2d< REAL const > const & radialPoints = integrator.m_radialGrid.points;
  ArrayView1d< REAL const > const & radialWeights = integrator.m_radialGrid.weights;

  ArrayView2d< REAL const > const & angularGrid = integrator.m_angularGrid.points;
  ArrayView1d< REAL const > const & angularWeights = integrator.m_angularGrid.weights;

  using AnswerType = decltype( f( CArray< REAL, 3 > {} ) * REAL {} );
  AnswerType answer = 0;
  for( IndexType i = 0; i < angularGrid.size( 1 ); ++i )
  {
    CArray< REAL, 3 > coords{ 0, angularGrid( 0, i ), angularGrid( 1, i ) };

    AnswerType tmp = 0;
    for( IndexType j = 0; j < radialPoints.size( 1 ); ++j )
    {
      coords[ 0 ] = radialPoints( 0, j );
      tmp = tmp + radialWeights( j ) * f( coords );
    }

    answer = answer + angularWeights( i ) * tmp;
  }

  return answer;
}

} // namespace tcscf::integration
