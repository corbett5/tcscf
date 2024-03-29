#pragma once

#include "../mathFunctions.hpp"
#include "../LvArrayInterface.hpp"

#include "changeOfVariables.hpp"


namespace tcscf::integration
{

/**
 * 
 */
template< typename REAL, int NDIM >
struct GridPoint
{
  CArray< REAL, NDIM > x;
  REAL weight;
};

/**
 * 
 */
template< typename REAL >
struct QuadratureGrid
{
  QuadratureGrid( int const ndim, int const npoints ):
    points( ndim, npoints ),
    weights( npoints )
  {}

  Array2d< REAL > points;
  Array1d< REAL > weights;
};

/**
 * 
 */
template< typename INTEGRATOR, typename CHANGE_OF_VARIABLES, typename F >
auto integrate(
  INTEGRATOR const & integrator,
  CHANGE_OF_VARIABLES const & changeOfVariables,
  F && f ) -> decltype( f( integrator.gridPoint( 0 ).x ) * typename INTEGRATOR::Real {} )
{
  using Real = typename INTEGRATOR::Real;
  static_assert( std::is_same_v< Real, typename CHANGE_OF_VARIABLES::Real > );

  using AnswerType = decltype( f( integrator.gridPoint( 0 ).x ) * typename INTEGRATOR::Real {} );
  AnswerType answer = 0;

  for( IndexType i = 0; i < integrator.numPoints(); ++i )
  {
    GridPoint const point = integrator.gridPoint( i );
    answer = answer + f( changeOfVariables.u( point.x  ) ) * changeOfVariables.jacobian( point.x ) * point.weight;
  }

  return answer * integrator.gridWeight();
}

/**
 * 
 */
template< typename INTEGRATOR, typename F >
auto integrate(
  INTEGRATOR const & integrator,
  F && f ) -> decltype( f( integrator.gridPoint( 0 ).x ) * typename INTEGRATOR::Real {} )
{
  return integrate( integrator, changeOfVariables::None< typename INTEGRATOR::Real, INTEGRATOR::NDIM > {}, std::forward< F >( f ) );
}

/**
 * 
 */
template< typename INTEGRATOR, typename CHANGE_OF_VARIABLES >
QuadratureGrid< typename INTEGRATOR::Real > createGrid(
  INTEGRATOR const & integrator,
  CHANGE_OF_VARIABLES const & changeOfVariables )
{
  using Real = typename INTEGRATOR::Real;
  constexpr int NDIM = INTEGRATOR::NDIM;

  static_assert( std::is_same_v< Real, typename CHANGE_OF_VARIABLES::Real > );

  QuadratureGrid< Real > grid( NDIM, integrator.numPoints() );

  for( IndexType i = 0; i < integrator.numPoints(); ++i )
  {
    GridPoint< Real, NDIM > const gridPoint = integrator.gridPoint( i );

    CArray< Real, NDIM > const & u = changeOfVariables.u( gridPoint.x );

    for( int dim = 0; dim < NDIM; ++dim )
    {
      grid.points( dim, i ) = u[ dim ];
    }

    grid.weights( i ) = gridPoint.weight * changeOfVariables.jacobian( gridPoint.x ) * integrator.gridWeight();
  }

  return grid;
}

/**
 * 
 */
template< typename INTEGRATOR >
QuadratureGrid< typename INTEGRATOR::Real > createGrid(
  INTEGRATOR const & integrator )
{
  return createGrid( integrator, changeOfVariables::None< typename INTEGRATOR::Real, INTEGRATOR::NDIM > {} );
}

/**
 * 
 */
template< int NDIM, typename REAL, typename F >
auto integrate(
  QuadratureGrid< REAL > const & grid,
  F && f ) -> decltype( f( CArray< REAL, NDIM > {} ) * REAL {} )
{
  LVARRAY_ERROR_IF_NE( grid.points.size( 0 ), NDIM );
  
  decltype( f( CArray< REAL, NDIM > {} ) * REAL {} ) answer = 0;
  for( IndexType i = 0; i < grid.points.size( 1 ); ++i )
  {
    CArray< REAL, NDIM > x;

    for( int dim = 0; dim < NDIM; ++dim )
    {
      x[ dim ] = grid.points( dim, i );
    }

    REAL const weight = grid.weights( i );
    answer += f( x ) * weight;
  }

  return answer;
}

} // namespace tcscf::integration
