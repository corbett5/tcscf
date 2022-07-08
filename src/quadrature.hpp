#pragma once

#include "mathFunctions.hpp"
#include "LvArrayInterface.hpp"

// TODO split this into multiple files and directories

namespace tcscf::changeOfVariables
{

/**
 * 
 */
template< typename REAL >
struct TreutlerAhlrichs
{
  using Real = REAL;

  /**
   * 
   */
  TreutlerAhlrichs( Real const epsilon ):
    m_epsilonOverLog2{ epsilon / ln2< Real > }
  {}

  /**
   * 
   */
  Real const du_dx( Real const x ) const
  {
    Real const logPart = std::log( 2 / (1 - x) );
    Real const xPlus1ToTheAlphaMinus1 = std::pow( x + 1, m_alpha - 1 );
    Real const xPlus1ToTheAlpha = xPlus1ToTheAlphaMinus1 * (x + 1);

    return m_epsilonOverLog2 * ( m_alpha * xPlus1ToTheAlphaMinus1 * logPart - xPlus1ToTheAlpha / (x - 1) );
  }
  
  /**
   * 
   */
  Real const u( Real const x ) const
  {
    return m_epsilonOverLog2 * std::pow( x + 1, m_alpha ) * std::log( 2 / (1 - x) );
  }

  Real const m_epsilonOverLog2;
  static constexpr Real m_alpha{ 0.6 };
};

} // tcscf::changeOfVariables





namespace tcscf::quadrature
{

namespace internal
{

template< typename REAL >
Array2d< REAL > getLebedevGrid( int const order );

} // namespace internal

/**
 * 
 */
template< typename INTEGRATOR, typename CHANGE_OF_VARIABLES, typename F >
typename INTEGRATOR::Real integrate(
  INTEGRATOR const & integrator,
  CHANGE_OF_VARIABLES const & changeOfVariables,
  F && f )
{
  using Real = typename INTEGRATOR::Real;
  static_assert( std::is_same_v< Real, typename CHANGE_OF_VARIABLES::Real > );

  return integrator.integrate(
    [=] ( auto const x )
    {
      return f( changeOfVariables.u( x ) ) * changeOfVariables.du_dx( x );
    }
  );
}

/**
 * 
 */
template< typename INTEGRATOR, typename CHANGE_OF_VARIABLES >
Array2d< typename INTEGRATOR::Real > createGrid(
  INTEGRATOR const & integrator,
  CHANGE_OF_VARIABLES const & changeOfVariables )
{
  using Real = typename INTEGRATOR::Real;
  static_assert( std::is_same_v< Real, typename CHANGE_OF_VARIABLES::Real > );

  Array2d< Real > grid( 2, integrator.numPoints() );

  for( IndexType i = 0; i < grid.size( 1 ); ++i )
  {
    CArray< Real, 2 > const xAndWeight = integrator.gridPoint( i );
    grid( 0, i ) = changeOfVariables.u( xAndWeight[ 0 ] );
    grid( 1, i ) = xAndWeight[ 1 ] * changeOfVariables.du_dx( xAndWeight[ 0 ] );
  }

  return grid;
}

/**
 * 
 */
template< typename REAL, typename F >
auto integrate( ArrayView2d< REAL const > const & grid, F && f ) -> decltype( f( REAL {} ) * REAL {} )
{
  LVARRAY_ERROR_IF_NE( grid.size( 0 ), 2 );

  decltype( f( REAL {} ) * REAL {} ) answer = 0;
  for( IndexType i = 0; i < grid.size( 1 ); ++i )
  {
    REAL const x = grid( 0, i );
    REAL const weight = grid( 1, i );
    answer += f( x ) * weight;
  }

  return answer;
}

/**
 * 
 */
template< typename REAL, typename F >
auto integrate2d( ArrayView2d< REAL const > const & grid, F && f ) -> decltype( f( REAL {}, REAL {} ) * REAL {} )
{
  LVARRAY_ERROR_IF_NE( grid.size( 0 ), 3 );

  decltype( f( REAL {}, REAL {} ) * REAL {} ) answer = 0;
  for( IndexType i = 0; i < grid.size( 1 ); ++i )
  {
    REAL const x1 = grid( 0, i );
    REAL const x2 = grid( 1, i );
    REAL const weight = grid( 2, i );
    answer += f( x1, x2 ) * weight;
  }

  return answer;
}




/**
 * Integrate (1 - x^2) f(x) from -1 to +1
 */
template< typename REAL >
struct ChebyshevGauss
{
  using Real = REAL;

  /**
   * 
   */
  ChebyshevGauss( IndexType const n ):
    m_n{ n },
    m_freq{ pi< Real > / (n + 1) }
  {}

  /**
   * 
   */
  constexpr IndexType numPoints() const
  { return m_n; }

  /**
   * 
   */
  template< typename F >
  constexpr Real integrate( F && f ) const
  {
    Real answer = 0;
    for( IndexType i = 0; i < m_n; ++i )
    {
      Real const theta = (i + 1) * m_freq;
      Real weight, x;
      LvArray::math::sincos( theta, weight, x );

      answer = answer + weight * f( x );
    }
  
    return m_freq * answer;
  }

  /**
   * 
   */
  constexpr CArray< Real, 2 > gridPoint( IndexType const i ) const
  {
    Real const theta = (i + 1) * m_freq;
    Real weight, x;
    LvArray::math::sincos( theta, weight, x );

    return { x, m_freq * weight };
  }

  IndexType const m_n;
  Real const m_freq;
};






/**
 * 
 */
template< typename REAL >
struct Lebedev
{
  using Real = REAL;

  /**
   * 
   */
  Lebedev( int const order ):
    m_grid( internal::getLebedevGrid< Real >( order ) )
  {}

  /**
   * 
   */
  constexpr IndexType numPoints() const
  { return m_grid.size( 1 ); }

  /**
   * 
   */
  template< typename F >
  constexpr auto integrate( F && f ) const -> decltype( f( Real {}, Real {} ) * Real {} )
  {
    return integrate2d( m_grid.toViewConst(), std::forward< F >( f ) );
  }

  Array2d< Real > const m_grid;
};

} // namespace tcscf::quadrature