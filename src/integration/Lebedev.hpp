#pragma once

#include "quadrature.hpp"

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
struct Lebedev
{
  using Real = REAL;
  static constexpr int NDIM = 2;

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
  constexpr GridPoint< Real, 2 > gridPoint( IndexType const i ) const
  {
    return { { m_grid( 0, i ), m_grid( 1, i ) }, m_grid( 2, i ) };    
  }

  /**
   * 
   */
  constexpr Real gridWeight() const
  { return 1; }

  ArrayView2d< Real const > const m_grid;
};

} // namespace tcscf::integration