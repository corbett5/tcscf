#pragma once

#include "quadrature.hpp"


namespace tcscf::integration
{

template< typename REAL >
struct ChebyshevGauss
{
  using Real = REAL;
  static constexpr int NDIM = 1;

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
  constexpr GridPoint< Real, 1 > gridPoint( IndexType const i ) const
  {
    Real const theta = (i + 1) * m_freq;
    Real weight, x;
    LvArray::math::sincos( theta, weight, x );

    return { { x }, weight };
  }

  /**
   * 
   */
  constexpr Real gridWeight() const
  { return m_freq; }

  IndexType const m_n;
  Real const m_freq;
};

} // namespace tcscf::integration
