#pragma once

#include "../mathFunctions.hpp"

namespace tcscf::integration::changeOfVariables
{

/**
 * 
 */
template< typename REAL >
struct None
{
  using Real = REAL;

  /**
   * 
   */
  template< camp::idx_t NDIM >
  constexpr CArray< Real, NDIM > const & u( CArray< Real, NDIM > const & x ) const
  {
    return x;
  }

  /**
   * 
   */
  template< camp::idx_t NDIM >
  constexpr CArray< Real, NDIM > u( CArray< Real, NDIM > && x ) const
  {
    return x;
  }

  /**
   * 
   */
  template< camp::idx_t NDIM >
  constexpr Real jacobian( CArray< Real, NDIM > const & LVARRAY_UNUSED_ARG( x ) ) const
  {
    return 1;
  }

};

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
  constexpr CArray< Real, 1 > u( CArray< Real, 1 > const & x ) const
  {
    return { m_epsilonOverLog2 * std::pow( x[ 0 ] + 1, m_alpha ) * std::log( 2 / (1 - x[ 0 ]) ) };
  }

  /**
   * 
   */
  constexpr Real jacobian( CArray< Real, 1 > const & x ) const
  {
    Real const logPart = std::log( 2 / (1 - x[ 0 ]) );
    Real const xPlus1ToTheAlphaMinus1 = std::pow( x[ 0 ] + 1, m_alpha - 1 );
    Real const xPlus1ToTheAlpha = xPlus1ToTheAlphaMinus1 * (x[ 0 ] + 1);

    return m_epsilonOverLog2 * ( m_alpha * xPlus1ToTheAlphaMinus1 * logPart - xPlus1ToTheAlpha / (x[ 0 ] - 1) );
  }

  Real const m_epsilonOverLog2;
  static constexpr Real m_alpha{ 0.6 };
};

} // tcscf::integration::changeOfVariables