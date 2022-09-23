#pragma once

#include "../mathFunctions.hpp"

namespace tcscf::integration::changeOfVariables
{

/**
 * 
 */
template< typename REAL, int NDIM_T >
struct None
{
  using Real = REAL;
  static constexpr int NDIM = NDIM_T;

  /**
   * 
   */
  constexpr CArray< Real, NDIM > const & u( CArray< Real, NDIM > const & x ) const
  {
    return x;
  }

  /**
   * 
   */
  constexpr CArray< Real, NDIM > u( CArray< Real, NDIM > && x ) const
  {
    return x;
  }

  /**
   * 
   */
  constexpr Real jacobian( CArray< Real, NDIM > const & LVARRAY_UNUSED_ARG( x ) ) const
  {
    return 1;
  }
};

/**
 * 
 */
template< typename REAL >
struct Linear
{
  using Real = REAL;
  static constexpr int NDIM = 1;

  /**
   * 
   */
  constexpr CArray< Real, 1 > u( CArray< Real, 1 > const & x ) const
  {
    return { scale * x[ 0 ] };
  }

  /**
   * 
   */
  constexpr Real jacobian( CArray< Real, 1 > const & LVARRAY_UNUSED_ARG( x ) ) const
  {
    return scale;
  }

  REAL const scale;
};

/**
 * 
 */
template< typename REAL >
struct XOver1mX
{
  using Real = REAL;
  static constexpr int NDIM = 1;

  /**
   * 
   */
  constexpr CArray< Real, 1 > u( CArray< Real, 1 > const & x ) const
  {
    return { x[ 0 ] / (1 - x[ 0 ]) };
  }

  /**
   * 
   */
  constexpr Real jacobian( CArray< Real, 1 > const & x ) const
  {
    return std::pow( 1 - x[ 0 ], -2 );
  }
};

/**
 * 
 */
template< typename REAL >
struct TreutlerAhlrichsM4
{
  using Real = REAL;
  static constexpr int NDIM = 1;

  /**
   * 
   */
  constexpr TreutlerAhlrichsM4( Real const a, Real const epsilon ):
    _a{ a },
    _epsilonOverLog2{ epsilon / ln2< Real > }
  {}

  /**
   * 
   */
  constexpr CArray< Real, 1 > u( CArray< Real, 1 > const & x ) const
  {
    return { _epsilonOverLog2 * std::pow( x[ 0 ] + _a, _alpha ) * std::log( (1 + _a) / (1 - x[ 0 ]) ) };
  }

  /**
   * 
   */
  constexpr Real jacobian( CArray< Real, 1 > const & x ) const
  {
    Real const logPart = std::log( (1 + _a) / (1 - x[ 0 ]) );
    Real const xPlusAToTheAlphaMinus1 = std::pow( x[ 0 ] + _a, _alpha - 1 );
    Real const xPlusAToTheAlpha = xPlusAToTheAlphaMinus1 * (x[ 0 ] + _a);

    return _epsilonOverLog2 * ( _alpha * xPlusAToTheAlphaMinus1 * logPart + xPlusAToTheAlpha / (1 - x[ 0 ]) );
  }

  Real const _a;
  Real const _epsilonOverLog2;
  static constexpr Real _alpha{ 0.6 };
};

template< typename REAL, int NDIM, typename ... TYPES >
struct Multiple
{
  using Real = REAL;

  /**
   * 
   */
  constexpr CArray< Real, NDIM > u( CArray< Real, NDIM > const & x ) const
  {
    CArray< Real, NDIM > u;
    int offset = 0;
    LvArray::typeManipulation::forEachInTuple(
      [&x, &u, &offset] ( auto const & changeOfVar )
      {
        constexpr int LOCAL_NDIM = std::remove_reference_t< decltype( changeOfVar ) >::NDIM;
        CArray< Real, LOCAL_NDIM > xLocal;
        for( int i = 0; i < LOCAL_NDIM; ++i )
        {
          xLocal[ i ] = x[ i + offset ];
        }

        CArray< Real, LOCAL_NDIM > const uLocal = changeOfVar.u( xLocal );

        for( int i = 0; i < LOCAL_NDIM; ++i )
        {
          u[ i + offset ] = uLocal[ i ];
        }

        offset += LOCAL_NDIM;
      }, _types
    );

    return u;
  }

  /**
   * 
   */
  constexpr Real jacobian( CArray< Real, NDIM > const & x ) const
  {
    Real globalJacobian = 1;
    int offset = 0;
    LvArray::typeManipulation::forEachInTuple(
      [&x, &globalJacobian, &offset] ( auto const & changeOfVar )
      {
        constexpr int LOCAL_NDIM = std::remove_reference_t< decltype( changeOfVar ) >::NDIM;
        CArray< Real, LOCAL_NDIM > xLocal;
        for( int i = 0; i < LOCAL_NDIM; ++i )
        {
          xLocal[ i ] = x[ i + offset ];
        }

        globalJacobian *= changeOfVar.jacobian( xLocal );
        offset += LOCAL_NDIM;
      }, _types
    );

    return globalJacobian;
  }
  
  std::tuple< TYPES ... > _types;
};

template< typename REAL, int NDIM, typename ... ARGS >
auto createMultiple( ARGS && ... args )
{
  return Multiple< REAL, NDIM, ARGS ... >{ { std::forward< ARGS >( args ) ... } };
}

} // tcscf::integration::changeOfVariables