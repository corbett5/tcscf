#pragma once

#include "LvArrayInterface.hpp"
#include "caliperInterface.hpp"
#include "qmcWrapper.hpp"
#include "mathFunctions.hpp"

#include "integration/TreutlerAhlrichsLebedev.hpp"

#include <iostream>

namespace tcscf
{

///////////////////////////////////////////////////////////////////////////////////////////////////
// General integrals
///////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * 
 */
template< typename BASIS_FUNCTION >
auto overlap( BASIS_FUNCTION const & b1, BASIS_FUNCTION const & b2 )
{
  using Real = typename BASIS_FUNCTION::Real;

  return qmc::sphericalCoordinates3DIntegral(
    [b1, b2] ( Real const r, Real const theta, Real const phi )
    {
      return conj( b1( r, theta, phi ) ) * b2( r, theta, phi );
    }
  );
}

/**
 * 
 */
template< typename BASIS_FUNCTION >
auto overlapBoost( BASIS_FUNCTION const & b1, BASIS_FUNCTION const & b2 )
{
  using Real = typename BASIS_FUNCTION::Real;

  return sphericalCoordinates3DIntegral< Real >(
    [b1, b2] ( Real const r, Real const theta, Real const phi )
    {
      return conj( b1( r, theta, phi ) ) * b2( r, theta, phi );
    }
  );
}

/**
 * 
 */
template< typename REAL >
constexpr REAL calculateR12(
  REAL const r1,
  REAL const theta1,
  REAL const phi1,
  REAL const r2,
  REAL const theta2,
  REAL const phi2 )
{
  REAL r12 = std::pow( r1, 2 ) + std::pow( r2, 2 );
  r12 -= 2 * r1 * r2 * (std::sin( theta1 ) * std::sin( theta2 ) * std::cos( phi1  - phi2 ) + std::cos( theta1 ) * std::cos( theta2 ));
  return std::sqrt( std::abs( r12 ) );
}

/**
 * 
 */
template< typename BASIS_FUNCTION >
auto r12MatrixElement(
  BASIS_FUNCTION const & b1,
  BASIS_FUNCTION const & b2,
  BASIS_FUNCTION const & b3,
  BASIS_FUNCTION const & b4 )
{
  using Real = typename BASIS_FUNCTION::Real;

  return qmc::sphericalCoordinates6DIntegral(
    [&] ( Real const r1, Real const theta1, Real const phi1, Real const r2, Real const theta2, Real const phi2 )
    {
      Real const r12 = calculateR12( r1, theta1, phi1, r2, theta2, phi2 );
      return conj( b1( r1, theta1, phi1 ) * b2( r2, theta2, phi2 ) ) * (1 / r12) * b3( r1, theta1, phi1 ) * b4( r2, theta2, phi2 );
    }
  );
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Specific Atomic integrals
///////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * 
 */
template< typename BASIS_FUNCTION >
typename BASIS_FUNCTION::Real atomicCoulombOperator(
  BASIS_FUNCTION const & b1,
  BASIS_FUNCTION const & b2,
  BASIS_FUNCTION const & b3,
  BASIS_FUNCTION const & b4 )
{
  TCSCF_MARK_FUNCTION;

  using Real = typename BASIS_FUNCTION::Real;

  LVARRAY_ERROR_IF_NE( b1.l, b3.l );
  LVARRAY_ERROR_IF_NE( b1.m, b3.m );
  LVARRAY_ERROR_IF_NE( b2.l, b4.l );
  LVARRAY_ERROR_IF_NE( b2.m, b4.m );

  int const l1 = b1.l;
  int const m1 = b1.m;

  int const l2 = b2.l;
  int const m2 = b2.m;

  return qmc::sphericalCoordinates5DIntegral(
    [&] ( Real const r1, Real const theta1, Real const phi1, Real const r2, Real const theta2 )
    {
      Real const r12 = calculateR12( r1, theta1, phi1, r2, theta2, Real( 0 ) );

      Real const b1V = b1.radialComponent( r1 );
      Real const b2V = b2.radialComponent( r2 );
      Real const b3V = b3.radialComponent( r1 );
      Real const b4V = b4.radialComponent( r2 );

      Real const angularComponent = std::pow( sphericalHarmonicMagnitude( l1, m1, theta1 ), 2 ) *
                                    std::pow( sphericalHarmonicMagnitude( l2, m2, theta2 ), 2 );
      
      return angularComponent * b1V * b2V * (1 / r12) * b3V * b4V;
    }
  );
}

/**
 * 
 */
template< typename BASIS_FUNCTION >
typename std::complex< typename BASIS_FUNCTION::Real > atomicExchangeOperator(
  BASIS_FUNCTION const & b1,
  BASIS_FUNCTION const & b2,
  BASIS_FUNCTION const & b3,
  BASIS_FUNCTION const & b4 )
{
  TCSCF_MARK_FUNCTION;

  using Real = typename BASIS_FUNCTION::Real;

  LVARRAY_ERROR_IF_NE( b1.l, b4.l );
  LVARRAY_ERROR_IF_NE( b1.m, b4.m );
  LVARRAY_ERROR_IF_NE( b2.l, b3.l );
  LVARRAY_ERROR_IF_NE( b2.m, b3.m );

  int const l1 = b1.l;
  int const m1 = b1.m;

  int const l2 = b2.l;
  int const m2 = b2.m;

  return qmc::sphericalCoordinates5DIntegral(
    [&] ( Real const r1, Real const theta1, Real const phi1, Real const r2, Real const theta2 )
    {
      Real const r12 = calculateR12( r1, theta1, phi1, r2, theta2, Real( 0 ) );

      Real const b1V = b1.radialComponent( r1 ) * sphericalHarmonicMagnitude( l1, m1, theta1 );
      Real const b2V = b2.radialComponent( r2 ) * sphericalHarmonicMagnitude( l2, m2, theta2 );
      Real const b3V = b3.radialComponent( r1 ) * sphericalHarmonicMagnitude( l2, m2, theta1 );
      Real const b4V = b4.radialComponent( r2 ) * sphericalHarmonicMagnitude( l1, m1, theta2 );

      return (b1V * b2V * (1 / r12) * b3V * b4V) * std::exp( I< Real > * (m2 - m1) * phi1 );
    }
  );
}

/**
 * 
 */
struct AtomicParams
{
  int const n;
  int const l;
  int const m;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Matrix stuff
///////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * 
 */
template< typename T, typename LAMBDA >
void fillOneElectronHermitianMatrix(
  ArrayView2d< T > const & matrix,
  LAMBDA && lambda )
{
  TCSCF_MARK_FUNCTION;

  LVARRAY_ERROR_IF_NE( matrix.size( 0 ), matrix.size( 1 ) );

  using ResultType = decltype( lambda( 0, 0 ) );

  for( int a = 0; a < matrix.size( 0 ); ++a )
  {
    for( int b = a; b < matrix.size( 0 ); ++b )
    {
      ResultType const value = lambda( a, b );
      matrix( a, b ) += value;

      if( a != b )
      {
        matrix( b, a ) += conj( value );
      }
    }
  }
}

/**
 * 
 */
template< typename T, typename LAMBDA >
void fillTwoElectronSymmetricHermitianArray(
  ArrayView4d< T > const & array,
  bool const realBasisFunctions,
  LAMBDA && lambda )
{
  TCSCF_MARK_FUNCTION;

  LVARRAY_ERROR_IF_NE( array.size( 0 ), array.size( 1 ) );
  LVARRAY_ERROR_IF_NE( array.size( 0 ), array.size( 2 ) );
  LVARRAY_ERROR_IF_NE( array.size( 0 ), array.size( 3 ) );

  int const N = array.size( 0 );

  for( int a = 0; a < N; ++a )
  {
    for( int b = a; b < N; ++b )
    {
      for( int c = a; c < N; ++c )
      {
        int const dStart = realBasisFunctions ? b : a;
        for( int d = dStart; d < N; ++d )
        {
          T const value{ lambda( a, b, c, d ) };

          array( a, b, c, d ) += value;

          bool const useSymmetry = a != b;
          bool const useHermitian = a != c && a != d;
          bool const useReal = realBasisFunctions && d != b;

          if( useSymmetry )
          {
            array( b, a, d, c ) += value;
          }

          if( useHermitian || (realBasisFunctions && a != c) )
          {
            array( c, d, a, b ) += conj( value );
          }

          if( useHermitian && useSymmetry )
          { 
            array( d, c, b, a ) += conj( value );
          }

          if( useReal )
          {
            array( a, d, c, b ) += value;
          }

          if( useReal && useSymmetry )
          {
            array( d, a, b, c ) += value;
          }

          if( useReal && useHermitian )
          {
            array( c, b, a, d ) += value;
          }

          if( useReal && useHermitian && useSymmetry )
          {
            array( b, c, d, a ) += value;
          }
        }
      }
    }
  }
}

/**
 * 
 */
template< typename BASIS, typename T >
void fillCoreMatrix(
  int const Z,
  std::vector< BASIS > const & basisFunctions,
  ArrayView2d< T > const & matrix )
{
  LVARRAY_ERROR_IF_NE( matrix.size( 0 ), IndexType( basisFunctions.size() ) );

  fillOneElectronHermitianMatrix( matrix,
    [Z, &basisFunctions] ( int const a, int const b )
    {
      return coreMatrixElement( Z, basisFunctions[ a ], basisFunctions[ b ] );
    }
  );
}

/**
 * 
 */
template< typename BASIS, typename T >
auto fillR12Array(
  std::vector< BASIS > const & basisFunctions,
  ArrayView4d< T > const & array )
{
  return fillTwoElectronSymmetricHermitianArray( array, BASIS::isBasisReal,
    [&basisFunctions] ( int const a, int const b, int const c, int const d )
    {
      return r12MatrixElement( basisFunctions[ a ], basisFunctions[ b ],
                               basisFunctions[ c ], basisFunctions[ d ] );
    }
  );
}

/**
 * 
 */
template< typename BASIS, typename T >
void fillAtomicR12Array(
  std::vector< BASIS > const & basisFunctions,
  ArrayView4d< T > const & array )
{
  using Real = typename BASIS::Real;

  int numEval = 0;

  fillTwoElectronSymmetricHermitianArray( array, true,
    [&basisFunctions, &numEval] ( int const a, int const b, int const c, int const d )
    {
      if( basisFunctions[ a ].l == basisFunctions[ c ].l &&
          basisFunctions[ a ].m == basisFunctions[ c ].m &&
          basisFunctions[ b ].l == basisFunctions[ d ].l &&
          basisFunctions[ b ].m == basisFunctions[ d ].m )
      {
        ++numEval;
        return std::complex< Real >{ atomicCoulombOperator( basisFunctions[ a ], basisFunctions[ b ], basisFunctions[ c ], basisFunctions[ d ] ) };
      }
      
      return std::complex< Real >{ 0 };
    }
  );

  fillTwoElectronSymmetricHermitianArray( array, false,
    [&basisFunctions, &numEval] ( int const a, int const b, int const c, int const d )
    {
      if( basisFunctions[ a ].l == basisFunctions[ c ].l &&
          basisFunctions[ a ].m == basisFunctions[ c ].m &&
          basisFunctions[ b ].l == basisFunctions[ d ].l &&
          basisFunctions[ b ].m == basisFunctions[ d ].m )
      {
        return std::complex< Real >{ 0 };
      }

      if( basisFunctions[ a ].l == basisFunctions[ d ].l &&
          basisFunctions[ a ].m == basisFunctions[ d ].m &&
          basisFunctions[ b ].l == basisFunctions[ c ].l &&
          basisFunctions[ b ].m == basisFunctions[ c ].m )
      {
        ++numEval;
        return atomicExchangeOperator(
          basisFunctions[ a ],
          basisFunctions[ b ],
          basisFunctions[ c ],
          basisFunctions[ d ] );
      }

      return std::complex< Real >{ 0 };
    }
  );
}

template< typename BASIS, typename T >
void fillAtomicR12Array(
  double const epsilon,
  int const nRadial,
  int const angularOrder,
  std::vector< BASIS > const & basisFunctions,
  ArrayView4d< T > const & array )
{
  using Real = typename BASIS::Real;

  integration::TreutlerAhlrichsLebedev< Real > grid( epsilon, nRadial, angularOrder );

  fillTwoElectronSymmetricHermitianArray( array, true,
    [&basisFunctions, &grid] ( int const a, int const b, int const c, int const d )
    {
      if( ( basisFunctions[ a ].l == basisFunctions[ c ].l &&
            basisFunctions[ a ].m == basisFunctions[ c ].m &&
            basisFunctions[ b ].l == basisFunctions[ d ].l &&
            basisFunctions[ b ].m == basisFunctions[ d ].m ) ||
          ( basisFunctions[ a ].l == basisFunctions[ d ].l &&
            basisFunctions[ a ].m == basisFunctions[ d ].m &&
            basisFunctions[ b ].l == basisFunctions[ c ].l &&
            basisFunctions[ b ].m == basisFunctions[ c ].m ) )
      {
        return integrateR1R12( grid, basisFunctions[ a ], basisFunctions[ b ], basisFunctions[ c ], basisFunctions[ d ],
          [] ( Real const LVARRAY_UNUSED_ARG( r1 ), Real const LVARRAY_UNUSED_ARG( r2 ), Real const r12 )
          { return 1.0 / r12; }
        );
      }
      
      return std::complex< Real >{ 0 };
    }
  );
}

} // namespace tcscf
