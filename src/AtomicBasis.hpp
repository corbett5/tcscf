#pragma once

#include "LvArrayInterface.hpp"
#include "caliperInterface.hpp"
#include "qmcWrapper.hpp"
#include "mathFunctions.hpp"

#include <iostream>

namespace tcscf
{

///////////////////////////////////////////////////////////////////////////////////////////////////
// Integral functions
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
      return std::conj( b1( r, theta, phi ) ) * b2( r, theta, phi );
    }
  );
}


/**
 * 
 */
template< typename BASIS_FUNCTION >
auto coreMatrixElement( int const Z, BASIS_FUNCTION const & b1, BASIS_FUNCTION const & b2 )
{
  LVARRAY_ERROR( "Generic version not yet implemented, it would be nice to include both analytic and numeric derivatives" );
}

/**
 * 
 */
template< typename BASIS_FUNCTION >
auto r12MatrixElement(
  BASIS_FUNCTION const & b1,
  BASIS_FUNCTION const & b2,
  BASIS_FUNCTION const & b3,
  BASIS_FUNCTION const & basis4 )
{
  using Real = typename BASIS_FUNCTION::Real;

  LVARRAY_LOG( "evaluating r12 integral." );

  return qmc::sphericalCoordinates6DIntegral(
    [&] ( Real const r1, Real const theta1, Real const phi1, Real const r2, Real const theta2, Real const phi2 )
    {
      Real r12 = std::pow( r1, 2 ) + std::pow( r2, 2 );
      r12 -= 2 * r1 * r2 * (std::sin( theta1 ) * std::sin( theta2 ) * std::cos( phi1  - phi2 ) + std::cos( theta1 ) * std::cos( theta2 ));
      r12 = std::sqrt( r12 );

      // TODO:: fix the std::complex< double >{ 1 / r12 } and add operators to std::complex
      return std::conj( b1( r1, theta1, phi1 ) * b2( r2, theta2, phi2 ) ) * std::complex< double >{ 1 / r12 } * b3( r1, theta1, phi1 ) * basis4( r2, theta2, phi2 );
    }
  );
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Matrix stuff
///////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * 
 */
template< typename LAMBDA >
auto oneElectronSymmetricHermitianArray(
  int const nBasis,
  LAMBDA && lambda ) -> Array2d< decltype( lambda( 0, 0 ) ) >
{
  TCSCF_MARK_FUNCTION;

  LVARRAY_ERROR_IF_LT( nBasis, 0 );

  using ResultType = decltype( lambda( 0, 0 ) );
  Array2d< ResultType > matrix( nBasis, nBasis );

  for( int a = 0; a < nBasis; ++a )
  {
    for( int b = a; b < nBasis; ++b )
    {
      ResultType const value = lambda( a, b );
      matrix( a, b ) = value;
      matrix( b, a ) = std::conj( value );
    }
  }

  return matrix;
}

/**
 * 
 */
template< typename LAMBDA >
auto twoElectronSymmetricHermitianArray(
  int const nBasis,
  bool const realBasisFunctions,
  LAMBDA && lambda ) -> Array4d< decltype( lambda( 0, 0, 0, 0 ) ) >
{
  LVARRAY_ERROR_IF( realBasisFunctions, "Not yet supported." );

  LVARRAY_ERROR_IF_LT( nBasis, 0 );

  using ResultType = decltype( lambda( 0, 0, 0, 0 ) );
  Array4d< ResultType > matrix( nBasis, nBasis, nBasis, nBasis );

  for( int a = 0; a < nBasis; ++a )
  {
    for( int b = a; b < nBasis; ++b )
    {
      for( int c = a; c < nBasis; ++c )
      {
        int dStart = realBasisFunctions ? b : a;
        for( int d = dStart; d < nBasis; ++d )
        {
          ResultType const value = lambda( a, b, c, d );

          matrix( a, b, c, d ) = value;

          if( a != b && a != c && a != d )
          {
            matrix( b, a, d, c ) = value;
          }

          if( a != b && a != c )
          {
            matrix( c, d, a, b ) = std::conj( value );
          }

          if( a != d )
          { 
            matrix( d, c, b, a ) = std::conj( value );
          }
        }
      }
    }
  }

  return matrix;
}

/**
 * 
 */
template< typename BASIS >
auto computeCoreMatrix( int const Z, std::vector< BASIS > const & basisFunctions )
{
  TCSCF_MARK_FUNCTION;

  return oneElectronSymmetricHermitianArray( basisFunctions.size(),
    [Z, &basisFunctions] ( int const a, int const b )
    {
      return coreMatrixElement( Z, basisFunctions[ a ], basisFunctions[ b ] );
    }
  );
}

/**
 * 
 */
template< typename BASIS >
auto computeR12Matrix( std::vector< BASIS > const & basisFunctions )
{
  TCSCF_MARK_FUNCTION;

  return twoElectronSymmetricHermitianArray( basisFunctions.size(), BASIS::isBasisReal,
    [&basisFunctions] ( int const a, int const b, int const c, int const d )
    {
      return r12MatrixElement( basisFunctions[ a ], basisFunctions[ b ],
                               basisFunctions[ c ], basisFunctions[ d ] );
    }
  );
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Integrators
///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename BASIS >
auto computeDoubleOverlap( BASIS const & b1, BASIS const & b2, BASIS const & b3, BASIS const & basis4 )
{
  using REAL = typename BASIS::REAL;

  return sphericalCoordinates6DIntegralQMC(
    [&] ( REAL const r1, REAL const theta1, REAL const phi1, REAL const r2, REAL const theta2, REAL const phi2 )
    {
      return std::conj( b1( r1, theta1, phi1 ) * b2( r2, theta2, phi2 ) ) * b3( r1, theta1, phi1 ) * basis4( r2, theta2, phi2 );
    }
  );
}

} // namespace tcscf
