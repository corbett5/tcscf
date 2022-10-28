#pragma once

#include "qmcWrapper.hpp"

#include "../RAJAInterface.hpp"

namespace tcscf::integration
{

/**
 * 
 */
template< typename REAL, int NDIM >
struct QMCGrid
{
  static_assert( NDIM == 3 || NDIM == 2 );

  using BASIS_VALUE_TYPE = std::conditional_t< NDIM == 3, std::complex< REAL >, REAL >;

  template< typename ATOMIC_BASIS >
  QMCGrid(
    IndexType gridSize,
    std::vector< ATOMIC_BASIS > const & basisFunctions,
    bool const storeGradients )
  {
    TCSCF_MARK_SCOPE( "QMCGrid::QMCGrid" );

    static_assert( std::is_same_v< REAL, typename ATOMIC_BASIS::Real > );

    constexpr changeOfVariables::TreutlerAhlrichsM4< REAL > rChange( 0, 0.9 );
    constexpr changeOfVariables::Linear< REAL > thetaChange{ pi< REAL > };
    constexpr changeOfVariables::Linear< REAL > phiChange{ 2 * pi< REAL > };

    QMCCache< NDIM, REAL > const qmcCache( gridSize );
    if constexpr ( NDIM == 2 )
    {
      quadratureGrid = integration::createGrid( qmcCache,
        changeOfVariables::createMultiple< REAL, 2 >( rChange, thetaChange ) );
    }
    else
    {
      quadratureGrid = integration::createGrid( qmcCache,
        changeOfVariables::createMultiple< REAL, 3 >( rChange, thetaChange, phiChange ) );
    }

    IndexType const nBasis = basisFunctions.size();
    basisValues.resize( quadratureGrid.weights.size(), nBasis );
    if( storeGradients )
    {
      basisGradients.resize( quadratureGrid.weights.size(), nBasis );
    }

    for( IndexType i = 0; i < quadratureGrid.weights.size(); ++i )
    {
      Spherical< REAL > const r {
        quadratureGrid.points( 0, i ),
        quadratureGrid.points( 1, i ),
        NDIM == 3 ? quadratureGrid.points( 2, i ) : REAL {} };

      if constexpr ( NDIM == 2 )
      {
        quadratureGrid.points( 0, i ) = r.x();
        quadratureGrid.points( 1, i ) = r.z();
      }
      else
      {
        quadratureGrid.points( 0, i ) = r.x();
        quadratureGrid.points( 1, i ) = r.y();
        quadratureGrid.points( 2, i ) = r.z();
      }

      REAL const jacobian = std::pow( r.r(), 2 ) * std::sin( r.theta() );
      quadratureGrid.weights( i ) *= jacobian;

      for( int b = 0; b < nBasis; ++b )
      {
        if constexpr ( NDIM == 2 )
        {
          basisValues( i, b ) = std::real( basisFunctions[ b ]( r ) );
        }
        else
        {
          basisValues( i, b ) = basisFunctions[ b ]( r );
        }

        if( storeGradients )
        {
          basisGradients( i, b ) = basisFunctions[ b ].gradient( r );
        }
      }
    }
  };


  integration::QuadratureGrid< REAL > quadratureGrid{ 0, 0 };
  Array2d< BASIS_VALUE_TYPE > basisValues;
  Array2d< Cartesian< std::complex< REAL > > > basisGradients;
};

/**
 *
 */
template< typename REAL, typename F >
REAL evaluateR2Integral(
  ArrayView2d< REAL const > const & points,
  ArrayView1d< REAL const > const & weights,
  ArrayView2d< REAL const > const & basisValues,
  Cartesian< REAL > const & r1,
  IndexType const b2,
  IndexType const b4,
  F && f )
{
  using Real = REAL;

  REAL answer = 0;
  for( IndexType idx = 0; idx < weights.size(); ++idx )
  {
    Cartesian< Real > const r2 { points( 0, idx ), 0, points( 1, idx ) };
    answer = answer + weights[ idx ] * basisValues( idx, b4 ) * f( r1, r2 ) * basisValues( idx, b2 );
  }

  return 2 * pi< Real > * answer;
}

/**
 *
 */
template< typename REAL, typename F >
std::complex< REAL > evaluateR2GradientIntegral(
  ArrayView2d< REAL const > const & points,
  ArrayView1d< REAL const > const & weights,
  ArrayView2d< REAL const > const & basisValues,
  ArrayView2d< Cartesian< std::complex< REAL > > const > const & basisGradients,
  Cartesian< REAL > const & r1,
  IndexType const b2,
  IndexType const b4,
  F && f )
{
  using Real = REAL;
  using Complex = std::complex< Real >;

  Complex answer = 0;
  for( IndexType idx = 0; idx < weights.size(); ++idx )
  {
    Cartesian< Real > const r2 { points( 0, idx ), 0, points( 1, idx ) };
    answer = answer + weights[ idx ] * basisValues( idx, b4 ) * dot( f( r1, r2 ), basisGradients( idx, b2 ) );
  }

  return 2 * pi< Real > * answer;
}

/**
 * Note: here we are integrating over r1 and r2 using the logarithmic grid.
 */
template< bool USE_GRADIENT, typename REAL, typename ATOMIC_BASIS, typename F >
Array4d< std::complex< REAL > > integrateAllR1R2(
  IndexType const nQMC1,
  IndexType const nQMC2,
  std::vector< ATOMIC_BASIS > const & basisFunctions,
  F && f )
{
  using PolicyType = ParallelHost;
  using Real = REAL;
  using Complex = std::complex< Real >;

  static_assert( std::is_same_v< Real, typename ATOMIC_BASIS::Real > );

  IndexType const nBasis = basisFunctions.size();

  QMCGrid< Real, 3 > const r1Grid( nQMC1, basisFunctions, false );
  QMCGrid< Real, 2 > const r2Grid( nQMC2, basisFunctions, USE_GRADIENT );

  Array4d< Complex > answer( nBasis, nBasis, nBasis, nBasis );

  ArrayView2d< Real const > const r1Points = r1Grid.quadratureGrid.points.toViewConst();
  ArrayView1d< Real const > const r1Weights = r1Grid.quadratureGrid.weights.toViewConst();
  ArrayView2d< Complex const > const r1BasisValues = r1Grid.basisValues.toViewConst();

  ArrayView2d< Real const > const r2Points = r2Grid.quadratureGrid.points.toViewConst();
  ArrayView1d< Real const > const r2Weights = r2Grid.quadratureGrid.weights.toViewConst();
  ArrayView2d< Real const > const r2BasisValues = r2Grid.basisValues.toViewConst();
  ArrayView2d< Cartesian< Complex > const > const r2BasisGradients = r2Grid.basisGradients.toViewConst();

  TCSCF_MARK_SCOPE( "integrate" );

  forAll< DefaultPolicy< PolicyType > >( r1Grid.quadratureGrid.points.size( 1 ),
    [=, answer=answer.toView()] ( IndexType const idx )
    {
      Cartesian< Real > const r1 = { r1Points( 0, idx ), r1Points( 1, idx ), r1Points( 2, idx ) };

      Real const r1Weight = r1Weights( idx );

      for( IndexType b2 = 0; b2 < nBasis; ++b2 )
      {
        for( IndexType b4 = 0; b4 < nBasis; ++b4 )
        {
          Complex innerIntegral;
          if constexpr ( USE_GRADIENT )
          {
            innerIntegral = r1Weight * evaluateR2GradientIntegral(
              r2Points,
              r2Weights,
              r2BasisValues,
              r2BasisGradients,
              r1,
              b2,
              b4,
              f );
          }
          else
          {
            innerIntegral = r1Weight * evaluateR2Integral(
              r2Points,
              r2Weights,
              r2BasisValues,
              r1,
              b2,
              b4,
              f );
          }
          
          for( IndexType b1 = 0; b1 < nBasis; ++b1 )
          {
            for( IndexType b3 = 0; b3 < nBasis; ++b3 )
            {
              Complex const r1Contrib = conj( r1BasisValues( idx, b1 ) ) * r1BasisValues( idx, b3 );
              Complex const addition = innerIntegral * r1Contrib;
              atomicAdd< PolicyType >( &answer( b1, b2, b3, b4 ), addition );
            }
          }
        }
      }
    }
  );

  return answer;
}

} // namespace tcscf::integration
