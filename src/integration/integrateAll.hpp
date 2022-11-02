#pragma once

#include "qmcWrapper.hpp"

#include "../RAJAInterface.hpp"

namespace tcscf::integration
{

/**
 */
template< typename REAL, int NDIM >
struct QMCGrid
{
  static_assert( NDIM == 3 || NDIM == 2 );

  using BASIS_VALUE_TYPE = std::conditional_t< NDIM == 3, std::complex< REAL >, REAL >;

  /**
   */
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

    forAll< DefaultPolicy< ParallelHost > >( quadratureGrid.weights.size(),
      [&] ( IndexType const i )
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
    );
  };

  /**
   */
  IndexType nBasis() const
  {
    return basisValues.size( 1 );
  }

  /**
   */
  bool gradientsStored() const
  {
    return !basisGradients.empty();
  }


  integration::QuadratureGrid< REAL > quadratureGrid{ 0, 0 };
  Array2d< BASIS_VALUE_TYPE > basisValues;
  Array2d< Cartesian< std::complex< REAL > > > basisGradients;
};

/**
 *
 */
static constexpr IndexType BATCH_SIZE = 16;

/**
 *
 */
template< typename REAL, typename F >
CArray< REAL, BATCH_SIZE > evaluateR2Integral(
  ArrayView2d< REAL const > const & points,
  ArrayView1d< REAL const > const & weights,
  ArrayView2d< REAL const > const & basisValues,
  Cartesian< REAL > const & r1,
  IndexType const b2Min,
  IndexType const batchSize,
  IndexType const b4,
  F && f )
{
  using Real = REAL;

  CArray< Real, BATCH_SIZE > answer {};
  for( IndexType idx = 0; idx < weights.size(); ++idx )
  {
    Cartesian< Real > const r2 { points( 0, idx ), 0, points( 1, idx ) };
    auto const integrand = weights[ idx ] * f( r1, r2 ) * basisValues( idx, b4 );

    for( IndexType b2 = b2Min; b2 < b2Min + batchSize; ++b2 )
    {
      answer[ b2 - b2Min ] += conj( basisValues( idx, b2 ) ) * integrand;
    }
  }

  LvArray::tensorOps::scale< BATCH_SIZE >( answer, 2 * pi< Real > );

  return answer;
}

/**
 * Evaluate the integral
 * \int \phi_{b_2}^*(\vec{r}_2) \vec{f}(\vec{r}_1, \vec{r}_2) \cdot \vec{\nabla} \phi_{b_4}(\vec{r}_2) d \vec{r}_2
 * for values of b4 in the range [b4Min, b4Min + batchSize)
 */
template< typename REAL, typename F >
CArray< std::complex< REAL >, BATCH_SIZE > evaluateR2GradientIntegral(
  ArrayView2d< REAL const > const & points,
  ArrayView1d< REAL const > const & weights,
  ArrayView2d< REAL const > const & basisValues,
  ArrayView2d< Cartesian< std::complex< REAL > > const > const & basisGradients,
  Cartesian< REAL > const & r1,
  IndexType const b2Min,
  IndexType const batchSize,
  IndexType const b4,
  F && f )
{
  using Real = REAL;

  CArray< std::complex< Real >, BATCH_SIZE > answer {};
  for( IndexType idx = 0; idx < weights.size(); ++idx )
  {
    Cartesian< Real > const r2 { points( 0, idx ), 0, points( 1, idx ) };
    auto const integrand = weights[ idx ] * dot( f( r1, r2 ), basisGradients( idx, b4 ) );

    for( IndexType b2 = b2Min; b2 < b2Min + batchSize; ++b2 )
    {
      answer[ b2 - b2Min ] += conj( basisValues( idx, b2 ) ) * integrand;
    }
  }

  LvArray::tensorOps::scale< BATCH_SIZE >( answer, 2 * pi< Real > );

  return answer;
}

/**
 * Note: here we are integrating over r1 and r2 using the logarithmic grid.
 */
template< bool USE_GRADIENT, typename REAL, typename F >
Array4d< std::complex< REAL > > integrateAllR1R2(
  QMCGrid< REAL, 3 > const & r1Grid,
  QMCGrid< REAL, 2 > const & r2Grid,
  F && f )
{
  using PolicyType = ParallelHost;
  using Real = REAL;
  using Complex = std::complex< Real >;

  using InnerIntegralType = std::conditional_t< USE_GRADIENT, Complex, Real >;

  LVARRAY_ERROR_IF( USE_GRADIENT && !r2Grid.gradientsStored(), "The gradients need to be stored." );
  LVARRAY_ERROR_IF_NE( r1Grid.nBasis(), r2Grid.nBasis() );

  IndexType const nBasis = r1Grid.nBasis();

  Array4d< Complex > answer( nBasis, nBasis, nBasis, nBasis );

  ArrayView2d< Real const > const r1Points = r1Grid.quadratureGrid.points.toViewConst();
  ArrayView1d< Real const > const r1Weights = r1Grid.quadratureGrid.weights.toViewConst();
  ArrayView2d< Complex const > const r1BasisValues = r1Grid.basisValues.toViewConst();

  ArrayView2d< Real const > const r2Points = r2Grid.quadratureGrid.points.toViewConst();
  ArrayView1d< Real const > const r2Weights = r2Grid.quadratureGrid.weights.toViewConst();
  ArrayView2d< Real const > const r2BasisValues = r2Grid.basisValues.toViewConst();
  ArrayView2d< Cartesian< Complex > const > const r2BasisGradients = r2Grid.basisGradients.toViewConst();

  forAll< DefaultPolicy< PolicyType > >( r1Grid.quadratureGrid.points.size( 1 ),
    [=, answer=answer.toView()] ( IndexType const idx )
    {
      Cartesian< Real > const r1 = { r1Points( 0, idx ), r1Points( 1, idx ), r1Points( 2, idx ) };

      Real const r1Weight = r1Weights( idx );

      for( IndexType b4 = 0; b4 < nBasis; ++b4 )
      {
        for( IndexType b2Min = 0; b2Min < nBasis; b2Min += BATCH_SIZE )
        {
          IndexType const curBatchSize = std::min( nBasis - b2Min, BATCH_SIZE );

          CArray< InnerIntegralType, BATCH_SIZE > innerIntegral;
          if constexpr ( USE_GRADIENT )
          {
            innerIntegral = evaluateR2GradientIntegral(
              r2Points,
              r2Weights,
              r2BasisValues,
              r2BasisGradients,
              r1,
              b2Min,
              curBatchSize,
              b4,
              f );
          }
          else
          {
            innerIntegral = evaluateR2Integral(
              r2Points,
              r2Weights,
              r2BasisValues,
              r1,
              b2Min,
              curBatchSize,
              b4,
              f );
          }

          LvArray::tensorOps::scale< BATCH_SIZE >( innerIntegral, r1Weight );

          for( IndexType b2 = b2Min; b2 < b2Min + curBatchSize; ++b2 )
          {
            for( IndexType b1 = 0; b1 < nBasis; ++b1 )
            {
              for( IndexType b3 = 0; b3 < nBasis; ++b3 )
              {
                Complex const r1Contrib = conj( r1BasisValues( idx, b1 ) ) * r1BasisValues( idx, b3 );
                Complex const addition = innerIntegral[ b2 - b2Min ] * r1Contrib;
                
                if constexpr ( USE_GRADIENT )
                {
                  atomicAdd< PolicyType >( &answer( b2, b1, b4, b3 ), addition );
                }
                else
                {
                  atomicAdd< PolicyType >( &answer( b1, b2, b3, b4 ), addition );
                }
              }
            }
          }
        }
      }
    }
  );

  return answer;
}

} // namespace tcscf::integration
