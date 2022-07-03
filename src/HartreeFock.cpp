#include "HartreeFock.hpp"

namespace tcscf
{

namespace internal
{

template< typename T >
void constructFockOperator(
  ArrayView2d< T, 0 > const & fockOperator,
  ArrayView2d< T const > const & oneElectronTerms,
  ArrayView4d< T const > const & twoElectronTerms,
  ArrayView2d< T const > const & density )
{
  IndexType const basisSize = fockOperator.size( 0 );
  for( IndexType i = 0; i < basisSize; ++i )
  {
    for( IndexType j = 0; j < basisSize; ++j )
    {
      T tmp = 0;
      for( IndexType a = 0; a < basisSize; ++a )
      {
        for( IndexType b = 0; b < basisSize; ++b)
        {
          tmp += density( a, b ) * (twoElectronTerms( j, b, a, i ) - twoElectronTerms( j, b, i, a ) / T{ 2 });
        }
      }

      fockOperator( j, i ) = oneElectronTerms( j, i ) + tmp;
    }
  }
}

template< typename T >
void getNewDensity(
  IndexType const numElectrons,
  ArrayView2d< T > const & density,
  ArrayView2d< T const, 0 > const & eigenvectors )
{
  LVARRAY_ERROR_IF_NE( numElectrons % 2, 0 );
  IndexType const basisSize = eigenvectors.size( 0 );
  
  // TODO: replace with with matrix multiplication C C^\dagger
  for( IndexType a = 0; a < basisSize; ++a )
  {
    for( IndexType b = 0; b < basisSize; ++b )
    {
      T tmp = 0;
      for( IndexType i = 0; i < numElectrons / 2; ++i )
      {
        tmp += eigenvectors( a, i ) * std::conj( eigenvectors( b, i ) );
      }

      density( a, b ) = T{ 2 } * tmp;
    }
  }
}

template< typename T >
T calculateEnergy(
  ArrayView2d< T const, 0 > const & fockOperator,
  ArrayView2d< T const > const & oneElectronTerms,
  ArrayView2d< T const > const & density )
{
  IndexType const basisSize = oneElectronTerms.size( 0 );
  
  T energy = 0;
  for( int i = 0; i < basisSize; ++i )
  {
    for( int j = 0; j < basisSize; ++j )
    {
      energy += density( i, j ) * (oneElectronTerms( j, i ) + fockOperator( j, i ));
    }
  }

  return energy / T{ 2 };
}

} // namespace internal

template< typename T >
void RCSHartreeFock< T >::compute(
  bool const orthogonal,
  ArrayView2d< T const > const & overlap,
  ArrayView2d< T const > const & oneElectronTerms,
  ArrayView4d< T const > const & twoElectronTerms )
{
  LVARRAY_ERROR_IF_NE( oneElectronTerms.size( 1 ), nBasis );

  LVARRAY_ERROR_IF_NE( twoElectronTerms.size( 0 ), nBasis );
  LVARRAY_ERROR_IF_NE( twoElectronTerms.size( 1 ), nBasis );
  LVARRAY_ERROR_IF_NE( twoElectronTerms.size( 2 ), nBasis );
  LVARRAY_ERROR_IF_NE( twoElectronTerms.size( 3 ), nBasis );

  LVARRAY_ERROR_IF_NE( density.size( 0 ), nBasis );
  LVARRAY_ERROR_IF_NE( density.size( 1 ), nBasis );

  if( !orthogonal )
  {
    LVARRAY_ERROR_IF_NE( overlap.size( 0 ), nBasis );
    LVARRAY_ERROR_IF_NE( overlap.size( 1 ), nBasis );
  }

  T previousEnergy = std::numeric_limits< Real >::max();

  LVARRAY_LOG_VAR( oneElectronTerms );

  for( int i = 0; i < 10; ++i )
  {
    internal::constructFockOperator( fockOperator.toView(), oneElectronTerms, twoElectronTerms, density.toViewConst() );

    T const energy = internal::calculateEnergy( fockOperator.toViewConst(), oneElectronTerms, density.toViewConst() );
    LVARRAY_LOG_VAR( energy );
    previousEnergy = energy;

    if( orthogonal )
    {
      hermitianEigendecomposition( fockOperator.toView(), eigenvalues.toView() );
    }
    else
    {
      LVARRAY_ERROR( "Generalized eigenvalue problem not supported yet" );
    }

    internal::getNewDensity( nElectrons, density, fockOperator.toViewConst() );
  }
}

// Explicit instantiations.
template struct RCSHartreeFock< std::complex< double > >;

} // namespace tcscf
