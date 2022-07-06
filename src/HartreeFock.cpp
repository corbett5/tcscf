#include "HartreeFock.hpp"

#include "caliperInterface.hpp"

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
  TCSCF_MARK_FUNCTION;

  IndexType const basisSize = fockOperator.size( 0 );
  for( IndexType u = 0; u < basisSize; ++u )
  {
    for( IndexType v = 0; v < basisSize; ++v )
    {
      T tmp = 0;
      for( IndexType a = 0; a < basisSize; ++a )
      {
        for( IndexType b = 0; b < basisSize; ++b)
        {
          tmp += density( a, b ) * (twoElectronTerms( u, b, v, a ) - twoElectronTerms( u, b, a, v ) / 2);
        }
      }

      fockOperator( u, v ) = oneElectronTerms( u, v ) + tmp;
    }
  }
}

void constructAtomicFockOperator(
  ArrayView2d< std::complex< double >, 0 > const & fockOperator,
  ArrayView2d< double const > const & oneElectronTerms,
  ArrayView4d< std::complex< double > const > const & twoElectronTerms,
  ArrayView2d< std::complex< double > const > const & density,
  std::vector< HydrogenLikeBasisFunction< double > > const & basisFunctions )
{
  TCSCF_MARK_FUNCTION;

  fockOperator.zero();

  IndexType const basisSize = fockOperator.size( 0 );
  for( IndexType u = 0; u < basisSize; ++u )
  {
    for( IndexType v = 0; v < basisSize; ++v )
    {
      if( basisFunctions[ u ].l != basisFunctions[ v ].l ) continue;
      if( basisFunctions[ u ].m != basisFunctions[ v ].m ) continue;

      std::complex< double > tmp = 0;
      for( IndexType a = 0; a < basisSize; ++a )
      {
        for( IndexType b = 0; b < basisSize; ++b)
        {
          if( basisFunctions[ a ].l != basisFunctions[ b ].l ) continue;
          if( basisFunctions[ a ].m != basisFunctions[ b ].m ) continue;

          tmp += density( a, b ) * (twoElectronTerms( u, b, v, a ) - twoElectronTerms( u, b, a, v ) / std::complex< double >{ 2 });
        }
      }

      fockOperator( u, v ) = oneElectronTerms( u, v ) + tmp;
    }
  }
}

template< typename T >
void getNewDensity(
  IndexType const nElectrons,
  ArrayView2d< T > const & density,
  ArrayView2d< T const, 0 > const & eigenvectors )
{
  TCSCF_MARK_FUNCTION;

  LVARRAY_ERROR_IF_NE( nElectrons % 2, 0 );
  IndexType const basisSize = eigenvectors.size( 0 );
  
  // TODO: replace with with matrix multiplication C C^\dagger
  for( IndexType u = 0; u < basisSize; ++u )
  {
    for( IndexType v = 0; v < basisSize; ++v )
    {
      T tmp = 0;
      for( IndexType a = 0; a < nElectrons / 2; ++a )
      {
        tmp += eigenvectors( u, a ) * conj( eigenvectors( v, a ) );
      }

      density( u, v ) = 2 * tmp;
    }
  }
}

template< typename T, typename U >
T calculateEnergy(
  ArrayView2d< T const, 0 > const & fockOperator,
  ArrayView2d< U const > const & oneElectronTerms,
  ArrayView2d< T const > const & density )
{
  TCSCF_MARK_FUNCTION;

  IndexType const basisSize = oneElectronTerms.size( 0 );
  
  T energy = 0;
  for( int u = 0; u < basisSize; ++u )
  {
    for( int v = 0; v < basisSize; ++v )
    {
      energy += density( v, u ) * (oneElectronTerms( u, v ) + fockOperator( u, v ));
    }
  }

  return energy / 2;
}

} // namespace internal

template< typename T >
void RCSHartreeFock< T >::compute(
  bool const orthogonal,
  ArrayView2d< T const > const & overlap,
  ArrayView2d< T const > const & oneElectronTerms,
  ArrayView4d< T const > const & twoElectronTerms )
{
  TCSCF_MARK_FUNCTION;

  LVARRAY_ERROR_IF_NE( oneElectronTerms.size( 1 ), basisSize );

  LVARRAY_ERROR_IF_NE( twoElectronTerms.size( 0 ), basisSize );
  LVARRAY_ERROR_IF_NE( twoElectronTerms.size( 1 ), basisSize );
  LVARRAY_ERROR_IF_NE( twoElectronTerms.size( 2 ), basisSize );
  LVARRAY_ERROR_IF_NE( twoElectronTerms.size( 3 ), basisSize );

  LVARRAY_ERROR_IF_NE( density.size( 0 ), basisSize );
  LVARRAY_ERROR_IF_NE( density.size( 1 ), basisSize );

  if( !orthogonal )
  {
    LVARRAY_ERROR_IF_NE( overlap.size( 0 ), basisSize );
    LVARRAY_ERROR_IF_NE( overlap.size( 1 ), basisSize );
  }

  T previousEnergy = std::numeric_limits< Real >::max();

  for( int iter = 0; iter < 100; ++iter )
  {
    internal::constructFockOperator(
      fockOperator.toView(),
      oneElectronTerms,
      twoElectronTerms,
      density.toViewConst() );

    T const energy = internal::calculateEnergy(
      fockOperator.toViewConst(),
      oneElectronTerms,
      density.toViewConst() );

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


void AtomicRCSHartreeFock::compute(
  ArrayView2d< double const > const & oneElectronTerms,
  ArrayView4d< std::complex< double > const > const & twoElectronTerms )
{
  TCSCF_MARK_FUNCTION;

  LVARRAY_ERROR_IF_NE( oneElectronTerms.size( 1 ), int( basisFunctions.size() ) );

  LVARRAY_ERROR_IF_NE( twoElectronTerms.size( 0 ), int( basisFunctions.size() ) );
  LVARRAY_ERROR_IF_NE( twoElectronTerms.size( 1 ), int( basisFunctions.size() ) );
  LVARRAY_ERROR_IF_NE( twoElectronTerms.size( 2 ), int( basisFunctions.size() ) );
  LVARRAY_ERROR_IF_NE( twoElectronTerms.size( 3 ), int( basisFunctions.size() ) );

  LVARRAY_ERROR_IF_NE( density.size( 0 ), int( basisFunctions.size() ) );
  LVARRAY_ERROR_IF_NE( density.size( 1 ), int( basisFunctions.size() ) );


  std::complex< double > previousEnergy = std::numeric_limits< Real >::max();

  for( int iter = 0; iter < 100; ++iter )
  {
    internal::constructAtomicFockOperator(
      fockOperator.toView(),
      oneElectronTerms,
      twoElectronTerms,
      density.toViewConst(),
      basisFunctions );

    std::complex< double > const energy = internal::calculateEnergy(
      fockOperator.toViewConst(),
      oneElectronTerms,
      density.toViewConst() );

    LVARRAY_LOG_VAR( energy );
    previousEnergy = energy;

    hermitianEigendecomposition( fockOperator.toView(), eigenvalues.toView() );

    internal::getNewDensity( nElectrons, density, fockOperator.toViewConst() );
  }
}

// Explicit instantiations.
template struct RCSHartreeFock< std::complex< double > >;

} // namespace tcscf
