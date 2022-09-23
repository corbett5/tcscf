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

template< typename REAL >
void constructAtomicFockOperator(
  ArrayView2d< std::complex< REAL >, 0 > const & fockOperator,
  ArrayView2d< REAL const > const & oneElectronTerms,
  ArrayView4d< std::complex< REAL > const > const & twoElectronTerms,
  ArrayView2d< std::complex< REAL > const > const & density,
  std::vector< AtomicParams > const & params )
{
  TCSCF_MARK_FUNCTION;

  fockOperator.zero();

  IndexType const basisSize = fockOperator.size( 0 );
  for( IndexType u = 0; u < basisSize; ++u )
  {
    for( IndexType v = 0; v < basisSize; ++v )
    {
      if( params[ u ].l != params[ v ].l ) continue;
      if( params[ u ].m != params[ v ].m ) continue;

      std::complex< REAL > tmp = 0;
      for( IndexType a = 0; a < basisSize; ++a )
      {
        for( IndexType b = 0; b < basisSize; ++b)
        {
          if( params[ a ].l != params[ b ].l ) continue;
          if( params[ a ].m != params[ b ].m ) continue;

          tmp += density( a, b ) * (twoElectronTerms( u, b, v, a ) - twoElectronTerms( u, b, a, v ) / 2);
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

template< typename REAL >
std::complex< REAL > AtomicRCSHartreeFock< REAL >::iteration(
  ArrayView2d< REAL const > const & oneElectronTerms,
  ArrayView4d< std::complex< REAL > const > const & twoElectronTerms )
{
  TCSCF_MARK_FUNCTION;

  LVARRAY_ERROR_IF_NE( oneElectronTerms.size( 1 ), int( params.size() ) );

  LVARRAY_ERROR_IF_NE( twoElectronTerms.size( 0 ), int( params.size() ) );
  LVARRAY_ERROR_IF_NE( twoElectronTerms.size( 1 ), int( params.size() ) );
  LVARRAY_ERROR_IF_NE( twoElectronTerms.size( 2 ), int( params.size() ) );
  LVARRAY_ERROR_IF_NE( twoElectronTerms.size( 3 ), int( params.size() ) );

  LVARRAY_ERROR_IF_NE( density.size( 0 ), int( params.size() ) );
  LVARRAY_ERROR_IF_NE( density.size( 1 ), int( params.size() ) );
  
  internal::constructAtomicFockOperator< Real >(
    fockOperator.toView(),
    oneElectronTerms,
    twoElectronTerms,
    density.toViewConst(),
    params );

  std::complex< Real > const energy = internal::calculateEnergy(
    fockOperator.toViewConst(),
    oneElectronTerms,
    density.toViewConst() );

  hermitianEigendecomposition( fockOperator.toView(), eigenvalues.toView() );

  internal::getNewDensity( nElectrons, density, fockOperator.toViewConst() );

  return energy;
}

template< typename REAL >
std::complex< REAL > AtomicRCSHartreeFock< REAL >::compute(
  ArrayView2d< REAL const > const & oneElectronTerms,
  ArrayView4d< std::complex< REAL > const > const & twoElectronTerms,
  int const maxIter )
{
  TCSCF_MARK_FUNCTION;

  std::complex< Real > previousEnergy = std::numeric_limits< Real >::max();

  for( int iter = 0; iter < maxIter; ++iter )
  {
    previousEnergy = iteration( oneElectronTerms, twoElectronTerms );
  }

  return previousEnergy;
}

// Explicit instantiations.

// TODO: Add support for real matrices
// template struct RCSHartreeFock< float >;
// template struct RCSHartreeFock< double>;

template struct RCSHartreeFock< std::complex< float > >;
template struct RCSHartreeFock< std::complex< double > >;

template struct AtomicRCSHartreeFock< float >;
template struct AtomicRCSHartreeFock< double >;

} // namespace tcscf
