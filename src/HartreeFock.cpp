#include "HartreeFock.hpp"

#include "caliperInterface.hpp"

#include "orthogonalization.hpp"
#include "dense/eigenDecomposition.hpp"

namespace tcscf
{

namespace internal
{

/**
 * TODO: replace with PossibleAliases
 */
template< typename T, typename U >
void constructFockOperator(
  ArraySlice2d< T, 0 > const & fockOperator,
  ArrayView2d< U const > const & oneElectronTerms,
  ArrayView4d< T const > const & twoElectronTerms,
  ArraySlice2d< T const > const & densitySameSpin,
  ArraySlice2d< T const > const & densityOppoSpin )
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
          tmp += (densitySameSpin( a, b ) + densityOppoSpin( a, b )) * twoElectronTerms( u, b, v, a ) -
                 densitySameSpin( a, b ) * twoElectronTerms( u, b, a, v );
        }
      }

      fockOperator( u, v ) = oneElectronTerms( u, v ) + tmp;
    }
  }
}

/**
 * Assumes eigenvalues and vectors are sorted from least to greatest
 */
template< typename T >
void getNewDensity(
  IndexType const nElectrons,
  ArraySlice2d< T > const & density,
  ArraySlice2d< T const, 0 > const & eigenvectors )
{
  TCSCF_MARK_FUNCTION;

  IndexType const basisSize = eigenvectors.size( 0 );
  
  // TODO: replace with with matrix multiplication C C^\dagger
  for( IndexType u = 0; u < basisSize; ++u )
  {
    for( IndexType v = 0; v < basisSize; ++v )
    {
      T tmp = 0;
      for( IndexType a = 0; a < nElectrons; ++a )
      {
        tmp += eigenvectors( u, a ) * conj( eigenvectors( v, a ) );
      }

      density( u, v ) = tmp;
    }
  }
}

/**
 * TODO: replace with PossibleAliases
 */
template< typename T, typename U >
T calculateEnergy( 
  ArraySlice2d< T const, 0 > const & fockOperatorSpinUp,
  ArraySlice2d< T const, 0 > const & fockOperatorSpinDown,
  ArraySlice2d< T const > const & densitySpinUp,
  ArraySlice2d< T const > const & densitySpinDown,
  ArraySlice2d< U const > const & oneElectronTerms )
{
  TCSCF_MARK_FUNCTION;

  IndexType const basisSize = fockOperatorSpinUp.size( 0 );

  T energy = 0;
  for( int u = 0; u < basisSize; ++u )
  {
    for( int v = 0; v < basisSize; ++v )
    {
      energy += densitySpinUp( v, u ) * (oneElectronTerms( u, v ) + fockOperatorSpinUp( u, v )) +
                densitySpinDown( v, u ) * (oneElectronTerms( u, v ) + fockOperatorSpinDown( u, v ));
    }
  }

  return energy / 2;
}

} // namespace internal

////////////////////////////////////////////////////////////////////////////////////////////////////////
template< typename T >
RealType< T > RCSHartreeFock< T >::compute(
  bool const orthogonal,
  ArrayView2d< T const > const & overlap,
  ArrayView2d< Real const > const & oneElectronTerms,
  ArrayView4d< T const > const & twoElectronTerms )
{
  TCSCF_MARK_FUNCTION;

  LVARRAY_ERROR_IF_NE( oneElectronTerms.size( 1 ), basisSize );

  LVARRAY_ERROR_IF_NE( twoElectronTerms.size( 0 ), basisSize );
  LVARRAY_ERROR_IF_NE( twoElectronTerms.size( 1 ), basisSize );
  LVARRAY_ERROR_IF_NE( twoElectronTerms.size( 2 ), basisSize );
  LVARRAY_ERROR_IF_NE( twoElectronTerms.size( 3 ), basisSize );

  if( !orthogonal )
  {
    LVARRAY_ERROR_IF_NE( overlap.size( 0 ), basisSize );
    LVARRAY_ERROR_IF_NE( overlap.size( 1 ), basisSize );
  }

  LvArray::dense::EigenDecompositionOptions eigenDecompositionOptions(
    LvArray::dense::EigenDecompositionOptions::EIGENVALUES_AND_RIGHT_VECTORS,
    1,
    nElectrons );

  Real energy = std::numeric_limits< Real >::max();

  for( int iter = 0; iter < 1000; ++iter )
  {
    internal::constructFockOperator(
      fockOperator.toSlice(),
      oneElectronTerms,
      twoElectronTerms,
      density.toSliceConst(),
      density.toSliceConst() );

    Real const newEnergy = internal::calculateEnergy(
      fockOperator.toSliceConst(),
      fockOperator.toSliceConst(),
      density.toSliceConst(),
      density.toSliceConst(),
      oneElectronTerms.toSliceConst() ).real();

    if( std::abs( (newEnergy - energy) / energy ) < 10 * std::numeric_limits< Real >::epsilon() )
    {
      return newEnergy;
    }

    energy = newEnergy;

    if( orthogonal )
    {
      LvArray::dense::heevr(
        LvArray::dense::BuiltInBackends::LAPACK,
        eigenDecompositionOptions,
        fockOperator.toView(),
        eigenvalues.toView(),
        eigenvectors.toView(),
        _support.toView(),
        _workspace,
        LvArray::dense::SymmetricMatrixStorageType::UPPER_TRIANGULAR );
    }
    else
    {
      LVARRAY_ERROR( "Generalized eigenvalue problem not yet supported." );
    }

    internal::getNewDensity( nElectrons / 2, density.toSlice(), eigenvectors.toSliceConst() );
  }

  LVARRAY_ERROR( "Did not converge :(" );
  return std::numeric_limits< Real >::max();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename T >
RealType< T > UOSHartreeFock< T >::compute(
  bool const orthogonal,
  ArrayView2d< T const > const & overlap,
  ArrayView2d< Real const > const & oneElectronTerms,
  ArrayView4d< T const > const & twoElectronTerms )
{
  TCSCF_MARK_FUNCTION;

  LVARRAY_ERROR_IF_NE( oneElectronTerms.size( 1 ), basisSize );

  LVARRAY_ERROR_IF_NE( twoElectronTerms.size( 0 ), basisSize );
  LVARRAY_ERROR_IF_NE( twoElectronTerms.size( 1 ), basisSize );
  LVARRAY_ERROR_IF_NE( twoElectronTerms.size( 2 ), basisSize );
  LVARRAY_ERROR_IF_NE( twoElectronTerms.size( 3 ), basisSize );

  if( !orthogonal )
  {
    LVARRAY_ERROR_IF_NE( overlap.size( 0 ), basisSize );
    LVARRAY_ERROR_IF_NE( overlap.size( 1 ), basisSize );
  }

  CArray< LvArray::dense::EigenDecompositionOptions, 2 > const eigenDecompositionOptions { {
    { LvArray::dense::EigenDecompositionOptions::EIGENVALUES_AND_RIGHT_VECTORS,
      1,
      nElectrons[ 0 ] },
    { LvArray::dense::EigenDecompositionOptions::EIGENVALUES_AND_RIGHT_VECTORS,
      1,
      nElectrons[ 1 ] }
  } };

  Real energy = std::numeric_limits< Real >::max();

  // TODO: figure out a way to initialize the density in a non-zero manner.

  for( int iter = 0; iter < 1000; ++iter )
  {
    for( int spin = 0; spin < 2; ++spin )
    {
      internal::constructFockOperator(
        fockOperator[ spin ],
        oneElectronTerms,
        twoElectronTerms,
        density[ spin ].toSliceConst(),
        density[ !spin ].toSliceConst() );
    }

    Real const newEnergy = internal::calculateEnergy(
      fockOperator[ 0 ].toSliceConst(),
      fockOperator[ 1 ].toSliceConst(),
      density[ 0 ].toSliceConst(),
      density[ 1 ].toSliceConst(),
      oneElectronTerms.toSliceConst() ).real();

    if( std::abs( (newEnergy - energy) / energy ) < 10 * std::numeric_limits< Real >::epsilon() )
    {
      return newEnergy;
    }

    energy = newEnergy;

    if( orthogonal )
    {
      for( int spin = 0; spin < 2; ++spin )
      {
        LvArray::dense::heevr(
          LvArray::dense::BuiltInBackends::LAPACK,
          eigenDecompositionOptions[ spin ],
          fockOperator[ spin ],
          eigenvalues[ spin ],
          eigenvectors[ spin ],
          _support.toSlice(),
          _workspace,
          LvArray::dense::SymmetricMatrixStorageType::UPPER_TRIANGULAR );
      }
    }
    else
    {
      LVARRAY_ERROR( "Generalized eigenvalue problem not yet supported." );
    }

    internal::getNewDensity( nElectrons[ 0 ], density[ 0 ], eigenvectors[ 0 ].toSliceConst() );
    internal::getNewDensity( nElectrons[ 1 ], density[ 1 ], eigenvectors[ 1 ].toSliceConst() );
  }

  LVARRAY_ERROR( "Did not converge :(" );
  return std::numeric_limits< Real >::max();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename T >
RealType< T > TCHartreeFock< T >::compute(
  bool const orthogonal,
  ArrayView2d< T const > const & overlap,
  ArrayView2d< Real const > const & oneElectronTerms,
  ArrayView4d< T const > const & twoElectronTermsSameSpin,
  ArrayView4d< T const > const & twoElectronTermsOppositeSpin )
{
  TCSCF_MARK_FUNCTION;

  LVARRAY_ERROR_IF_NE( oneElectronTerms.size( 1 ), basisSize );

  LVARRAY_ERROR_IF_NE( twoElectronTermsSameSpin.size( 0 ), basisSize );
  LVARRAY_ERROR_IF_NE( twoElectronTermsSameSpin.size( 1 ), basisSize );
  LVARRAY_ERROR_IF_NE( twoElectronTermsSameSpin.size( 2 ), basisSize );
  LVARRAY_ERROR_IF_NE( twoElectronTermsSameSpin.size( 3 ), basisSize );

  LVARRAY_ERROR_IF_NE( twoElectronTermsOppositeSpin.size( 0 ), basisSize );
  LVARRAY_ERROR_IF_NE( twoElectronTermsOppositeSpin.size( 1 ), basisSize );
  LVARRAY_ERROR_IF_NE( twoElectronTermsOppositeSpin.size( 2 ), basisSize );
  LVARRAY_ERROR_IF_NE( twoElectronTermsOppositeSpin.size( 3 ), basisSize );

  if( !orthogonal )
  {
    LVARRAY_ERROR_IF_NE( overlap.size( 0 ), basisSize );
    LVARRAY_ERROR_IF_NE( overlap.size( 1 ), basisSize );
  }

  LvArray::dense::EigenDecompositionOptions const eigenDecompositionOptions(
    LvArray::dense::EigenDecompositionOptions::EIGENVALUES_AND_RIGHT_VECTORS );

  Real energy = std::numeric_limits< Real >::max();

  // TODO: figure out a way to initialize the density in a non-zero manner.
  Array2d< IndexType > sortedIndices( 2, basisSize );
  for( int i = 0; i < basisSize; ++i )
  {
    sortedIndices( 0, i ) = i;
    sortedIndices( 1, i ) = i;
  }

  for( int iter = 0; iter < 1000; ++iter )
  {
    constructFockOperator( oneElectronTerms, twoElectronTermsSameSpin, twoElectronTermsOppositeSpin );

    Real const newEnergy = calculateEnergy( oneElectronTerms, twoElectronTermsSameSpin, twoElectronTermsOppositeSpin ).real();

    if( std::abs( (newEnergy - energy) / energy ) < 10 * std::numeric_limits< Real >::epsilon() )
    {
      return newEnergy;
    }

    energy = newEnergy;

    if( orthogonal )
    {
      for( int spin = 0; spin < 2; ++spin )
      {
        LvArray::dense::geev(
          LvArray::dense::BuiltInBackends::LAPACK,
          eigenDecompositionOptions,
          fockOperator[ spin ],
          eigenvalues[ spin ],
          eigenvectors[ spin ],
          eigenvectors[ spin ],
          _workspace );
      }
    }
    else
    {
      LVARRAY_ERROR( "Generalized eigenvalue problem not yet supported." );
    }

    for( int spin = 0; spin < 2; ++spin )
    {
      std::sort( sortedIndices[ spin ].begin(), sortedIndices[ spin ].end(),
        [this, spin] ( int const a, int const b )
        {
          return std::real( eigenvalues( spin, a ) ) < std::real( eigenvalues( spin, b ) );
        }
      );

      for( IndexType i = 0; i < nElectrons[ spin ]; ++i )
      {
        occupiedOrbitalPseudoEnergy[ spin ][ i ] = std::real( eigenvalues( spin, sortedIndices( spin, i ) ) );

        for( IndexType j = 0; j < basisSize; ++j )
        {
          occupiedOrbitals[ spin ]( j, i ) = eigenvectors( spin, j, sortedIndices( spin, i ) );
        }
      }

      // orthogonalization::modifiedGramSchmidt( occupiedOrbitals[ spin ] );
    }

    internal::getNewDensity( nElectrons[ 0 ], density[ 0 ], occupiedOrbitals[ 0 ].toSliceConst() );
    internal::getNewDensity( nElectrons[ 1 ], density[ 1 ], occupiedOrbitals[ 1 ].toSliceConst() );
  }

  LVARRAY_ERROR( "Did not converge :(" );
  return std::numeric_limits< Real >::max();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename T >
void TCHartreeFock< T >::constructFockOperator(
  ArrayView2d< Real const > const & oneElectronTerms,
  ArrayView4d< T const > const & twoElectronTermsSameSpin,
  ArrayView4d< T const > const & twoElectronTermsOppositeSpin )
{
  TCSCF_MARK_FUNCTION;

  for( int spin = 0; spin < 2; ++spin )
  {
    for( IndexType u = 0; u < basisSize; ++u )
    {
      for( IndexType v = 0; v < basisSize; ++v )
      {
        T tmp = 0;
        for( IndexType a = 0; a < basisSize; ++a )
        {
          for( IndexType b = 0; b < basisSize; ++b)
          {
            tmp += density( spin, a, b ) * (twoElectronTermsSameSpin( u, b, v, a ) - twoElectronTermsSameSpin( u, b, a, v ));
            tmp += density( 1 - spin, a, b ) * twoElectronTermsOppositeSpin( u, b, v, a );
          }
        }

        fockOperator( spin, u, v ) = oneElectronTerms( u, v ) + tmp / 2;
      }
    }
  }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename T >
T TCHartreeFock< T >::calculateEnergy(
  ArrayView2d< Real const > const & oneElectronTerms,
  ArrayView4d< T const > const & twoElectronTermsSameSpin,
  ArrayView4d< T const > const & twoElectronTermsOppositeSpin ) const
{
  TCSCF_MARK_FUNCTION;

  T energy = 0;
  for( IndexType a = 0; a < basisSize; ++a )
  {
    for( IndexType b = 0; b < basisSize; ++b )
    {
      energy += density( 0, b, a ) * (2 * oneElectronTerms( a, b ) + fockOperator( 0, a, b ));
      energy += density( 1, b, a ) * (2 * oneElectronTerms( a, b ) + fockOperator( 1, a, b ));
    }
  }

  T sameSpinContrib = 0;
  for( IndexType a = 0; a < basisSize; ++a )
  {
    for( IndexType b = 0; b < basisSize; ++b )
    {
      for( IndexType c = 0; c < basisSize; ++c )
      {
        for( IndexType d = 0; d < basisSize; ++d )
        {
          T const totalDensity = density( 0, b, a ) * density( 0, d, c ) + density( 1, b, a ) * density( 1, d, c );
          sameSpinContrib += totalDensity * (twoElectronTermsSameSpin(a, c, b, d) - twoElectronTermsSameSpin(a, c, d, b));
        }
      }
    }
  }
  energy += sameSpinContrib / 4;

  T oppoSpinContrib = 0;
  for( IndexType a = 0; a < basisSize; ++a )
  {
    for( IndexType b = 0; b < basisSize; ++b )
    {
      for( IndexType c = 0; c < basisSize; ++c )
      {
        for( IndexType d = 0; d < basisSize; ++d )
        {
          T const totalDensity = density( 0, b, a ) * density( 1, d, c ) + density( 1, b, a ) * density( 0, d, c );
          oppoSpinContrib += totalDensity * twoElectronTermsOppositeSpin( a, c, b, d );
        }
      }
    }
  }
  energy += oppoSpinContrib / 4;

  return energy / 3;
}

// Explicit instantiations.

// TODO: Add support for real matrices
// template struct RCSHartreeFock< float >;
// template struct RCSHartreeFock< double >;

template struct RCSHartreeFock< std::complex< float > >;
template struct RCSHartreeFock< std::complex< double > >;

template struct UOSHartreeFock< std::complex< float > >;
template struct UOSHartreeFock< std::complex< double > >;

template struct TCHartreeFock< std::complex< float > >;
template struct TCHartreeFock< std::complex< double > >;

} // namespace tcscf
