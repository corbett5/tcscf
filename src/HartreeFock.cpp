#include "HartreeFock.hpp"

#include "caliperInterface.hpp"

#include "dense/eigenDecomposition.hpp"

namespace tcscf
{

namespace internal
{

// TODO: make these member functions
template< typename T, typename U >
void constructFockOperator(
  ArrayView2d< T, 0 > const & fockOperator,
  ArrayView2d< U const > const & oneElectronTerms,
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

/**
 * Assumes eigenvalues and vectors are sorted from least to greatest
 */
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

/**
 * 
 */
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
      fockOperator.toView(),
      oneElectronTerms,
      twoElectronTerms,
      density.toViewConst() );

    Real const newEnergy = internal::calculateEnergy(
      fockOperator.toViewConst(),
      oneElectronTerms,
      density.toViewConst() ).real();

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

    internal::getNewDensity( nElectrons, density, eigenvectors.toViewConst() );
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
    constructFockOperator( oneElectronTerms, twoElectronTerms );

    Real const newEnergy = calculateEnergy( oneElectronTerms ).real();

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

    getNewDensity();
  }

  LVARRAY_ERROR( "Did not converge :(" );
  return std::numeric_limits< Real >::max();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename T >
void UOSHartreeFock< T >::constructFockOperator(
  ArrayView2d< Real const > const & oneElectronTerms,
  ArrayView4d< T const > const & twoElectronTerms )
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
            tmp += (density( 0, a, b ) + density( 1, a, b )) * twoElectronTerms( u, b, v, a ) - density( spin, a, b ) * twoElectronTerms( u, b, a, v );
          }
        }

        fockOperator( spin, u, v ) = oneElectronTerms( u, v ) + tmp;
      }
    }
  }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename T >
T UOSHartreeFock< T >::calculateEnergy( ArrayView2d< Real const > const & oneElectronTerms ) const
{
  TCSCF_MARK_FUNCTION;

  T energy = 0;
  for( int u = 0; u < basisSize; ++u )
  {
    for( int v = 0; v < basisSize; ++v )
    {
      energy += density( 0, v, u ) * (oneElectronTerms( u, v ) + fockOperator( 0, u, v )) +
                density( 1, v, u ) * (oneElectronTerms( u, v ) + fockOperator( 1, u, v ));
    }
  }

  return energy / 2;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename T >
void UOSHartreeFock< T >::getNewDensity()
{
  TCSCF_MARK_FUNCTION;

  for( int spin = 0; spin < 2; ++spin )
  {
    // TODO: replace with with matrix multiplication C C^\dagger
    for( IndexType u = 0; u < basisSize; ++u )
    {
      for( IndexType v = 0; v < basisSize; ++v )
      {
        T tmp = 0;
        for( IndexType a = 0; a < nElectrons[ spin ]; ++a )
        {
          tmp += eigenvectors( spin, u, a ) * conj( eigenvectors( spin, v, a ) );
        }

        density( spin, u, v ) = tmp;
      }
    }
  }
}


////////////////////////////////////////////////////////////////////////////////////////////////////////
// template< typename T >
// RealType< T > RCSNonHermitianHartreeFock< T >::compute(
//   bool const orthogonal,
//   ArrayView2d< T const > const & overlap,
//   ArrayView2d< Real const > const & oneElectronTerms,
//   ArrayView4d< T const > const & twoElectronTerms )
// {
//   TCSCF_MARK_FUNCTION;

//   LVARRAY_ERROR_IF_NE( oneElectronTerms.size( 1 ), basisSize );

//   LVARRAY_ERROR_IF_NE( twoElectronTerms.size( 0 ), basisSize );
//   LVARRAY_ERROR_IF_NE( twoElectronTerms.size( 1 ), basisSize );
//   LVARRAY_ERROR_IF_NE( twoElectronTerms.size( 2 ), basisSize );
//   LVARRAY_ERROR_IF_NE( twoElectronTerms.size( 3 ), basisSize );

//   LVARRAY_ERROR_IF_NE( density.size( 0 ), basisSize );
//   LVARRAY_ERROR_IF_NE( density.size( 1 ), basisSize );

//   if( !orthogonal )
//   {
//     LVARRAY_ERROR_IF_NE( overlap.size( 0 ), basisSize );
//     LVARRAY_ERROR_IF_NE( overlap.size( 1 ), basisSize );
//   }

//   LvArray::dense::EigenDecompositionOptions eigenDecompositionOptions(
//     LvArray::dense::EigenDecompositionOptions::EIGENVALUES_AND_RIGHT_VECTORS );

//   Array1d< int > sortedIndices( basisSize );
//   for( int i = 0; i < basisSize; ++i )
//   {
//     sortedIndices[ i ] = i;
//   }

//   Real energy = std::numeric_limits< Real >::max();

//   for( int iter = 0; iter < 1000; ++iter )
//   {
//     internal::constructFockOperator(
//       fockOperator.toView(),
//       oneElectronTerms,
//       twoElectronTerms,
//       density.toViewConst() );

//     // TODO: Need to calculate the pseudo energy here
//     Real const newEnergy = internal::calculateEnergy(
//       fockOperator.toViewConst(),
//       oneElectronTerms,
//       density.toViewConst() ).real();

//     if( std::abs( (newEnergy - energy) / energy ) < 10 * std::numeric_limits< Real >::epsilon() )
//     {
//       return newEnergy;
//     }

//     energy = newEnergy;

//     if( orthogonal )
//     {
//       LvArray::dense::geev(
//         LvArray::dense::BuiltInBackends::LAPACK,
//         eigenDecompositionOptions,
//         fockOperator.toView(),
//         eigenvalues.toView(),
//         decltype( eigenvectors ) {},
//         eigenvectors.toView(),
//         _workspace );
//     }
//     else
//     {
//       LVARRAY_ERROR( "Generalized eigenvalue problem not yet supported." );
//     }

//     std::sort( sortedIndices.begin(), sortedIndices.end(),
//       [this] ( int const a, int const b )
//       {
//         return std::real( eigenvalues[ a ] ) < std::real( eigenvalues[ b ] );
//       }
//     );

//     // TODO: Then I need to orthonormalize the vectors using either QR or Gram-Schmidt.
//     internal::getNewDensity( nElectrons, density, eigenvectors.toViewConst() );
//   }

//   LVARRAY_ERROR( "Did not converge :(" );
//   return std::numeric_limits< Real >::max();
// }


// /**
//  *
//  */
// template< typename T >
// void getNewDensity(
//   IndexType const nElectrons,
//   ArrayView2d< T > const & density,
//   ArrayView2d< T const, 0 > const & eigenvectors,
//   ArrayView1d< int const > const & indices )
// {
//   TCSCF_MARK_FUNCTION;

//   LVARRAY_ERROR_IF_NE( nElectrons % 2, 0 );
//   IndexType const basisSize = eigenvectors.size( 0 );
  
//   for( IndexType u = 0; u < basisSize; ++u )
//   {
//     for( IndexType v = 0; v < basisSize; ++v )
//     {
//       T tmp = 0;
//       for( IndexType a = 0; a < nElectrons / 2; ++a )
//       {
//         tmp += eigenvectors( u, indices[ a ] ) * conj( eigenvectors( v, indices[ a ] ) );
//       }

//       density( u, v ) = 2 * tmp;
//     }
//   }
// }

// Explicit instantiations.

// TODO: Add support for real matrices
// template struct RCSHartreeFock< float >;
// template struct RCSHartreeFock< double >;

template struct RCSHartreeFock< std::complex< float > >;
template struct RCSHartreeFock< std::complex< double > >;

template struct UOSHartreeFock< std::complex< float > >;
template struct UOSHartreeFock< std::complex< double > >;

} // namespace tcscf
