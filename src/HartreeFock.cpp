#include "HartreeFock.hpp"

#include "caliperInterface.hpp"

#include "orthogonalization.hpp"
#include "integration/integrateAll.hpp"

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

/**
 * 
 */
template< typename T, typename U >
void occupiedOrbitalValues(
  ArrayView2d< decltype( T {} * U {} ) > const & occupiedValues,
  ArrayView2d< T const > const & basisValues,
  ArraySlice2d< U const, 0 > const & eigenvectors )
{
  TCSCF_MARK_SCOPE( "Calculating occupied orbital values" );

  using ResultType = decltype( T {} * U {} );

  IndexType const gridSize = basisValues.size( 0 );
  IndexType const nBasis = basisValues.size( 1 );
  IndexType const nElectrons = occupiedValues.size( 1 );

  LVARRAY_ERROR_IF_NE( occupiedValues.size( 0 ), gridSize );
  LVARRAY_ERROR_IF_NE( eigenvectors.size( 0 ), nBasis );
  LVARRAY_ERROR_IF_NE( eigenvectors.size( 1 ), nElectrons );

  // TODO: replace with matrix vector.
  forAll< DefaultPolicy< ParallelHost > >( gridSize,
    [=] ( IndexType const r1Idx )
    {
      for( IndexType i = 0; i < nElectrons; ++i )
      {
        ResultType value = 0;
        
        for( IndexType j = 0; j < nBasis; ++j )
        {
          value += basisValues( r1Idx, j ) * eigenvectors( j, i );
        }

        occupiedValues( r1Idx, i ) = value;
      }
    }
  );
}

/**
 * 
 */
template< typename T, typename U >
void occupiedOrbitalGradients(
  ArrayView2d< Cartesian< decltype( T {} * U {} ) > > const & occupiedGradients,
  ArrayView2d< Cartesian< T > const > const & basisGradients,
  ArraySlice2d< U const, 0 > const & eigenvectors )
{
  TCSCF_MARK_SCOPE( "Calculating occupied orbital values" );

  using ResultType = decltype( T {} * U {} );

  IndexType const gridSize = basisGradients.size( 0 );
  IndexType const nBasis = basisGradients.size( 1 );
  IndexType const nElectrons = occupiedGradients.size( 1 );

  LVARRAY_ERROR_IF_NE( occupiedGradients.size( 0 ), gridSize );
  LVARRAY_ERROR_IF_NE( eigenvectors.size( 0 ), nBasis );
  LVARRAY_ERROR_IF_NE( eigenvectors.size( 1 ), nElectrons );

  // TODO: replace with matrix vector.
  forAll< DefaultPolicy< ParallelHost > >( gridSize,
    [=] ( IndexType const r1Idx )
    {
      for( IndexType i = 0; i < nElectrons; ++i )
      {
        Cartesian< ResultType > grad {};
        
        for( IndexType j = 0; j < nBasis; ++j )
        {
          grad.scaledAdd( eigenvectors( j, i ), basisGradients( r1Idx, j ) );
        }

        occupiedGradients( r1Idx, i ) = grad;
      }
    }
  );
}

/**
 * TODO: Merge with calculate Vaa
 */
template< typename T, int USD, typename U >
void calculateFaa(
  ArraySlice1d< T, USD > const & Faa,
  ArrayView3d< U const > const & Fji,
  ArraySlice2d< T const > const & density )
{
  TCSCF_MARK_SCOPE( "calculateFaa" );
  
  IndexType const nGridR1 = Faa.size();
  IndexType const nBasis = Fji.size( 1 );

  LVARRAY_ERROR_IF_NE( Faa.size(), nGridR1 );

  LVARRAY_ERROR_IF_NE( Fji.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( Fji.size( 1 ), nBasis );
  LVARRAY_ERROR_IF_NE( Fji.size( 2 ), nBasis );

  LVARRAY_ERROR_IF_NE( density.size( 0 ), nBasis );
  LVARRAY_ERROR_IF_NE( density.size( 1 ), nBasis );


  forAll< DefaultPolicy< ParallelHost > >( nGridR1,
    [=] ( IndexType const r1Idx )
    {
      T result = 0;
      for( IndexType j = 0; j < nBasis; ++j )
      {
        for( IndexType i = 0; i < nBasis; ++i )
        {
          result += density( i, j ) * Fji( r1Idx, j, i );
        }
      }

      Faa[ r1Idx ] = result;
    }
  );
}

template< typename T, int USD, typename U >
void calculateFai(
  ArraySlice2d< T, USD > const & Fai,
  ArrayView3d< U const > const & Fji,
  ArrayView2d< T const > const & r1BasisValues,
  ArraySlice2d< T const > const & density )
{
  TCSCF_MARK_SCOPE( "calculateFai" );
  
  IndexType const nGridR1 = Fai.size( 0 );
  IndexType const nBasis = Fai.size( 1 );

  LVARRAY_ERROR_IF_NE( Fji.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( Fji.size( 1 ), nBasis );
  LVARRAY_ERROR_IF_NE( Fji.size( 2 ), nBasis );

  LVARRAY_ERROR_IF_NE( r1BasisValues.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( r1BasisValues.size( 1 ), nBasis );

  LVARRAY_ERROR_IF_NE( density.size( 0 ), nBasis );
  LVARRAY_ERROR_IF_NE( density.size( 1 ), nBasis );

  forAll< DefaultPolicy< ParallelHost > >( nGridR1,
    [=] ( IndexType const r1Idx )
    {
      for( IndexType j = 0; j < nBasis; ++j )
      {
        T kSum = 0;
        for( IndexType k = 0; k < nBasis; ++k )
        {
          kSum += density( k, j ) * r1BasisValues( r1Idx, k );
        }

        for( IndexType i = 0; i < nBasis; ++i )
        {
          Fai( r1Idx, i ) += kSum * Fji( r1Idx, j, i );
        }
      }
    }
  );
}

template< typename T >
void calculateVaa(
  ArraySlice1d< Cartesian< T > > const & Vaa,
  ArrayView3d< Cartesian< T > const > const & Vji,
  ArraySlice2d< T const > const & density )
{
  TCSCF_MARK_SCOPE( "calculateVaa" );
  
  IndexType const nGridR1 = Vaa.size();
  IndexType const nBasis = Vji.size( 1 );

  LVARRAY_ERROR_IF_NE( Vaa.size(), nGridR1 );

  LVARRAY_ERROR_IF_NE( Vji.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( Vji.size( 1 ), nBasis );
  LVARRAY_ERROR_IF_NE( Vji.size( 2 ), nBasis );

  LVARRAY_ERROR_IF_NE( density.size( 0 ), nBasis );
  LVARRAY_ERROR_IF_NE( density.size( 1 ), nBasis );


  forAll< DefaultPolicy< ParallelHost > >( nGridR1,
    [=] ( IndexType const r1Idx )
    {
      Cartesian< T > result {};
      for( IndexType j = 0; j < nBasis; ++j )
      {
        for( IndexType i = 0; i < nBasis; ++i )
        {
          result.scaledAdd( density( i, j ), Vji( r1Idx, j, i ) );
        }
      }

      Vaa[ r1Idx ] = result;
    }
  );
}

template< typename T >
void calculateVai(
  ArraySlice2d< T > const & Vai,
  ArrayView3d< Cartesian< T > const > const & Vji,
  ArrayView2d< T const, 0 > const & C,
  ArrayView2d< Cartesian< T > const > const & r1OccupiedGradients )
{
  TCSCF_MARK_SCOPE( "calculateVai" );

  IndexType const nGridR1 = Vai.size( 0 );
  IndexType const nBasis = Vai.size( 1 );

  IndexType const nOccupied = C.size( 1 );

  LVARRAY_ERROR_IF_NE( Vai.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( Vai.size( 1 ), nBasis );

  LVARRAY_ERROR_IF_NE( Vji.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( Vji.size( 1 ), nBasis );
  LVARRAY_ERROR_IF_NE( Vji.size( 2 ), nBasis );

  LVARRAY_ERROR_IF_NE( C.size( 0 ), nBasis );
  LVARRAY_ERROR_IF_NE( C.size( 1 ), nOccupied );

  LVARRAY_ERROR_IF_NE( r1OccupiedGradients.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( r1OccupiedGradients.size( 1 ), nOccupied );

  forAll< DefaultPolicy< ParallelHost > >( nGridR1,
    [=] ( IndexType const r1Idx )
    {
      for( IndexType j = 0; j < nBasis; ++j )
      {
        Cartesian< T > aSum {};
        for( IndexType a = 0; a < nOccupied; ++a )
        {
          aSum.scaledAdd( conj( C( j, a ) ), r1OccupiedGradients( r1Idx, a ) );
        }

        for( IndexType i = 0; i < nBasis; ++i )
        {
          Vai( r1Idx, i ) += dot( Vji( r1Idx, j, i ), aSum );
        }
      }
    }
  );
}

template< typename T >
void calculateVja(
  ArraySlice2d< Cartesian< T > > const & Vja,
  ArrayView3d< Cartesian< T > const > const & Vji,
  ArrayView2d< T const, 0 > const & C,
  ArrayView2d< T const > const & r1OccupiedValues )
{
  TCSCF_MARK_SCOPE( "calculateVja" );

  IndexType const nGridR1 = Vja.size( 0 );
  IndexType const nBasis = Vja.size( 1 );

  IndexType const nOccupied = C.size( 1 );

  LVARRAY_ERROR_IF_NE( Vja.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( Vja.size( 1 ), nBasis );

  LVARRAY_ERROR_IF_NE( Vji.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( Vji.size( 1 ), nBasis );
  LVARRAY_ERROR_IF_NE( Vji.size( 2 ), nBasis );

  LVARRAY_ERROR_IF_NE( C.size( 0 ), nBasis );
  LVARRAY_ERROR_IF_NE( C.size( 1 ), nOccupied );

  LVARRAY_ERROR_IF_NE( r1OccupiedValues.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( r1OccupiedValues.size( 1 ), nOccupied );

  forAll< DefaultPolicy< ParallelHost > >( nGridR1,
    [=] ( IndexType const r1Idx )
    {
      for( IndexType i = 0; i < nBasis; ++i )
      {
        T aSum = 0;
        for( IndexType a = 0; a < nOccupied; ++a )
        {
          aSum += C( i, a ) * conj( r1OccupiedValues( r1Idx, a ) );
        }

        for( IndexType j = 0; j < nBasis; ++j )
        {
          Vja( r1Idx, j ).scaledAdd( aSum, Vji( r1Idx, j, i ) );
        }
      }
    }
  );
}

} // namespace internal

////////////////////////////////////////////////////////////////////////////////////////////////////////
template< typename T >
RealType< T > RCSHartreeFock< T >::compute(
  bool const orthogonal,
  ArrayView2d< T const > const & overlap,
  ArrayView2d< Real const > const & oneElectronTerms,
  integration::QMCGrid< Real, 3 > const & r1Grid,
  ArrayView3d< Real const > const & Fji )
{
  TCSCF_MARK_FUNCTION;

  LVARRAY_ERROR_IF_NE( oneElectronTerms.size( 1 ), basisSize );
  LVARRAY_ERROR_IF_NE( r1Grid.nBasis(), basisSize );

  if( !orthogonal )
  {
    LVARRAY_ERROR_IF_NE( overlap.size( 0 ), basisSize );
    LVARRAY_ERROR_IF_NE( overlap.size( 1 ), basisSize );
  }

  LvArray::dense::EigenDecompositionOptions eigenDecompositionOptions(
    LvArray::dense::EigenDecompositionOptions::EIGENVALUES_AND_RIGHT_VECTORS,
    1,
    nElectrons / 2 );

  Real energy = std::numeric_limits< Real >::max();

  Array1d< T > const Faa( r1Grid.nGrid() );
  Array2d< T > const Fai( r1Grid.nGrid(), basisSize );

  for( _iter = 0; _iter < 50; ++_iter )
  {
    Faa.zero();
    Fai.zero();
    internal::calculateFaa( Faa.toSlice(), Fji, density.toSliceConst() );
    internal::calculateFai( Fai.toSlice(), Fji, r1Grid.basisValues.toViewConst(), density.toSliceConst() );

    {
      TCSCF_MARK_SCOPE( "Constructing fock operator" );

      // TODO: Move calculating the energy into this function
      forAll< DefaultPolicy< ParallelHost > >( basisSize * basisSize,
        [&] ( IndexType const ji )
        {
          IndexType const j = ji / basisSize;
          IndexType const i = ji % basisSize;

          T twoElectronContribution = 0;
          for( IndexType r1Idx = 0; r1Idx < r1Grid.nGrid(); ++r1Idx )
          {
            twoElectronContribution += conj( r1Grid.basisValues( r1Idx, j ) ) * (2 * r1Grid.basisValues( r1Idx, i ) * Faa( r1Idx ) - Fai( r1Idx, i ) );
          }

          fockOperator( j, i ) = oneElectronTerms( j, i ) + twoElectronContribution;
        }
      );
    }

    Real const newEnergy = internal::calculateEnergy(
      fockOperator.toSliceConst(),
      fockOperator.toSliceConst(),
      density.toSliceConst(),
      density.toSliceConst(),
      oneElectronTerms.toSliceConst() ).real();

    if( std::abs( (newEnergy - energy) / energy ) < 1e-8 )
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
    nElectrons / 2 );

  Real energy = std::numeric_limits< Real >::max();

  for( _iter = 0; _iter < 50; ++_iter )
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

    if( std::abs( (newEnergy - energy) / energy ) < 1e-8 )
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
  integration::QMCGrid< Real, 3 > const & r1Grid,
  ArrayView3d< Real const > const & Fji )
{
  TCSCF_MARK_FUNCTION;

  LVARRAY_ERROR_IF_NE( oneElectronTerms.size( 1 ), basisSize );

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

  Array2d< T > const Faa( 2, r1Grid.nGrid() );
  Array3d< T > const Fai( 2, r1Grid.nGrid(), basisSize );

  for( _iter = 0; _iter < 50; ++_iter )
  {
    Faa.zero();
    Fai.zero();
    for( int spin = 0; spin < 2; ++spin )
    {
      internal::calculateFaa( Faa[ spin ], Fji, density[ spin ].toSliceConst() );
      internal::calculateFai( Fai[ spin ], Fji, r1Grid.basisValues.toViewConst(), density[ spin ].toSliceConst() );
    }

    for( int spin = 0; spin < 2; ++spin )
    {
      TCSCF_MARK_SCOPE( "Constructing fock operator" );

      // TODO: Move calculating the energy into this function
      forAll< DefaultPolicy< ParallelHost > >( basisSize * basisSize,
        [&] ( IndexType const ji )
        {
          IndexType const j = ji / basisSize;
          IndexType const i = ji % basisSize;

          T twoElectronContribution = 0;
          for( IndexType r1Idx = 0; r1Idx < r1Grid.nGrid(); ++r1Idx )
          {
            twoElectronContribution += conj( r1Grid.basisValues( r1Idx, j ) ) *
              (r1Grid.basisValues( r1Idx, i ) * (Faa( spin, r1Idx ) + Faa( !spin, r1Idx )) - Fai( spin, r1Idx, i ));
          }

          fockOperator( spin, j, i ) = oneElectronTerms( j, i ) + twoElectronContribution;
        }
      );
    }

    Real const newEnergy = internal::calculateEnergy(
      fockOperator[ 0 ].toSliceConst(),
      fockOperator[ 1 ].toSliceConst(),
      density[ 0 ].toSliceConst(),
      density[ 1 ].toSliceConst(),
      oneElectronTerms.toSliceConst() ).real();

    if( std::abs( (newEnergy - energy) / energy ) < 1e-8 )
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

  for( _iter = 0; _iter < 50; ++_iter )
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

    if( std::abs( (newEnergy - energy) / energy ) < 1e-8 )
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

  for( _iter = 0; _iter < 50; ++_iter )
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

      // TODO: Pretty sure this isn't necessary when only considering a two electron system with spin up and spin down.
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
RealType< T > TCHartreeFock< T >::compute(
  bool const orthogonal,
  ArrayView2d< T const > const & overlap,
  ArrayView2d< Real const > const & oneElectronTerms,
  ArrayView4d< T const > const & twoElectronTermsSameSpin,
  ArrayView4d< T const > const & twoElectronTermsOppositeSpin,
  integration::QMCGrid< Real, 3 > const & r1Grid,
  ArrayView3d< Real const > const & FjiSame,
  ArrayView3d< Real const > const & FjiOppo,
  ArrayView3d< Cartesian< T > const > const & VjiSame,
  ArrayView3d< Cartesian< T > const > const & VjiOppo )
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

  Array2d< T > const r1OccupiedValues[ 2 ]{ Array2d< T >( r1Grid.nGrid(), nElectrons[ 0 ] ),
                                            Array2d< T >( r1Grid.nGrid(), nElectrons[ 1 ] ) };

  Array2d< Cartesian< T > > const r1OccupiedGradients[ 2 ]{ Array2d< Cartesian< T > >( r1Grid.nGrid(), nElectrons[ 0 ] ),
                                                            Array2d< Cartesian< T > >( r1Grid.nGrid(), nElectrons[ 1 ] ) };

  Array2d< T > const FaaSame( 2, r1Grid.nGrid() );
  Array2d< T > const FaaOppo( 2, r1Grid.nGrid() );
  
  Array3d< T > const FaiSame( 2, r1Grid.nGrid(), basisSize );

  Array2d< Cartesian< T > > const VaaSame( 2, r1Grid.nGrid() );
  Array2d< Cartesian< T > > const VaaOppo( 2, r1Grid.nGrid() );

  Array3d< T > const Vai( 2, r1Grid.nGrid(), basisSize );
  Array3d< Cartesian< T > > const Vja( 2, r1Grid.nGrid(), basisSize );

  for( _iter = 0; _iter < 100; ++_iter )
  {
    FaaSame.zero();
    FaaOppo.zero();
    FaiSame.zero();

    VaaSame.zero();
    VaaOppo.zero();
    Vai.zero();
    Vja.zero();

    for( int spin = 0; spin < 2; ++spin )
    {
      internal::occupiedOrbitalValues( r1OccupiedValues[ spin ].toView(), r1Grid.basisValues.toViewConst(), occupiedOrbitals[ spin ].toSliceConst() );
      internal::occupiedOrbitalGradients( r1OccupiedGradients[ spin ].toView(), r1Grid.basisGradients.toViewConst(), occupiedOrbitals[ spin ].toSliceConst() );
    
      internal::calculateFaa( FaaSame[ spin ], FjiSame, density[ spin ].toSliceConst() );
      internal::calculateFaa( FaaOppo[ spin ], FjiOppo, density[ spin ].toSliceConst() );

      internal::calculateFai( FaiSame[ spin ], FjiSame, r1Grid.basisValues.toViewConst(), density[ spin ].toSliceConst() );

      internal::calculateVaa( VaaSame[ spin ], VjiSame, density[ spin ].toSliceConst() );
      internal::calculateVaa( VaaOppo[ spin ], VjiOppo, density[ spin ].toSliceConst() );

      internal::calculateVai( Vai[ spin ], VjiSame, occupiedOrbitals[ spin ].toViewConst(), r1OccupiedGradients[ spin ].toViewConst() );
      internal::calculateVja( Vja[ spin ], VjiSame, occupiedOrbitals[ spin ].toViewConst(), r1OccupiedValues[ spin ].toViewConst() );
    }

    // constructFockOperator( oneElectronTerms, twoElectronTermsSameSpin, twoElectronTermsOppositeSpin );

    fockOperator.zero();
    for( int spin = 0; spin < 2; ++spin )
    {
      TCSCF_MARK_SCOPE( "Constructing fock operator" );

      // TODO: Move calculating the energy into this function
      forAll< DefaultPolicy< ParallelHost > >( basisSize * basisSize,
        [&] ( IndexType const ji )
        {
          IndexType const j = ji / basisSize;
          IndexType const i = ji % basisSize;

          T scalar = 0;
          T vector = 0;
          for( IndexType r1Idx = 0; r1Idx < r1Grid.nGrid(); ++r1Idx )
          {
            Cartesian< T > sumOfScaledGradients {};
            for( IndexType a = 0; a < nElectrons[ spin ]; ++a )
            {
              sumOfScaledGradients.scaledAdd( conj( r1OccupiedValues[ spin ]( r1Idx, a ) ), r1OccupiedGradients[ spin ]( r1Idx, a ) );
            }
            
            scalar += conj( r1Grid.basisValues( r1Idx, j ) ) * FaaSame( spin, r1Idx ) * r1Grid.basisValues( r1Idx, i );

            vector += conj( r1Grid.basisValues( r1Idx, j ) ) * dot( VaaSame( spin, r1Idx ), r1Grid.basisGradients( r1Idx, i ) );

            vector += dot( VjiSame( r1Idx, j, i ), sumOfScaledGradients );


            scalar -= conj( r1Grid.basisValues( r1Idx, j ) ) * FaiSame( spin, r1Idx, i );
            
            vector -= conj( r1Grid.basisValues( r1Idx, j ) ) * Vai( spin, r1Idx, i );

            vector -= dot( Vja( spin, r1Idx, j ), r1Grid.basisGradients( r1Idx, i ) );


            Cartesian< T > sumOfScaledGradientsOppo {};
            for( IndexType a = 0; a < nElectrons[ spin ]; ++a )
            {
              sumOfScaledGradientsOppo.scaledAdd( conj( r1OccupiedValues[ !spin ]( r1Idx, a ) ), r1OccupiedGradients[ !spin ]( r1Idx, a ) );
            }

            scalar += conj( r1Grid.basisValues( r1Idx, j ) ) * FaaOppo( !spin, r1Idx ) * r1Grid.basisValues( r1Idx, i );

            vector += conj( r1Grid.basisValues( r1Idx, j ) ) * dot( VaaOppo( !spin, r1Idx ), r1Grid.basisGradients( r1Idx, i ) );

            vector += dot( VjiOppo( r1Idx, j, i ), sumOfScaledGradientsOppo );
          }

          atomicAdd< ParallelHost >( &fockOperator( spin, j, i ), oneElectronTerms( j, i ) + (scalar + vector) / 2 );
          atomicAdd< ParallelHost >( &fockOperator( spin, i, j ), -conj( vector ) / 2 );
        }
      );
    }

    Real const newEnergy = internal::calculateEnergy(
      fockOperator[ 0 ].toSliceConst(),
      fockOperator[ 1 ].toSliceConst(),
      density[ 0 ].toSliceConst(),
      density[ 1 ].toSliceConst(),
      oneElectronTerms.toSliceConst() ).real();

    // TODO: Push the energy calculation into the above
    // Real const newEnergy = calculateEnergy( oneElectronTerms, twoElectronTermsSameSpin, twoElectronTermsOppositeSpin ).real();

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

      // TODO: Pretty sure this isn't necessary when only considering a two electron system with spin up and spin down.
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
