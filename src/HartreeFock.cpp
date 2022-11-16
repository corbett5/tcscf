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


template< typename T, typename U >
T constructSCFOperator(
  ArrayView3d< T, 1 > const & scfOperator,
  ArrayView2d< U const > const & oneElectronTerms,
  ArrayView3d< T const > const & twoElectronTerms,
  ArrayView3d< T const > const & density )
{
  TCSCF_MARK_FUNCTION;

  IndexType const nBasis = scfOperator.size( 1 );

  LVARRAY_ERROR_IF_NE( scfOperator.size( 0 ), 2 );
  LVARRAY_ERROR_IF_NE( scfOperator.size( 1 ), nBasis );
  LVARRAY_ERROR_IF_NE( scfOperator.size( 2 ), nBasis );

  LVARRAY_ERROR_IF_NE( oneElectronTerms.size( 0 ), nBasis );
  LVARRAY_ERROR_IF_NE( oneElectronTerms.size( 1 ), nBasis );

  LVARRAY_ERROR_IF_NE( twoElectronTerms.size( 0 ), 2 );
  LVARRAY_ERROR_IF_NE( twoElectronTerms.size( 1 ), nBasis );
  LVARRAY_ERROR_IF_NE( twoElectronTerms.size( 2 ), nBasis );

  LVARRAY_ERROR_IF_NE( density.size( 0 ), 2 );
  LVARRAY_ERROR_IF_NE( density.size( 1 ), nBasis );
  LVARRAY_ERROR_IF_NE( density.size( 2 ), nBasis );

  RAJA::ReduceSum< Reduce< ParallelHost >, T > energy( 0 );
  forAll< DefaultPolicy< ParallelHost > >( nBasis * nBasis,
    [=] ( IndexType const ji )
    {
      IndexType const j = ji / nBasis;
      IndexType const i = ji % nBasis;
    
      for( int spin = 0; spin < 2; ++spin )
      {
        scfOperator( spin, j, i ) = oneElectronTerms( j, i ) + twoElectronTerms( spin, j, i );
        energy += density( spin, i, j ) * (oneElectronTerms( j, i ) + twoElectronTerms( spin, j, i ) / 2);
      }
    }
  );

  return energy.get();
}


template< typename T >
void calculateScaledGradients(
  ArraySlice1d< Cartesian< T > > const & scaledGradients,
  ArrayView2d< T const > const & r1BasisValues,
  ArrayView2d< Cartesian< T > const > const & r1BasisGradients,
  ArraySlice2d< T const > const & density )
{
  TCSCF_MARK_SCOPE( "calculateScaledGradients" );

  IndexType const nGridR1 = scaledGradients.size( 0 );
  IndexType const nBasis = r1BasisValues.size( 1 );

  LVARRAY_ERROR_IF_NE( r1BasisValues.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( r1BasisValues.size( 1 ), nBasis );

  LVARRAY_ERROR_IF_NE( r1BasisGradients.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( r1BasisGradients.size( 1 ), nBasis );

  LVARRAY_ERROR_IF_NE( density.size( 0 ), nBasis );
  LVARRAY_ERROR_IF_NE( density.size( 1 ), nBasis );

  forAll< DefaultPolicy< ParallelHost > >( nGridR1,
    [=] ( IndexType const r1Idx )
    {
      Cartesian< T > answer {};
      for( IndexType j = 0; j < nBasis; ++j )
      {
        for( IndexType i = 0; i < nBasis; ++i )
        {
          answer.scaledAdd( density( i, j ) * conj( r1BasisValues( r1Idx, j ) ), r1BasisGradients( r1Idx, i ) );
        }
      }

      scaledGradients[ r1Idx ] = answer;
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


template< typename U, typename V, typename T >
void calculateIaa(
  ArraySlice1d< U > const & Iaa,
  ArrayView3d< V const > const & Iji,
  ArraySlice2d< T const > const & density )
{
  TCSCF_MARK_SCOPE( "calculateIaa" );
  
  IndexType const nGridR1 = Iaa.size();
  IndexType const nBasis = Iji.size( 1 );

  LVARRAY_ERROR_IF_NE( Iaa.size(), nGridR1 );

  LVARRAY_ERROR_IF_NE( Iji.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( Iji.size( 1 ), nBasis );
  LVARRAY_ERROR_IF_NE( Iji.size( 2 ), nBasis );

  LVARRAY_ERROR_IF_NE( density.size( 0 ), nBasis );
  LVARRAY_ERROR_IF_NE( density.size( 1 ), nBasis );


  forAll< DefaultPolicy< ParallelHost > >( nGridR1,
    [=] ( IndexType const r1Idx )
    {
      U result {};
      for( IndexType j = 0; j < nBasis; ++j )
      {
        for( IndexType i = 0; i < nBasis; ++i )
        {
          result += Iji( r1Idx, j, i ) * density( i, j );
        }
      }

      Iaa[ r1Idx ] += result;
    }
  );
}


template< typename T >
void calculateVai(
  ArraySlice2d< T > const & Vai,
  ArrayView3d< Cartesian< T > const > const & Vji,
  ArrayView2d< Cartesian< T > const > const & r1BasisGradients,
  ArraySlice2d< T const > const & density )
{
  TCSCF_MARK_SCOPE( "calculateVai" );

  IndexType const nGridR1 = Vai.size( 0 );
  IndexType const nBasis = Vai.size( 1 );

  LVARRAY_ERROR_IF_NE( Vai.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( Vai.size( 1 ), nBasis );

  LVARRAY_ERROR_IF_NE( Vji.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( Vji.size( 1 ), nBasis );
  LVARRAY_ERROR_IF_NE( Vji.size( 2 ), nBasis );

  LVARRAY_ERROR_IF_NE( r1BasisGradients.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( r1BasisGradients.size( 1 ), nBasis );

  LVARRAY_ERROR_IF_NE( density.size( 0 ), nBasis );
  LVARRAY_ERROR_IF_NE( density.size( 1 ), nBasis );

  forAll< DefaultPolicy< ParallelHost > >( nGridR1,
    [=] ( IndexType const r1Idx )
    {
      for( IndexType j = 0; j < nBasis; ++j )
      {
        Cartesian< T > kSum {};
        for( IndexType k = 0; k < nBasis; ++k )
        {
          kSum.scaledAdd( density( k, j ), r1BasisGradients( r1Idx, k ) );
        }

        for( IndexType i = 0; i < nBasis; ++i )
        {
          Vai( r1Idx, i ) += dot( Vji( r1Idx, j, i ), kSum );
        }
      }
    }
  );
}


template< typename T >
void calculateVja(
  ArraySlice2d< Cartesian< T > > const & Vja,
  ArrayView3d< Cartesian< T > const > const & Vji,
  ArrayView2d< T const > const & r1BasisValues,
  ArraySlice2d< T const > const & density )
{
  TCSCF_MARK_SCOPE( "calculateVja" );

  IndexType const nGridR1 = Vja.size( 0 );
  IndexType const nBasis = Vja.size( 1 );

  LVARRAY_ERROR_IF_NE( Vja.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( Vja.size( 1 ), nBasis );

  LVARRAY_ERROR_IF_NE( Vji.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( Vji.size( 1 ), nBasis );
  LVARRAY_ERROR_IF_NE( Vji.size( 2 ), nBasis );

  LVARRAY_ERROR_IF_NE( r1BasisValues.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( r1BasisValues.size( 1 ), nBasis );

  LVARRAY_ERROR_IF_NE( density.size( 0 ), nBasis );
  LVARRAY_ERROR_IF_NE( density.size( 1 ), nBasis );

  forAll< DefaultPolicy< ParallelHost > >( nGridR1,
    [=] ( IndexType const r1Idx )
    {
      for( IndexType i = 0; i < nBasis; ++i )
      {
        T kSum = 0;
        for( IndexType k = 0; k < nBasis; ++k )
        {
          kSum += density( i, k ) * conj( r1BasisValues( r1Idx, k ) );
        }

        for( IndexType j = 0; j < nBasis; ++j )
        {
          Vja( r1Idx, j ).scaledAdd( kSum, Vji( r1Idx, j, i ) );
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
    internal::calculateIaa( Faa.toSlice(), Fji, density.toSliceConst() );
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
      internal::calculateIaa( Faa[ spin ], Fji, density[ spin ].toSliceConst() );
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
RealType< T > TCHartreeFock< T >::compute(
  bool const orthogonal,
  ArrayView2d< T const > const & overlap,
  ArrayView2d< Real const > const & oneElectronTerms,
  integration::QMCGrid< Real, 3 > const & r1Grid,
  ArrayView3d< Real const > const & FjiSame,
  ArrayView3d< Real const > const & FjiOppo,
  ArrayView3d< Cartesian< T > const > const & VjiSame,
  ArrayView3d< Cartesian< T > const > const & VjiOppo )
{
  TCSCF_MARK_FUNCTION;

  LVARRAY_ERROR_IF_NE( oneElectronTerms.size( 1 ), basisSize );

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

  Array2d< T > const FaaSamePlusOppo( 2, r1Grid.nGrid() );
  Array3d< T > const FaiSame( 2, r1Grid.nGrid(), basisSize );

  Array2d< Cartesian< T > > const VaaSamePlusOppo( 2, r1Grid.nGrid() );
  Array3d< T > const Vai( 2, r1Grid.nGrid(), basisSize );
  Array3d< Cartesian< T > > const Vja( 2, r1Grid.nGrid(), basisSize );

  Array2d< Cartesian< T > > const scaledGradients( 2, r1Grid.nGrid() );

  Array3d< T > const vectorComponent( 2, basisSize, basisSize );
  Array3d< T > const twoElectronTerms( 2, basisSize, basisSize );

  for( _iter = 0; _iter < 100; ++_iter )
  {
    FaaSamePlusOppo.zero();
    FaiSame.zero();

    VaaSamePlusOppo.zero();
    Vai.zero();
    Vja.zero();

    scaledGradients.zero();

    for( int spin = 0; spin < 2; ++spin )
    {
      internal::calculateIaa( FaaSamePlusOppo[ spin ], FjiSame, density[ spin ].toSliceConst() );
      internal::calculateIaa( FaaSamePlusOppo[ spin ], FjiOppo, density[ !spin ].toSliceConst() );

      internal::calculateFai( FaiSame[ spin ], FjiSame, r1Grid.basisValues.toViewConst(), density[ spin ].toSliceConst() );

      internal::calculateIaa( VaaSamePlusOppo[ spin ], VjiSame, density[ spin ].toSliceConst() );
      internal::calculateIaa( VaaSamePlusOppo[ spin ], VjiOppo, density[ !spin ].toSliceConst() );

      internal::calculateVai( Vai[ spin ], VjiSame, r1Grid.basisGradients.toViewConst(), density[ spin ].toSliceConst() );
      internal::calculateVja( Vja[ spin ], VjiSame, r1Grid.basisValues.toViewConst(), density[ spin ].toSliceConst() );

      internal::calculateScaledGradients( scaledGradients[ spin ], r1Grid.basisValues.toViewConst(), r1Grid.basisGradients.toViewConst(), density[ spin ].toSliceConst() );
    }

    fockOperator.zero();
    for( int spin = 0; spin < 2; ++spin )
    {
      TCSCF_MARK_SCOPE( "Constructing the two electron terms" );

      forAll< DefaultPolicy< ParallelHost > >( basisSize * basisSize,
        [&] ( IndexType const ji )
        {
          IndexType const j = ji / basisSize;
          IndexType const i = ji % basisSize;

          T scalar = 0;
          T vector = 0;
          for( IndexType r1Idx = 0; r1Idx < r1Grid.nGrid(); ++r1Idx )
          {
            scalar += conj( r1Grid.basisValues( r1Idx, j ) ) * (FaaSamePlusOppo( spin, r1Idx ) * r1Grid.basisValues( r1Idx, i ) - FaiSame( spin, r1Idx, i ));

            vector += conj( r1Grid.basisValues( r1Idx, j ) ) * (dot( VaaSamePlusOppo( spin, r1Idx ), r1Grid.basisGradients( r1Idx, i ) ) - Vai( spin, r1Idx, i ));

            vector += dot( VjiSame( r1Idx, j, i ), scaledGradients( spin, r1Idx ) );
            
            vector += dot( VjiOppo( r1Idx, j, i ), scaledGradients( !spin, r1Idx ) );

            vector -= dot( Vja( spin, r1Idx, j ), r1Grid.basisGradients( r1Idx, i ) );
          }

          twoElectronTerms( spin, j, i ) = scalar / 2;
          vectorComponent( spin, j, i ) = vector / 2;
        }
      );

      forAll< DefaultPolicy< ParallelHost > >( basisSize * basisSize,
        [&] ( IndexType const ji )
        {
          IndexType const j = ji / basisSize;
          IndexType const i = ji % basisSize;

          twoElectronTerms( spin, j, i ) += vectorComponent( spin, j, i ) - conj( vectorComponent( spin, i, j ) );
        }
      );
    }

    Real const newEnergy = internal::constructSCFOperator(
      fockOperator.toView(),
      oneElectronTerms.toViewConst(), 
      twoElectronTerms.toViewConst(),
      density.toViewConst() ).real();

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
