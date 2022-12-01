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
  ArrayView3d< T const > const & threeElectronTerms,
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

  LVARRAY_ERROR_IF_NE( threeElectronTerms.size( 0 ), 2 );
  LVARRAY_ERROR_IF_NE( threeElectronTerms.size( 1 ), nBasis );
  LVARRAY_ERROR_IF_NE( threeElectronTerms.size( 2 ), nBasis );

  LVARRAY_ERROR_IF_NE( density.size( 0 ), 2 );
  LVARRAY_ERROR_IF_NE( density.size( 1 ), nBasis );
  LVARRAY_ERROR_IF_NE( density.size( 2 ), nBasis );

  RAJA::ReduceSum< Reduce< ParallelHost >, T > energy( 0 );
  RAJA::ReduceSum< Reduce< ParallelHost >, T > energyThreeTermsP( 0 );
  RAJA::ReduceSum< Reduce< ParallelHost >, T > energyThreeTermsM( 0 );
  forAll< DefaultPolicy< ParallelHost > >( nBasis * nBasis,
    [=] ( IndexType const ji )
    {
      IndexType const j = ji / nBasis;
      IndexType const i = ji % nBasis;
    
      for( int spin = 0; spin < 2; ++spin )
      {
        scfOperator( spin, j, i ) = oneElectronTerms( j, i ) + twoElectronTerms( spin, j, i ); // - threeElectronTerms( spin, j, i );
        energy += density( spin, i, j ) * (oneElectronTerms( j, i ) + twoElectronTerms( spin, j, i ) / 2 - threeElectronTerms( spin, j, i ) / 3);
        
        T const threeContrib = -density( spin, i, j ) * threeElectronTerms( spin, j, i ) / 3;
        if( threeContrib.real() < 0 )
        {
          energyThreeTermsM += threeContrib;
        }
        else
        {
          energyThreeTermsP += threeContrib;
        }
      }
    }
  );

  auto const e3p = energyThreeTermsP.get().real();
  auto const e3m = energyThreeTermsM.get().real();
  auto const e3 = e3p + e3m;
  LVARRAY_LOG( "energy = " << energy.get().real() << ", e3 = " << e3 << ", e3+ = " << e3p << ", e3- = " << e3m );
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


template< typename T >
void calculateVjb(
  ArrayView3d< Cartesian< T > > const & VjbSame,
  ArrayView3d< Cartesian< T > > const & VjbOppo,
  ArrayView3d< Cartesian< T > const > const & VjiSame,
  ArrayView3d< Cartesian< T > const > const & VjiOppo,
  ArrayView2d< T const, 0 > const & eigenvectors )
{
  TCSCF_MARK_FUNCTION;

  IndexType const nGridR1 = VjbSame.size( 0 );
  IndexType const nBasis = VjbSame.size( 1 );
  IndexType const nOccupied = VjbSame.size( 2 );

  LVARRAY_ERROR_IF_NE( VjbSame.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( VjbSame.size( 1 ), nBasis );
  LVARRAY_ERROR_IF_NE( VjbSame.size( 2 ), nOccupied );

  LVARRAY_ERROR_IF_NE( VjbOppo.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( VjbOppo.size( 1 ), nBasis );
  LVARRAY_ERROR_IF_NE( VjbOppo.size( 2 ), nOccupied );

  LVARRAY_ERROR_IF_NE( VjiSame.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( VjiSame.size( 1 ), nBasis );
  LVARRAY_ERROR_IF_NE( VjiSame.size( 2 ), nBasis );

  LVARRAY_ERROR_IF_NE( VjiOppo.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( VjiOppo.size( 1 ), nBasis );
  LVARRAY_ERROR_IF_NE( VjiOppo.size( 2 ), nBasis );

  LVARRAY_ERROR_IF_NE( eigenvectors.size( 0 ), nBasis );
  LVARRAY_ERROR_IF_NE( eigenvectors.size( 1 ), nOccupied );

  // TODO: replace with a single matrix matrix multiplication.
  forAll< DefaultPolicy< ParallelHost > >( nGridR1,
    [=] ( IndexType const r1Idx )
    {
      for( IndexType j = 0; j < nBasis; ++j )
      {
        for( IndexType b = 0; b < nOccupied; ++b )
        {
          Cartesian< T > resultSame {};
          Cartesian< T > resultOppo {};
          for( IndexType i = 0; i < nBasis; ++i )
          {
            resultSame.scaledAdd( eigenvectors( i, b ), VjiSame( r1Idx, j, i ) );
            resultOppo.scaledAdd( eigenvectors( i, b ), VjiOppo( r1Idx, j, i ) );
          }

          VjbSame( r1Idx, j, b ) = resultSame;
          VjbOppo( r1Idx, j, b ) = resultOppo;
        }
      }
    }
  );
}


template< typename T, typename REAL >
void calculateVab(
  ArrayView3d< Cartesian< T > > const & VabSame,
  ArrayView3d< Cartesian< T > > const & VabOppo,
  ArrayView1d< REAL > const & sumOfNormsVabSame,
  ArrayView1d< REAL > const & sumOfNormsVabOppo,
  ArrayView3d< Cartesian< T > const > const & VjbSame,
  ArrayView3d< Cartesian< T > const > const & VjbOppo,
  ArrayView2d< T const, 0 > const & eigenvectors )
{
  TCSCF_MARK_FUNCTION;

  IndexType const nGridR1 = VabSame.size( 0 );
  IndexType const nOccupied = VabSame.size( 1 );
  IndexType const nBasis = VjbSame.size( 1 );

  LVARRAY_ERROR_IF_NE( VabSame.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( VabSame.size( 1 ), nOccupied );
  LVARRAY_ERROR_IF_NE( VabSame.size( 2 ), nOccupied );

  LVARRAY_ERROR_IF_NE( VabOppo.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( VabOppo.size( 1 ), nOccupied );
  LVARRAY_ERROR_IF_NE( VabOppo.size( 2 ), nOccupied );

  LVARRAY_ERROR_IF_NE( VjbSame.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( VjbSame.size( 1 ), nBasis );
  LVARRAY_ERROR_IF_NE( VjbSame.size( 2 ), nOccupied );

  LVARRAY_ERROR_IF_NE( VjbOppo.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( VjbOppo.size( 1 ), nBasis );
  LVARRAY_ERROR_IF_NE( VjbOppo.size( 2 ), nOccupied );

  LVARRAY_ERROR_IF_NE( eigenvectors.size( 0 ), nBasis );
  LVARRAY_ERROR_IF_NE( eigenvectors.size( 1 ), nOccupied );

  sumOfNormsVabSame.zero();
  sumOfNormsVabOppo.zero();

  // TODO: replace with a single matrix matrix multiplication.
  forAll< DefaultPolicy< ParallelHost > >( nGridR1,
    [=] ( IndexType const r1Idx )
    {
      for( IndexType a = 0; a < nOccupied; ++a )
      {
        for( IndexType b = 0; b < nOccupied; ++b )
        {
          Cartesian< T > resultSame {};
          Cartesian< T > resultOppo {};
          for( IndexType j = 0; j < nBasis; ++j )
          {
            resultSame.scaledAdd( conj( eigenvectors( j, a ) ), VjbSame( r1Idx, j, b ) );
            resultOppo.scaledAdd( conj( eigenvectors( j, a ) ), VjbOppo( r1Idx, j, b ) );
          }

          VabSame( r1Idx, a, b ) = resultSame;
          VabOppo( r1Idx, a, b ) = resultOppo;

          sumOfNormsVabSame( r1Idx ) += resultSame.norm();
          sumOfNormsVabOppo( r1Idx ) += resultOppo.norm();
        }
      }
    }
  );
}

template< typename T >
void calculateVTilde(
  ArrayView2d< Cartesian< T > > const & VTilde,
  ArrayView3d< Cartesian< T > const > const & VjiSame,
  ArrayView3d< Cartesian< T > const > const & VjiOppo,
  ArrayView3d< T const > const & density )
{
  TCSCF_MARK_FUNCTION;

  IndexType const nGridR1 = VTilde.size( 1 );
  IndexType const nBasis = VjiSame.size( 1 );

  LVARRAY_ERROR_IF_NE( VTilde.size( 0 ), 2 );
  LVARRAY_ERROR_IF_NE( VTilde.size( 1 ), nGridR1 );

  LVARRAY_ERROR_IF_NE( VjiSame.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( VjiSame.size( 1 ), nBasis );
  LVARRAY_ERROR_IF_NE( VjiSame.size( 2 ), nBasis );

  LVARRAY_ERROR_IF_NE( VjiOppo.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( VjiOppo.size( 1 ), nBasis );
  LVARRAY_ERROR_IF_NE( VjiOppo.size( 2 ), nBasis );

  LVARRAY_ERROR_IF_NE( density.size( 0 ), 2 );
  LVARRAY_ERROR_IF_NE( density.size( 1 ), nBasis );
  LVARRAY_ERROR_IF_NE( density.size( 2 ), nBasis );

  forAll< DefaultPolicy< ParallelHost > >( nGridR1,
    [=] ( IndexType const r1Idx )
    {
      Cartesian< T > result[ 2 ] {};
      for( IndexType j = 0; j < nBasis; ++j )
      {
        for( IndexType i = 0; i < nBasis; ++i )
        {
          for( int spin = 0; spin < 2; ++spin )
          {
            result[ spin ].scaledAdd( density( spin, i, j ), VjiSame( r1Idx, j, i ) );
            result[ spin ].scaledAdd( density( !spin, i, j ), VjiOppo( r1Idx, j, i ) );
          }
        }
      }

      VTilde( 0, r1Idx ) = result[ 0 ];
      VTilde( 1, r1Idx ) = result[ 1 ];
    }
  );
}


template< typename T >
void calculateVTildeSubI(
  ArrayView3d< Cartesian< T > > const & VTildeSubI,
  ArrayView3d< Cartesian< T > const > const & VjiSame,
  ArrayView2d< T const > const & r1BasisValues,
  ArrayView3d< T const > const & density )
{
  TCSCF_MARK_FUNCTION;

  IndexType const nGridR1 = VTildeSubI.size( 1 );
  IndexType const nBasis = VTildeSubI.size( 2 );

  LVARRAY_ERROR_IF_NE( VTildeSubI.size( 0 ), 2 );
  LVARRAY_ERROR_IF_NE( VTildeSubI.size( 1 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( VTildeSubI.size( 2 ), nBasis );

  LVARRAY_ERROR_IF_NE( VjiSame.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( VjiSame.size( 1 ), nBasis );
  LVARRAY_ERROR_IF_NE( VjiSame.size( 2 ), nBasis );

  LVARRAY_ERROR_IF_NE( r1BasisValues.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( r1BasisValues.size( 1 ), nBasis );

  LVARRAY_ERROR_IF_NE( density.size( 0 ), 2 );
  LVARRAY_ERROR_IF_NE( density.size( 1 ), nBasis );
  LVARRAY_ERROR_IF_NE( density.size( 2 ), nBasis );

  VTildeSubI.zero();
  forAll< DefaultPolicy< ParallelHost > >( nGridR1,
    [=] ( IndexType const r1Idx )
    {
      for( IndexType k = 0; k < nBasis; ++k )
      {
        T ellSum[ 2 ] {};
        for( IndexType ell = 0; ell < nBasis; ++ell )
        {
          for( int spin = 0; spin < 2; ++spin )
          {
            ellSum[ spin ] += density( spin, ell, k ) * r1BasisValues( r1Idx, ell );
          }
        }

        for( IndexType i = 0; i < nBasis; ++i )
        {
          for( int spin = 0; spin < 2; ++spin )
          {
            VTildeSubI( spin, r1Idx, i ).scaledAdd( ellSum[ spin ], VjiSame( r1Idx, k, i ) );
          }
        }
      }
    }
  );
}


template< typename T >
void calculateSumOverb_Vbi_dot_VTildeb(
  ArrayView3d< T > const & SumOverb_Vbi_dot_VTildeb,
  ArrayView3d< Cartesian< T > const > const & VTildeSubI,
  ArrayView3d< Cartesian< T > const > const & VjiSame,
  ArrayView3d< T const > const & density )
{
  TCSCF_MARK_FUNCTION;

  IndexType const nGridR1 = SumOverb_Vbi_dot_VTildeb.size( 1 );
  IndexType const nBasis = SumOverb_Vbi_dot_VTildeb.size( 2 );

  LVARRAY_ERROR_IF_NE( SumOverb_Vbi_dot_VTildeb.size( 0 ), 2 );
  LVARRAY_ERROR_IF_NE( SumOverb_Vbi_dot_VTildeb.size( 1 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( SumOverb_Vbi_dot_VTildeb.size( 2 ), nBasis );

  LVARRAY_ERROR_IF_NE( VTildeSubI.size( 0 ), 2 );
  LVARRAY_ERROR_IF_NE( VTildeSubI.size( 1 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( VTildeSubI.size( 2 ), nBasis );

  LVARRAY_ERROR_IF_NE( VjiSame.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( VjiSame.size( 1 ), nBasis );
  LVARRAY_ERROR_IF_NE( VjiSame.size( 2 ), nBasis );

  LVARRAY_ERROR_IF_NE( density.size( 0 ), 2 );
  LVARRAY_ERROR_IF_NE( density.size( 1 ), nBasis );
  LVARRAY_ERROR_IF_NE( density.size( 2 ), nBasis );

  SumOverb_Vbi_dot_VTildeb.zero();
  forAll< DefaultPolicy< ParallelHost > >( nGridR1,
    [=] ( IndexType const r1Idx )
    {
      for( IndexType k = 0; k < nBasis; ++k )
      {
        Cartesian< T > ellSum[ 2 ] {};
        for( IndexType ell = 0; ell < nBasis; ++ell )
        {
          for( int spin = 0; spin < 2; ++spin )
          {
            ellSum[ spin ].scaledAdd( density( spin, ell, k ), VTildeSubI( spin, r1Idx, ell ) );
          }
        }

        for( IndexType i = 0; i < nBasis; ++i )
        {
          for( int spin = 0; spin < 2; ++spin )
          {
            SumOverb_Vbi_dot_VTildeb( spin, r1Idx, i ) += dot( VjiSame( r1Idx, k, i ), ellSum[ spin ] );
          }
        }
      }
    }
  );
}

template< typename T >
void calculateOccupiedElectronDensity(
  ArrayView2d< RealType< T > > const & occupiedElectronDensity,
  ArrayView2d< T const > const & basisValues,
  ArrayView3d< T const > const & density )
{
  IndexType const nGridR1 = occupiedElectronDensity.size( 1 );
  IndexType const nBasis = basisValues.size( 1 );

  LVARRAY_ERROR_IF_NE( occupiedElectronDensity.size( 0 ), 2 );
  LVARRAY_ERROR_IF_NE( occupiedElectronDensity.size( 1 ), nGridR1 );

  LVARRAY_ERROR_IF_NE( basisValues.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( basisValues.size( 1 ), nBasis );

  LVARRAY_ERROR_IF_NE( density.size( 0 ), 2 );
  LVARRAY_ERROR_IF_NE( density.size( 1 ), nBasis );
  LVARRAY_ERROR_IF_NE( density.size( 2 ), nBasis );

  forAll< DefaultPolicy< ParallelHost > >( nGridR1,
    [=] ( IndexType const r1Idx )
    {
      T curDensity[ 2 ] {};
      for( IndexType k = 0; k < nBasis; ++k )
      {
        for( IndexType ell = 0; ell < nBasis; ++ell )
        {
          for( int spin = 0; spin < 2; ++spin )
          {
            curDensity[ spin ] += density( spin, ell, k ) * conj( basisValues( r1Idx, k ) ) * basisValues( r1Idx, ell );
          }
        }
      }

      for( int spin = 0; spin < 2; ++spin )
      {
        LVARRAY_ERROR_IF_GT( std::abs( curDensity[ spin ].imag() ), 1e-9 );
        occupiedElectronDensity( spin, r1Idx ) = curDensity[ spin ].real();
      }
    }
  );
}


template< typename T >
void calculateSumOverb_conjVTildeb_occb(
  ArrayView2d< Cartesian< T > > const & SumOverb_conjVTildeb_occb,
  ArrayView3d< Cartesian< T > const > const & VTildeSubI,
  ArrayView2d< T const > const & basisValues,
  ArrayView3d< T const > const & density )
{
  IndexType const nGridR1 = SumOverb_conjVTildeb_occb.size( 1 );
  IndexType const nBasis = VTildeSubI.size( 2 );

  LVARRAY_ERROR_IF_NE( SumOverb_conjVTildeb_occb.size( 0 ), 2 );
  LVARRAY_ERROR_IF_NE( SumOverb_conjVTildeb_occb.size( 1 ), nGridR1 );

  LVARRAY_ERROR_IF_NE( VTildeSubI.size( 0 ), 2 );
  LVARRAY_ERROR_IF_NE( VTildeSubI.size( 1 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( VTildeSubI.size( 2 ), nBasis );

  LVARRAY_ERROR_IF_NE( basisValues.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( basisValues.size( 1 ), nBasis );

  LVARRAY_ERROR_IF_NE( density.size( 0 ), 2 );
  LVARRAY_ERROR_IF_NE( density.size( 1 ), nBasis );
  LVARRAY_ERROR_IF_NE( density.size( 2 ), nBasis );

  forAll< DefaultPolicy< ParallelHost > >( nGridR1,
    [=] ( IndexType const r1Idx )
    {
      Cartesian< T > curSum[ 2 ] {};
      for( IndexType k = 0; k < nBasis; ++k )
      {
        for( IndexType ell = 0; ell < nBasis; ++ell )
        {
          for( int spin = 0; spin < 2; ++spin )
          {
            curSum[ spin ].scaledAdd( density( spin, ell, k ) * basisValues( r1Idx, ell ), conj( VTildeSubI( spin, r1Idx, k ) ) );
          }
        }
      }

      for( int spin = 0; spin < 2; ++spin )
      {
        SumOverb_conjVTildeb_occb( spin, r1Idx ) = curSum[ spin ];
      }
    }
  );
}

#define CHECK_BOUNDS_1( array, size0 ) \
  LVARRAY_ERROR_IF_NE( array.size( 0 ), size0 )

#define CHECK_BOUNDS_2( array, size0, size1 ) \
  LVARRAY_ERROR_IF_NE( array.size( 0 ), size0 ); \
  LVARRAY_ERROR_IF_NE( array.size( 1 ), size1 )

#define CHECK_BOUNDS_3( array, size0, size1, size2 ) \
  LVARRAY_ERROR_IF_NE( array.size( 0 ), size0 ); \
  LVARRAY_ERROR_IF_NE( array.size( 1 ), size1 ); \
  LVARRAY_ERROR_IF_NE( array.size( 2 ), size2 )

#define CHECK_BOUNDS_6( array, size0, size1, size2, size3, size4, size5 ) \
  LVARRAY_ERROR_IF_NE( array.size( 0 ), size0 ); \
  LVARRAY_ERROR_IF_NE( array.size( 1 ), size1 ); \
  LVARRAY_ERROR_IF_NE( array.size( 2 ), size2 ); \
  LVARRAY_ERROR_IF_NE( array.size( 3 ), size3 ); \
  LVARRAY_ERROR_IF_NE( array.size( 4 ), size4 ); \
  LVARRAY_ERROR_IF_NE( array.size( 5 ), size5 )


template< typename T, typename REAL >
void computeQ(
  ArrayView6d< T > const & QSameSame,
  ArrayView6d< T > const & QSameOppo, 
  ArrayView6d< T > const & QOppoOppo,
  ArrayView3d< Cartesian< T > const > const & VjiSame, 
  ArrayView3d< Cartesian< T > const > const & VjiOppo,
  ArrayView1d< REAL const > const & r1Weights,
  ArrayView2d< T const > const & r1BasisValues )
{
  using Real = REAL;

  IndexType const nBasis = QSameSame.size( 0 );
  IndexType const nGridR1 = VjiSame.size( 0 );

  CHECK_BOUNDS_6( QSameSame, nBasis, nBasis, nBasis, nBasis, nBasis, nBasis );
  CHECK_BOUNDS_6( QSameOppo, nBasis, nBasis, nBasis, nBasis, nBasis, nBasis );
  CHECK_BOUNDS_6( QOppoOppo, nBasis, nBasis, nBasis, nBasis, nBasis, nBasis );

  CHECK_BOUNDS_3( VjiSame, nGridR1, nBasis, nBasis );
  CHECK_BOUNDS_3( VjiOppo, nGridR1, nBasis, nBasis );

  CHECK_BOUNDS_1( r1Weights, nGridR1 );

  CHECK_BOUNDS_2( r1BasisValues, nGridR1, nBasis );

  using PolicyType = ParallelHost;
  forAll< DefaultPolicy< PolicyType > >( nGridR1,
    [=] ( IndexType const r1Idx )
    {
      Real const r1Weight = r1Weights( r1Idx );

      for( IndexType j = 0; j < nBasis; ++j )
      {
        for( IndexType p = 0; p < nBasis; ++p )
        {
          for( IndexType r = 0; r < nBasis; ++r )
          {
            for( IndexType i = 0; i < nBasis; ++i )
            {
              for( IndexType q = 0; q < nBasis; ++q )
              {
                for( IndexType s = 0; s < nBasis; ++s )
                {
                  T const ss123 = conj( r1BasisValues( r1Idx, j ) ) * dot( VjiSame( r1Idx, p, q ), VjiSame( r1Idx, r, s ) ) * r1BasisValues( r1Idx, i );
                  T const so123 = conj( r1BasisValues( r1Idx, j ) ) * dot( VjiSame( r1Idx, p, q ), VjiOppo( r1Idx, r, s ) ) * r1BasisValues( r1Idx, i );
                  T const oo123 = conj( r1BasisValues( r1Idx, j ) ) * dot( VjiOppo( r1Idx, p, q ), VjiOppo( r1Idx, r, s ) ) * r1BasisValues( r1Idx, i );

                  T const ss213 = conj( r1BasisValues( r1Idx, p ) ) * dot( VjiSame( r1Idx, j, i ), VjiSame( r1Idx, r, s ) ) * r1BasisValues( r1Idx, q );
                  T const so213 = conj( r1BasisValues( r1Idx, p ) ) * dot( VjiSame( r1Idx, j, i ), VjiOppo( r1Idx, r, s ) ) * r1BasisValues( r1Idx, q );
                  T const oo213 = conj( r1BasisValues( r1Idx, p ) ) * dot( VjiOppo( r1Idx, j, i ), VjiOppo( r1Idx, r, s ) ) * r1BasisValues( r1Idx, q );

                  T const ss312 = conj( r1BasisValues( r1Idx, r ) ) * dot( VjiSame( r1Idx, j, i ), VjiSame( r1Idx, p, q ) ) * r1BasisValues( r1Idx, s );
                  T const so312 = conj( r1BasisValues( r1Idx, r ) ) * dot( VjiSame( r1Idx, j, i ), VjiOppo( r1Idx, p, q ) ) * r1BasisValues( r1Idx, s );
                  T const oo312 = conj( r1BasisValues( r1Idx, r ) ) * dot( VjiOppo( r1Idx, j, i ), VjiOppo( r1Idx, p, q ) ) * r1BasisValues( r1Idx, s );


                  atomicAdd< PolicyType >( &QSameSame( j, p, r, i, q, s), r1Weight * (ss123 + ss213 + ss312) );
                  atomicAdd< PolicyType >( &QSameOppo( j, p, r, i, q, s), r1Weight * (so123 + so213 + so312) );
                  atomicAdd< PolicyType >( &QOppoOppo( j, p, r, i, q, s), r1Weight * (oo123 + oo213 + oo312) );
                }
              }
            }
          }
        }
      }
    }
  );
}


template< typename T >
void assembleG(
  ArrayView3d< T > const & G_3New,
  ArrayView6d< T const > const & QSameSame,
  ArrayView6d< T const > const & QSameOppo, 
  ArrayView6d< T const > const & QOppoOppo,
  ArrayView3d< T const > const & density )
{
  IndexType const nBasis = G_3New.size( 1 );

  CHECK_BOUNDS_3( G_3New, 2, nBasis, nBasis );

  CHECK_BOUNDS_6( QSameSame, nBasis, nBasis, nBasis, nBasis, nBasis, nBasis );
  CHECK_BOUNDS_6( QSameOppo, nBasis, nBasis, nBasis, nBasis, nBasis, nBasis );
  CHECK_BOUNDS_6( QOppoOppo, nBasis, nBasis, nBasis, nBasis, nBasis, nBasis );

  CHECK_BOUNDS_3( density, 2, nBasis, nBasis );

  forAll< DefaultPolicy< ParallelHost > >( nBasis * nBasis,
    [&] ( IndexType const ji )
    {
      IndexType const j = ji / nBasis;
      IndexType const i = ji % nBasis;

      T value[ 2 ] {};
      for( IndexType p = 0; p < nBasis; ++p )
      {
        for( IndexType q = 0; q < nBasis; ++q )
        {
          for( IndexType r = 0; r < nBasis; ++r )
          {
            for( IndexType s = 0; s < nBasis; ++s )
            {
              LVARRAY_ERROR_IF_GT( std::abs( 2 * QSameSame( j, p, r, i, q, s ) - QSameOppo( j, p, r, i, q, s ) ), 1e-17 );
              LVARRAY_ERROR_IF_GT( std::abs( 4 * QSameSame( j, p, r, i, q, s ) - QOppoOppo( j, p, r, i, q, s ) ), 1e-17 );

              LVARRAY_ERROR_IF_GT( std::abs( QSameSame( j, p, r, i, q, s ) - QSameSame( j, r, p, i, s, q ) ), 1e-17 );
              LVARRAY_ERROR_IF_GT( std::abs( QSameOppo( j, p, r, i, q, s ) - QSameOppo( j, r, p, i, s, q ) ), 1e-17 );


              for( int spin = 0; spin < 2; ++spin )
              {
                auto sameSame = QSameSame[ j ][ p ][ r ];
                auto sameOppo = QSameOppo[ j ][ p ][ r ];
                auto oppoOppo = QOppoOppo[ j ][ p ][ r ];

                value[ spin ] += density( spin, q, p ) * density( spin, s, r ) * (sameSame( i, q, s ) - sameSame( i, s, q )
                                                                                  + 2 * sameSame( q, s, i ) - 2 * sameSame( s, q, i ));

                value[ spin ] += 2 * density( spin, q, p ) * density( !spin, s, r ) * (sameOppo( i, q, s ) - sameOppo( q, i, s ));

                value[ spin ] += density( !spin, q, p ) * density( !spin, s, r ) * (oppoOppo( i, q, s ) - oppoOppo( i, s, q ));
              }
            }
          }
        }
      }

      for( int spin = 0; spin < 2; ++spin )
      {
        G_3New( 0, j, i ) = value[ 0 ] / 2;
        G_3New( 1, j, i ) = value[ 1 ] / 2;
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

      ArrayView1d< Real const > const r1Weights = r1Grid.quadratureGrid.weights.toViewConst();

      // TODO: Move calculating the energy into this function
      forAll< DefaultPolicy< ParallelHost > >( basisSize * basisSize,
        [&] ( IndexType const ji )
        {
          IndexType const j = ji / basisSize;
          IndexType const i = ji % basisSize;

          T twoElectronContribution = 0;
          for( IndexType r1Idx = 0; r1Idx < r1Grid.nGrid(); ++r1Idx )
          {
            twoElectronContribution += r1Weights[ r1Idx ] * conj( r1Grid.basisValues( r1Idx, j ) ) * (2 * r1Grid.basisValues( r1Idx, i ) * Faa( r1Idx ) - Fai( r1Idx, i ) );
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

      ArrayView1d< Real const > const r1Weights = r1Grid.quadratureGrid.weights.toViewConst();

      // TODO: Move calculating the energy into this function
      forAll< DefaultPolicy< ParallelHost > >( basisSize * basisSize,
        [&] ( IndexType const ji )
        {
          IndexType const j = ji / basisSize;
          IndexType const i = ji % basisSize;

          T twoElectronContribution = 0;
          for( IndexType r1Idx = 0; r1Idx < r1Grid.nGrid(); ++r1Idx )
          {
            twoElectronContribution += r1Weights[ r1Idx ] * conj( r1Grid.basisValues( r1Idx, j ) ) *
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
  ArrayView3d< Cartesian< T > const > const & VjiOppo,
  std::vector< OchiBasisFunction< Real > > const & basisFunctions )
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





  CArray< Array3d< Cartesian< T > >, 2 > const VjbSame {
    Array3d< Cartesian< T > >( r1Grid.nGrid(), basisSize, nElectrons[ 0 ] ),
    Array3d< Cartesian< T > >( r1Grid.nGrid(), basisSize, nElectrons[ 1 ] ) };
  
  CArray< Array3d< Cartesian< T > >, 2 > const VjbOppo {
    Array3d< Cartesian< T > >( r1Grid.nGrid(), basisSize, nElectrons[ 0 ] ),
    Array3d< Cartesian< T > >( r1Grid.nGrid(), basisSize, nElectrons[ 1 ] ) };

  CArray< Array3d< Cartesian< T > >, 2 > const VabSame {
    Array3d< Cartesian< T > >( r1Grid.nGrid(), nElectrons[ 0 ], nElectrons[ 0 ] ),
    Array3d< Cartesian< T > >( r1Grid.nGrid(), nElectrons[ 1 ], nElectrons[ 1 ] ) };
  
  CArray< Array3d< Cartesian< T > >, 2 > const VabOppo {
    Array3d< Cartesian< T > >( r1Grid.nGrid(), nElectrons[ 0 ], nElectrons[ 0 ] ),
    Array3d< Cartesian< T > >( r1Grid.nGrid(), nElectrons[ 1 ], nElectrons[ 1 ] ) };

  CArray< Array1d< Real >, 2 > const sumOfNormsVabSame {
    Array1d< Real >( r1Grid.nGrid() ),
    Array1d< Real >( r1Grid.nGrid() ) };

  CArray< Array1d< Real >, 2 > const sumOfNormsVabOppo {
    Array1d< Real >( r1Grid.nGrid() ),
    Array1d< Real >( r1Grid.nGrid() ) };

  Array2d< Cartesian< T > > const VTilde( 2, r1Grid.nGrid() );

  Array3d< Cartesian< T > > const VTildeSubI( 2, r1Grid.nGrid(), basisSize );

  Array3d< T > const SumOverb_Vbi_dot_VTildeb( 2, r1Grid.nGrid(), basisSize );

  Array2d< Real > const occupiedElectronDensity( 2, r1Grid.nGrid() );

  Array2d< Cartesian< T > > const SumOverb_conjVTildeb_occb( 2, r1Grid.nGrid() );


  Array3d< T > const vectorComponent( 2, basisSize, basisSize );
  Array3d< T > const twoElectronTerms( 2, basisSize, basisSize );
  Array3d< T > const G_3( 2, basisSize, basisSize );


  // {
  //   density.zero();
  //   occupiedOrbitals[ 0 ].zero();
  //   occupiedOrbitals[ 1 ].zero();

  //   std::mt19937_64 gen;
  //   auto dist = std::normal_distribution< Real >( 10, 10 );
    
  //   for( int spin = 0; spin < 2; ++spin )
  //   {
  //     LVARRAY_ERROR_IF_NE( nElectrons[ spin ], 1 );
      
  //     T norm = 0;
  //     for( auto & val : occupiedOrbitals[ spin ] )
  //     {
  //       val = dist( gen );
  //       norm += val * val;
  //     }

  //     norm = std::sqrt( norm );
  //     for( auto & val : occupiedOrbitals[ spin ] )
  //     {
  //       val /= norm;
  //     }

  //     internal::getNewDensity( nElectrons[ spin ], density[ spin ], occupiedOrbitals[ spin ].toSliceConst() );
  //   }
  // }





  Array6d< T > const QSameSame( basisSize, basisSize, basisSize, basisSize, basisSize, basisSize );
  Array6d< T > const QSameOppo( basisSize, basisSize, basisSize, basisSize, basisSize, basisSize );
  Array6d< T > const QOppoOppo( basisSize, basisSize, basisSize, basisSize, basisSize, basisSize );

  internal::computeQ( QSameSame, QSameOppo, QOppoOppo, VjiSame, VjiOppo, r1Grid.quadratureGrid.weights.toViewConst(), r1Grid.basisValues.toViewConst() );



  Array3d< T > const G_3New( 2, basisSize, basisSize );

  for( _iter = 0; _iter < 100; ++_iter )
  {
    FaaSamePlusOppo.zero();
    FaiSame.zero();

    VaaSamePlusOppo.zero();
    Vai.zero();
    Vja.zero();

    scaledGradients.zero();

    internal::assembleG( G_3New, QSameSame.toViewConst(), QSameOppo.toViewConst(), QOppoOppo.toViewConst(), density.toViewConst() );

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

      
      
      
      internal::calculateVjb( VjbSame[ spin ], VjbOppo[ spin ], VjiSame, VjiOppo, occupiedOrbitals[ spin ].toViewConst() );

      internal::calculateVab( VabSame[ spin ], VabOppo[ spin ], sumOfNormsVabSame[ spin ].toView(), sumOfNormsVabOppo[ spin ].toView(), VjbSame[ spin ].toViewConst(), VjbOppo[ spin ].toViewConst(), occupiedOrbitals[ spin ].toViewConst() );


    }

    // Three electron precomputation
    internal::calculateVTilde( VTilde, VjiSame, VjiOppo, density.toViewConst() );

    internal::calculateVTildeSubI( VTildeSubI, VjiSame, r1Grid.basisValues.toViewConst(), density.toViewConst() );

    internal::calculateSumOverb_Vbi_dot_VTildeb( SumOverb_Vbi_dot_VTildeb, VTildeSubI.toViewConst(), VjiSame, density.toViewConst() );


    // {
    //   for( IndexType r1Idx = 0; r1Idx < r1Grid.nGrid(); ++r1Idx )
    //   {
    //     for( int spin = 0; spin < 2; ++spin )
    //     {
    //       for( IndexType i = 0; i < basisSize; ++i )
    //       {
    //         T value = 0;
    //         for( IndexType k = 0; k < basisSize; ++k )
    //         {
    //           for( IndexType ell = 0; ell < basisSize; ++ell )
    //           {
    //             for( IndexType m = 0; m < basisSize; ++m )
    //             {
    //               for( IndexType n = 0; n < basisSize; ++n )
    //               {
    //                 value += density( spin, ell, k ) * density( spin, n, m ) * dot( VjiSame( r1Idx, k, i ), VjiSame( r1Idx, m, ell) ) * r1Grid.basisValues( r1Idx, n );
    //               }
    //             }
    //           }
    //         }

    //         LVARRAY_ERROR_IF_GT( std::abs( value - SumOverb_Vbi_dot_VTildeb( spin, r1Idx, i ) ), 1e-16 );
    //       }
    //     }
    //   }
    // }

    internal::calculateOccupiedElectronDensity( occupiedElectronDensity, r1Grid.basisValues.toViewConst(), density.toViewConst() );

    internal::calculateSumOverb_conjVTildeb_occb( SumOverb_conjVTildeb_occb, VTildeSubI.toViewConst(), r1Grid.basisValues.toViewConst(), density.toViewConst() );

    fockOperator.zero();
    for( int spin = 0; spin < 2; ++spin )
    {
      TCSCF_MARK_SCOPE( "Constructing the two and three electron terms" );

      ArrayView1d< Real const > const r1Weights = r1Grid.quadratureGrid.weights.toViewConst();

      forAll< DefaultPolicy< ParallelHost > >( basisSize * basisSize,
        [&] ( IndexType const ji )
        {
          IndexType const j = ji / basisSize;
          IndexType const i = ji % basisSize;

          T scalar = 0;
          T vector = 0;
          T scalar3 = 0;
          for( IndexType r1Idx = 0; r1Idx < r1Grid.nGrid(); ++r1Idx )
          {
            T const scalarAddition = conj( r1Grid.basisValues( r1Idx, j ) ) * (FaaSamePlusOppo( spin, r1Idx ) * r1Grid.basisValues( r1Idx, i ) - FaiSame( spin, r1Idx, i ));
            scalar += r1Weights[ r1Idx ] * scalarAddition;

            T vectorAddition = 0;
            vectorAddition += conj( r1Grid.basisValues( r1Idx, j ) ) * (dot( VaaSamePlusOppo( spin, r1Idx ), r1Grid.basisGradients( r1Idx, i ) ) - Vai( spin, r1Idx, i ));

            vectorAddition += dot( VjiSame( r1Idx, j, i ), scaledGradients( spin, r1Idx ) );
            
            vectorAddition += dot( VjiOppo( r1Idx, j, i ), scaledGradients( !spin, r1Idx ) );

            vectorAddition -= dot( Vja( spin, r1Idx, j ), r1Grid.basisGradients( r1Idx, i ) );

            vector += r1Weights[ r1Idx ] * vectorAddition;


            T scalar3Addition = 0;
            scalar3Addition += conj( r1Grid.basisValues( r1Idx, j ) ) * (VTilde( spin, r1Idx ).norm() - sumOfNormsVabSame[ spin ][ r1Idx ] - sumOfNormsVabOppo[ !spin ][ r1Idx ]) * r1Grid.basisValues( r1Idx, i );

            scalar3Addition += 2 * conj( r1Grid.basisValues( r1Idx, j ) ) * (SumOverb_Vbi_dot_VTildeb( spin, r1Idx, i ) - dot( VTilde( spin, r1Idx ), VTildeSubI( spin, r1Idx, i ) ));

            scalar3Addition += 2 * (conj( SumOverb_Vbi_dot_VTildeb( spin, r1Idx, j ) ) - dot( conj( VTildeSubI( spin, r1Idx, j ) ), VTilde( spin, r1Idx ) ) ) * r1Grid.basisValues( r1Idx, i );

            T sumOverBSame {};
            T sumOverBOppo {};
            for( IndexType b = 0; b < nElectrons[ spin ]; ++b )
            {
              sumOverBSame += dot( VjbSame[ spin ]( r1Idx, j, b ), conj( VjbSame[ spin ]( r1Idx, i, b ) ) );
              sumOverBOppo += dot( VjbOppo[ spin ]( r1Idx, j, b ), conj( VjbOppo[ spin ]( r1Idx, i, b ) ) );
            }
            
            scalar3Addition += 2 * occupiedElectronDensity( spin, r1Idx ) * (dot( VjiSame( r1Idx, j, i ), VTilde( spin, r1Idx ) ) - sumOverBSame);

            scalar3Addition += 2 * (dot( VTildeSubI( spin, r1Idx, i ), conj( VTildeSubI( spin, r1Idx, j ) ) ) - dot( VjiSame( r1Idx, j, i ), SumOverb_conjVTildeb_occb( spin, r1Idx ) ));

            scalar3Addition += 2 * occupiedElectronDensity( !spin, r1Idx ) * (dot( VjiOppo( r1Idx, j, i ), VTilde( !spin, r1Idx ) ) - sumOverBOppo);

            scalar3Addition -= 2 * dot( VjiOppo( r1Idx, j, i ), SumOverb_conjVTildeb_occb( !spin, r1Idx ) );

            scalar3 += r1Weights[ r1Idx ] * scalar3Addition;
          }

          twoElectronTerms( spin, j, i ) = scalar / 2;
          vectorComponent( spin, j, i ) = vector / 2;
          G_3( spin, j, i ) = scalar3 / 2;
        }
      );

      forAll< DefaultPolicy< ParallelHost > >( basisSize * basisSize,
        [&] ( IndexType const ji )
        {
          IndexType const j = ji / basisSize;
          IndexType const i = ji % basisSize;

          twoElectronTerms( spin, j, i ) += vectorComponent( spin, j, i ) - conj( vectorComponent( spin, i, j ) );

          if( basisFunctions[ j ].l != basisFunctions[ i ].l || basisFunctions[ j ].m != basisFunctions[ i ].m )
          {
            twoElectronTerms( spin, j, i ) = 0;
            G_3New( spin, j, i ) = 0;
            G_3( spin, j, i ) = 0;
          }
        }
      );
    }


    // for( int spin = 0; spin < 2; ++spin )
    // {
    //   Real maxAbsDiff = 0;
    //   Real d1;
    //   Real d2;
    //   for( IndexType j = 0; j < basisSize; ++j )
    //   {
    //     for( IndexType i = 0; i < basisSize; ++i )
    //     {
    //       Real const diff = std::max( maxAbsDiff, std::abs( G_3( spin, j, i ) - G_3New( spin, j, i ) ) );
    //       if( maxAbsDiff < diff )
    //       {
    //         maxAbsDiff = diff;
    //         d1 = G_3( spin, j, i ).real();
    //         d2 = G_3New( spin, j, i ).real();
    //       }
    //     }
    //   }

    //   LVARRAY_LOG( "maxAbsDiff = " << maxAbsDiff << ", d1 = " << d1 << ", d2 = " << d2 );
    // }

    Real const newEnergy = internal::constructSCFOperator(
      fockOperator.toView(),
      oneElectronTerms.toViewConst(), 
      twoElectronTerms.toViewConst(),
      G_3New.toViewConst(),
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
