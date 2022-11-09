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
  for( IndexType r1Idx = 0; r1Idx < gridSize; ++r1Idx )
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
  for( IndexType r1Idx = 0; r1Idx < gridSize; ++r1Idx )
  {
    for( IndexType i = 0; i < nElectrons; ++i )
    {
      Cartesian< ResultType > grad = 0;
      
      for( IndexType j = 0; j < nBasis; ++j )
      {
        grad += basisGradients( r1Idx, j ) * eigenvectors( j, i );
      }

      occupiedGradients( r1Idx, i ) = grad;
    }
  }
}

/**
 *
 */
template< typename T, typename SCALAR, typename VECTOR >
void evaluateR2IntegralSumOverOccupied(
  ArrayView2d< RealType< T > const > const & points,
  ArrayView1d< RealType< T > const > const & weights,
  Cartesian< RealType< T > > const & r1,
  ArraySlice1d< T const > const & occupiedValuesR1,
  ArrayView2d< T const > const & occupiedValuesR2,
  ArraySlice1d< T > const integralValues,
  ArrayView2d< T const > const & b4Values,
  ArrayView2d< Cartesian< T > const > const & b4Gradients,
  SCALAR const & f,
  VECTOR const & v )
{
  using Real = RealType< T >;

  for( IndexType r2Idx = 0; r2Idx < weights.size(); ++r2Idx )
  {
    T b2Sum = 0;
    for( IndexType b2 = 0; b2 < occupiedValuesR1.size( 1 ); ++b2 )
    {
      b2Sum += occupiedValuesR1[ b2 ] * conj( occupiedValuesR2( r2Idx, b2 ) );
    }

    Cartesian< Real > const r2 { points( 0, r2Idx ), points( 1, r2Idx ), points( 2, r2Idx ) };
    Real const scalalPotential = f( r1, r2 ) * weights[ r2Idx ];

    if constexpr ( std::is_same_v< VECTOR, std::nullptr_t > )
    {
      for( IndexType b4 = 0; b4 < integralValues.size(); ++b4 )
      {
        integralValues[ b4 ] += b2Sum * scalalPotential * b4Values( r2Idx, b4 );
      }
    }
    else
    {
      Cartesian< T > const vector = v( r1, r2 );
      
      for( IndexType b4 = 0; b4 < integralValues.size(); ++b4 )
      {
        integralValues[ b4 ] += b2Sum * (scalalPotential * b4Values( r2Idx, b4 ) + weights[ r2Idx ] * dot( vector, b4Gradients( r2Idx, b4 ) ));
      }
    }
  }
}

/**
 *
 */
template< typename T, typename SCALAR, typename VECTOR >
T evaluateR2Integral(
  ArrayView2d< RealType< T > const > const & points,
  ArrayView1d< RealType< T > const > const & weights,
  Cartesian< RealType< T > > const & r1,
  ArrayView2d< T const > const & b2Values,
  ArrayView2d< Cartesian< T > const > const & b2Gradients,
  SCALAR const & f,
  VECTOR const & v )
{
  using Real = RealType< T >;

  T answer {};
  for( IndexType r2Idx = 0; r2Idx < weights.size(); ++r2Idx )
  {
    Cartesian< Real > const r2 { points( 0, r2Idx ), points( 1, r2Idx ), points( 2, r2Idx ) };
    auto const scalalPotential = f( r1, r2 );

    Real sumOfNorms = 0;
    for( IndexType b2 = 0; b2 < b2Values.size( 1 ); ++b2 )
    {
      sumOfNorms += std::norm( b2Values( r2Idx, b2 ) );
    }

    if constexpr ( std::is_same_v< VECTOR, std::nullptr_t > )
    {
      answer += weights[ r2Idx ] * sumOfNorms * scalalPotential;
    }
    else
    {
      Cartesian< T > sumOfGradients {};
      for( IndexType b2 = 0; b2 < b2Values.size( 1 ); ++b2 )
      {
        sumOfGradients += conj( b2Values( r2Idx, b2 ) ) * std::norm( b2Gradients( r2Idx, b2 ) );
      }

      answer += weights[ r2Idx ] * (sumOfNorms * scalalPotential + dot( v( r1, r2 ), sumOfGradients ));
    }
  }

  return answer;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
template< typename T, typename SCALAR, typename VECTOR=std::nullptr_t >
void sumOccupiedAI(
  ArraySlice2d< T > const & innerIntegralsAI,
  integration::QMCGrid< RealType< T >, 3 > const & r1Grid,
  integration::QMCGrid< RealType< T >, 3 > const & r2Grid,
  ArrayView2d< T const > const & r1OccupiedValues,
  ArrayView2d< T const > const & r2OccupiedValues,
  SCALAR && f,
  VECTOR && v={} )
{
  TCSCF_MARK_FUNCTION;

  using PolicyType = ParallelHost;
  using Real = RealType< T >;
  
  IndexType const nBasis = r1Grid.nBasis();

  ArrayView2d< Real const > const r1Points = r1Grid.quadratureGrid.points.toViewConst();
  ArrayView1d< Real const > const r1Weights = r1Grid.quadratureGrid.weights.toViewConst();

  ArrayView2d< Real const > const r2Points = r2Grid.quadratureGrid.points.toViewConst();
  ArrayView1d< Real const > const r2Weights = r2Grid.quadratureGrid.weights.toViewConst();
  ArrayView2d< T const > const r2BasisValues = r2Grid.basisValues.toViewConst();
  ArrayView2d< Cartesian< T > const > const r2BasisGradients = r2Grid.basisGradients.toViewConst();

  forAll< DefaultPolicy< PolicyType > >( r1Grid.quadratureGrid.points.size( 1 ),
    [=] ( IndexType const r1Idx )
    {
      Cartesian< Real > const r1 = { r1Points( 0, r1Idx ), r1Points( 1, r1Idx ), r1Points( 2, r1Idx ) };
      Real const r1Weight = r1Weights( r1Idx );

      internal::evaluateR2IntegralSumOverOccupied< T >(
        r2Points,
        r2Weights,
        r1,
        r1OccupiedValues[ r1Idx ],
        r2OccupiedValues,
        innerIntegralsAI[ r1Idx ],
        r2BasisValues,
        r2BasisGradients,
        f,
        v );
      
      for( IndexType i = 0; i < nBasis; ++i )
      {
        innerIntegralsAI( r1Idx, i ) *= r1Weight;
      }
    }
  );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
template< typename T, typename SCALAR, typename VECTOR=std::nullptr_t >
void sumOccupiedAA(
  ArraySlice1d< T > const & innerIntegralsAA,
  integration::QMCGrid< RealType< T >, 3 > const & r1Grid,
  integration::QMCGrid< RealType< T >, 3 > const & r2Grid,
  ArrayView2d< T const > const & r2OccupiedValues,
  ArrayView2d< Cartesian< T > const > const r2OccupiedGradients,
  SCALAR && f,
  VECTOR && v={} )
{
  TCSCF_MARK_FUNCTION;

  using PolicyType = ParallelHost;
  using Real = RealType< T >;

  ArrayView2d< Real const > const r1Points = r1Grid.quadratureGrid.points.toViewConst();
  ArrayView1d< Real const > const r1Weights = r1Grid.quadratureGrid.weights.toViewConst();

  ArrayView2d< Real const > const r2Points = r2Grid.quadratureGrid.points.toViewConst();
  ArrayView1d< Real const > const r2Weights = r2Grid.quadratureGrid.weights.toViewConst();

  forAll< DefaultPolicy< PolicyType > >( r1Grid.quadratureGrid.points.size( 1 ),
    [=] ( IndexType const r1Idx )
    {
      Cartesian< Real > const r1 = { r1Points( 0, r1Idx ), r1Points( 1, r1Idx ), r1Points( 2, r1Idx ) };
      Real const r1Weight = r1Weights( r1Idx );

      innerIntegralsAA( r1Idx ) = r1Weight * internal::evaluateR2Integral(
        r2Points,
        r2Weights,
        r1,
        r2OccupiedValues,
        r2OccupiedGradients,
        f,
        v );
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
  integration::QMCGrid< Real, 3 > const & r2Grid )
{
  TCSCF_MARK_FUNCTION;

  LVARRAY_ERROR_IF_NE( oneElectronTerms.size( 1 ), basisSize );
  LVARRAY_ERROR_IF_NE( r1Grid.nBasis(), basisSize );
  LVARRAY_ERROR_IF_NE( r2Grid.nBasis(), basisSize );

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

  Array2d< T > const r1OccupiedValues( r1Grid.nGrid(), nElectrons / 2 );
  Array2d< T > const r2OccupiedValues( r2Grid.nGrid(), nElectrons / 2 );

  Array2d< T > const innerIntegralsKI( r1Grid.nGrid(), basisSize );
  Array1d< T > const innerIntegralsKK( r1Grid.nGrid() );

  Array2d< T > const fockTwoTermsSameSpin( basisSize, basisSize );

  for( int iter = 0; iter < 1000; ++iter )
  {
    internal::occupiedOrbitalValues( r1OccupiedValues, r1Grid.basisValues.toViewConst(), eigenvectors.toSliceConst() );
    internal::occupiedOrbitalValues( r2OccupiedValues, r2Grid.basisValues.toViewConst(), eigenvectors.toSliceConst() );

    innerIntegralsKI.zero();
    internal::sumOccupiedAI(
      innerIntegralsKI.toSlice(),
      r1Grid,
      r2Grid,
      r1OccupiedValues.toViewConst(),
      r2OccupiedValues.toViewConst(),
      [] (Cartesian< Real > const & r1, Cartesian< Real > const & r2 )
      {
        return 1 / (r1 - r2).r();
      } );

    innerIntegralsKK.zero();
    internal::sumOccupiedAA(
      innerIntegralsKK.toSlice(),
      r1Grid,
      r2Grid,
      r2OccupiedValues.toViewConst(),
      {},
      [] (Cartesian< Real > const & r1, Cartesian< Real > const & r2 )
      {
        return 1 / (r1 - r2).r();
      } );

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
            twoElectronContribution += conj( r1Grid.basisValues( r1Idx, j ) ) * (2 * r1Grid.basisValues( r1Idx, i ) * innerIntegralsKK( r1Idx ) - innerIntegralsKI( r1Idx, i ) );
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
  integration::QMCGrid< Real, 3 > const & r2Grid )
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

  Array2d< T > const r1OccupiedValues[ 2 ]{ Array2d< T >( r1Grid.nGrid(), nElectrons[ 0 ] ),
                                            Array2d< T >( r1Grid.nGrid(), nElectrons[ 1 ] ) };

  Array2d< T > const r2OccupiedValues[ 2 ]{ Array2d< T >( r2Grid.nGrid(), nElectrons[ 0 ] ),
                                            Array2d< T >( r2Grid.nGrid(), nElectrons[ 0 ] ) };

  Array3d< T > const innerIntegralsKI( 2, r1Grid.nGrid(), basisSize );
  Array2d< T > const innerIntegralsKK( 2, r1Grid.nGrid() );

  for( int iter = 0; iter < 1000; ++iter )
  {
    for( int spin = 0; spin < 2; ++spin )
    {
      internal::occupiedOrbitalValues( r1OccupiedValues[ spin ].toView(), r1Grid.basisValues.toViewConst(), eigenvectors[ spin ].toSliceConst() );
      internal::occupiedOrbitalValues( r2OccupiedValues[ spin ].toView(), r2Grid.basisValues.toViewConst(), eigenvectors[ spin ].toSliceConst() );
    }

    innerIntegralsKI.zero();
    innerIntegralsKK.zero();
    for( int spin = 0; spin < 2; ++spin )
    {
      internal::sumOccupiedAI(
        innerIntegralsKI[ spin ],
        r1Grid,
        r2Grid,
        r1OccupiedValues[ spin ].toViewConst(),
        r2OccupiedValues[ spin ].toViewConst(),
        [] (Cartesian< Real > const & r1, Cartesian< Real > const & r2 )
        {
          return 1 / (r1 - r2).r();
        } );

      internal::sumOccupiedAA(
        innerIntegralsKK[ spin ],
        r1Grid,
        r2Grid,
        r2OccupiedValues[ spin ].toViewConst(),
        {},
        [] (Cartesian< Real > const & r1, Cartesian< Real > const & r2 )
        {
          return 1 / (r1 - r2).r();
        } );
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
              (r1Grid.basisValues( r1Idx, i ) * (innerIntegralsKK( spin, r1Idx ) + innerIntegralsKK( !spin, r1Idx )) - innerIntegralsKI( spin, r1Idx, i ));
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
