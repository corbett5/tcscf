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
      Cartesian< ResultType > grad {};
      
      for( IndexType j = 0; j < nBasis; ++j )
      {
        grad.scaledAdd( eigenvectors( j, i ), basisGradients( r1Idx, j ) );
      }

      occupiedGradients( r1Idx, i ) = grad;
    }
  }
}

/**
 *
 */
template< bool USE_VECTOR, typename T >
void ai(
  ArraySlice1d< T > const integralValues,
  ArraySlice1d< T const > const & occupiedValuesR1,
  ArrayView2d< RealType< T > const > const & basisValuesR2,
  ArrayView2d< T const > const & occupiedValuesR2,
  ArrayView2d< Cartesian< T > const > const & basisGradientsR2,
  ArraySlice1d< RealType< T > const > const & scalarFunction,
  ArraySlice1d< Cartesian< RealType< T > > const > const & vectorFunction )
{
  using Real = RealType< T >;

  IndexType const nGridR2 = scalarFunction.size();
  IndexType const nOccupied = occupiedValuesR1.size();
  IndexType const nBasis = integralValues.size();

  for( IndexType r2Idx = 0; r2Idx < nGridR2; ++r2Idx )
  {
    T aSum = 0;
    for( IndexType a = 0; a < nOccupied; ++a )
    {
      aSum += occupiedValuesR1[ a ] * conj( occupiedValuesR2( r2Idx, a ) );
    }

    Real const scalalPotential = scalarFunction[ r2Idx ];

    if constexpr ( !USE_VECTOR )
    {
      for( IndexType i = 0; i < nBasis; ++i )
      {
        integralValues[ i ] += aSum * scalalPotential * basisValuesR2( r2Idx, i );
      }
    }
    else
    {
      Cartesian< Real > const vector = vectorFunction[ r2Idx ];
      
      for( IndexType i = 0; i < integralValues.size(); ++i )
      {
        integralValues[ i ] += aSum * (scalalPotential * basisValuesR2( r2Idx, i ) + dot( vector, basisGradientsR2( r2Idx, i ) ));
      }
    }
  }

  constexpr Real scale = 2 * pi< Real >;
  for( auto & value : integralValues )
  {
    value *= scale;
  }
}

/**
 *
 */
template< bool USE_VECTOR, typename T >
void ja(
  ArraySlice1d< T > const integralValues,
  ArraySlice1d< T const > const & occupiedValuesR1,
  ArrayView2d< RealType< T > const > const & basisValuesR2,
  ArrayView2d< T const > const & occupiedValuesR2,
  ArrayView2d< Cartesian< T > const > const & occupiedGradientsR2,
  ArraySlice1d< RealType< T > const > const & scalarFunction,
  ArraySlice1d< Cartesian< RealType< T > > const > const & vectorFunction )
{
  using Real = RealType< T >;

  IndexType const nGridR2 = scalarFunction.size();
  IndexType const nOccupied = occupiedValuesR1.size();
  IndexType const nBasis = integralValues.size();

  for( IndexType r2Idx = 0; r2Idx < nGridR2; ++r2Idx )
  {
    T aSum = 0;
    for( IndexType a = 0; a < nOccupied; ++a )
    {
      aSum += conj( occupiedValuesR1[ a ] ) * occupiedValuesR2( r2Idx, a );
    }

    T integrand = aSum * scalarFunction[ r2Idx ];

    if constexpr ( USE_VECTOR )
    {
      Cartesian< T > gradASum {};
      
      for( IndexType a = 0; a < nOccupied; ++a )
      {
        gradASum.scaledAdd( conj( occupiedValuesR1[ a ] ), occupiedGradientsR2( r2Idx, a ) );
      }

      integrand += dot( vectorFunction[ r2Idx ], gradASum );
    }
    
    for( IndexType j = 0; j < nBasis; ++j )
    {
      integralValues[ j ] += basisValuesR2( r2Idx, j ) * integrand;
    }
  }

  Real const scale = 2 * pi< Real >;
  for( auto & value : integralValues )
  {
    value *= scale;
  }
}

/**
 *
 */
template< bool USE_VECTOR, typename T >
void aa(
  T & result,
  ArrayView2d< T const > const & occupiedValuesR2,
  ArrayView2d< Cartesian< T > const > const & occupiedGradientsR2,
  ArraySlice1d< RealType< T > const > const & scalarFunction,
  ArraySlice1d< Cartesian< RealType< T > > const > const & vectorFunction )
{
  using Real = RealType< T >;

  IndexType const nGridR2 = scalarFunction.size();
  IndexType const nOccupied = occupiedValuesR2.size( 1 );

  T answer {};
  for( IndexType r2Idx = 0; r2Idx < nGridR2; ++r2Idx )
  {
    Real sumOfNorms = 0;
    for( IndexType a = 0; a < nOccupied; ++a )
    {
      sumOfNorms += std::norm( occupiedValuesR2( r2Idx, a ) );
    }

    T contribution = sumOfNorms * scalarFunction[ r2Idx ];
    if constexpr ( USE_VECTOR )
    {
      Cartesian< T > sumOfGradients {};
      for( IndexType a = 0; a < nOccupied; ++a )
      {
        sumOfGradients.scaledAdd( conj( occupiedValuesR2( r2Idx, a ) ), occupiedGradientsR2( r2Idx, a ) );
      }

      contribution += dot( vectorFunction[ r2Idx ], sumOfGradients );
    }

    answer += contribution;
  }

  result = 2 * pi< Real > * answer;
}

/**
 *
 */
template< bool USE_VECTOR, typename T >
void aa12(
  std::pair< T, Cartesian< T > > & result,
  ArrayView2d< T const > const & occupiedValuesR2,
  ArraySlice1d< RealType< T > const > const & scalarFunction,
  ArraySlice1d< Cartesian< RealType< T > > const > const & vectorFunction )
{
  using Real = RealType< T >;

  IndexType const nGridR2 = scalarFunction.size();
  IndexType const nOccupied = occupiedValuesR2.size( 1 );

  T scalarIntegral {};
  Cartesian< Real > vectorIntegral {};
  for( IndexType r2Idx = 0; r2Idx < nGridR2; ++r2Idx )
  {
    Real sumOfNorms = 0;
    for( IndexType a = 0; a < nOccupied; ++a )
    {
      sumOfNorms += std::norm( occupiedValuesR2( r2Idx, a ) );
    }

    // TODO This should be f( r1, r2 ) not f( r2, r1 )
    scalarIntegral += sumOfNorms * scalarFunction[ r2Idx ];

    if constexpr ( USE_VECTOR )
    {
      vectorIntegral.scaledAdd( sumOfNorms, vectorFunction[ r2Idx ] );
    }
  }

  result.first = 2 * pi< Real > * scalarIntegral;
  result.second = { 2 * pi< Real > * vectorIntegral.x(),
                    2 * pi< Real > * vectorIntegral.y(),
                    2 * pi< Real > * vectorIntegral.z() };
}

static constexpr int AI = 0;
static constexpr int AA = 1;
static constexpr int JA = 2;
static constexpr int AA12 = 3;

////////////////////////////////////////////////////////////////////////////////////////////////////////
template< int INTEGRAL_TYPE, bool USE_VECTOR, typename U, typename T, int DIM >
void computeInnerIntegrals(
  ArraySlice< U, DIM > const & innerIntegrals,
  integration::QMCGrid< RealType< T >, 3 > const & r1Grid,
  integration::QMCGrid< RealType< T >, 2 > const & r2Grid,
  ArrayView2d< T const > const & r1OccupiedValues,
  ArrayView2d< T const > const & r2OccupiedValues,
  ArrayView2d< Cartesian< T > const > const r2OccupiedGradients,
  ArrayView2d< RealType< T > const > const & scalarFunction,
  ArrayView2d< Cartesian< RealType< T > > const > const & vectorFunction )
{
  std::string caliperTag;
  if( INTEGRAL_TYPE == AI )
  {
    caliperTag = "Computing I^a_i";
  }
  if( INTEGRAL_TYPE == JA )
  {
    caliperTag = "Computing I^j_a";
  }
  if( INTEGRAL_TYPE == AA )
  {
    caliperTag = "Computing I^a_a";
  }
  if( INTEGRAL_TYPE == AA12 )
  {
    caliperTag = "Computing I12^a_a";
  }

  TCSCF_MARK_SCOPE_STRING( caliperTag.data() );

  using PolicyType = ParallelHost;
  using Real = RealType< T >;
  
  ArrayView2d< Real const > const r2BasisValues = r2Grid.basisValues.toViewConst();
  ArrayView2d< Cartesian< T > const > const r2BasisGradients = r2Grid.basisGradients.toViewConst();

  forAll< DefaultPolicy< PolicyType > >( r1Grid.quadratureGrid.points.size( 1 ),
    [=] ( IndexType const r1Idx )
    {
      if constexpr ( INTEGRAL_TYPE == AI )
      {
        internal::ai< USE_VECTOR >(
          innerIntegrals[ r1Idx ],
          r1OccupiedValues[ r1Idx ],
          r2BasisValues,
          r2OccupiedValues,
          r2BasisGradients,
          scalarFunction[ r1Idx ],
          vectorFunction[ r1Idx ] );
      }
      if constexpr ( INTEGRAL_TYPE == JA )
      {
        internal::ja< USE_VECTOR >(
          innerIntegrals[ r1Idx ],
          r1OccupiedValues[ r1Idx ],
          r2BasisValues,
          r2OccupiedValues,
          r2OccupiedGradients,
          scalarFunction[ r1Idx ],
          vectorFunction[ r1Idx ] );
      }
      if constexpr ( INTEGRAL_TYPE == AA )
      {
        internal::aa< USE_VECTOR >(
          innerIntegrals[ r1Idx ],
          r2OccupiedValues,
          r2OccupiedGradients,
          scalarFunction[ r1Idx ],
          vectorFunction[ r1Idx ] );
      }
      if constexpr ( INTEGRAL_TYPE == AA12 )
      {
        internal::aa12< USE_VECTOR >(
          innerIntegrals[ r1Idx ],
          r2OccupiedValues,
          scalarFunction[ r1Idx ],
          vectorFunction[ r1Idx ] );
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
  integration::QMCGrid< Real, 2 > const & r2Grid,
  ArrayView2d< Real const > const & r12Inv )
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

  for( _iter = 0; _iter < 50; ++_iter )
  {
    internal::occupiedOrbitalValues( r1OccupiedValues, r1Grid.basisValues.toViewConst(), eigenvectors.toSliceConst() );
    internal::occupiedOrbitalValues( r2OccupiedValues, r2Grid.basisValues.toViewConst(), eigenvectors.toSliceConst() );

    innerIntegralsKI.zero();
    internal::computeInnerIntegrals< internal::AI, false >(
      innerIntegralsKI.toSlice(),
      r1Grid,
      r2Grid,
      r1OccupiedValues.toViewConst(),
      r2OccupiedValues.toViewConst(),
      {},
      r12Inv,
      {} );
    
    internal::computeInnerIntegrals< internal::AA, false >(
      innerIntegralsKK.toSlice(),
      r1Grid,
      r2Grid,
      r1OccupiedValues.toViewConst(),
      r2OccupiedValues.toViewConst(),
      {},
      r12Inv,
      {} );

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
  integration::QMCGrid< Real, 2 > const & r2Grid,
  ArrayView2d< Real const > const & r12Inv )
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

  for( _iter = 0; _iter < 50; ++_iter )
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
      internal::computeInnerIntegrals< internal::AI, false >(
        innerIntegralsKI[ spin ],
        r1Grid,
        r2Grid,
        r1OccupiedValues[ spin ].toViewConst(),
        r2OccupiedValues[ spin ].toViewConst(),
        {},
        r12Inv,
        {} );
      
      internal::computeInnerIntegrals< internal::AA, false >(
        innerIntegralsKK[ spin ],
        r1Grid,
        r2Grid,
        r1OccupiedValues[ spin ].toViewConst(),
        r2OccupiedValues[ spin ].toViewConst(),
        {},
        r12Inv,
        {} );
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
  integration::QMCGrid< Real, 2 > const & r2Grid,
  ArrayView2d< Real const > const & scalarSame,
  ArrayView2d< Real const > const & scalarOppo,
  ArrayView2d< Cartesian< Real > const > const & vectorSame21,
  ArrayView2d< Cartesian< Real > const > const & vectorOppo21,
  ArrayView2d< Cartesian< Real > const > const & vectorSame12,
  ArrayView2d< Cartesian< Real > const > const & vectorOppo12 )
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

  Array2d< T > const r2OccupiedValues[ 2 ]{ Array2d< T >( r2Grid.nGrid(), nElectrons[ 0 ] ),
                                            Array2d< T >( r2Grid.nGrid(), nElectrons[ 1 ] ) };

  Array2d< Cartesian< T > > const r2OccupiedGradients[ 2 ]{ Array2d< Cartesian< T > >( r2Grid.nGrid(), nElectrons[ 0 ] ),
                                                            Array2d< Cartesian< T > >( r2Grid.nGrid(), nElectrons[ 1 ] ) };

  Array2d< T > const Iaa( 2, r1Grid.nGrid() );
  Array3d< T > const Iai( 2, r1Grid.nGrid(), basisSize );
  Array2d< std::pair< T, Cartesian< T > > > const Iaa12( 2, r1Grid.nGrid() );
  Array3d< T > const Ija( 2, r1Grid.nGrid(), basisSize );

  Array2d< T > const IOppoaa( 2, r1Grid.nGrid() );
  Array2d< std::pair< T, Cartesian< T > > > const IOppoaa12( 2, r1Grid.nGrid() );

  for( _iter = 0; _iter < 100; ++_iter )
  {
    for( int spin = 0; spin < 2; ++spin )
    {
      internal::occupiedOrbitalValues( r1OccupiedValues[ spin ].toView(), r1Grid.basisValues.toViewConst(), occupiedOrbitals[ spin ].toSliceConst() );
      internal::occupiedOrbitalValues( r2OccupiedValues[ spin ].toView(), r2Grid.basisValues.toViewConst(), occupiedOrbitals[ spin ].toSliceConst() );
      internal::occupiedOrbitalGradients( r2OccupiedGradients[ spin ].toView(), r2Grid.basisGradients.toViewConst(), occupiedOrbitals[ spin ].toSliceConst() );
    }

    Iaa.zero();
    Iai.zero();
    Iaa12.zero();
    Ija.zero();
    IOppoaa.zero();
    IOppoaa12.zero();
    for( int spin = 0; spin < 2; ++spin )
    {
      internal::computeInnerIntegrals< internal::AA, true >(
        Iaa[ spin ],
        r1Grid,
        r2Grid,
        r1OccupiedValues[ spin ].toViewConst(),
        r2OccupiedValues[ spin ].toViewConst(),
        r2OccupiedGradients[ spin ].toViewConst(),
        scalarSame,
        vectorSame21 );
      
      internal::computeInnerIntegrals< internal::AI, true >(
        Iai[ spin ],
        r1Grid,
        r2Grid,
        r1OccupiedValues[ spin ].toViewConst(),
        r2OccupiedValues[ spin ].toViewConst(),
        r2OccupiedGradients[ spin ].toViewConst(),
        scalarSame,
        vectorSame21 );
      
      internal::computeInnerIntegrals< internal::AA12, true >(
        Iaa12[ spin ],
        r1Grid,
        r2Grid,
        r1OccupiedValues[ spin ].toViewConst(),
        r2OccupiedValues[ spin ].toViewConst(),
        r2OccupiedGradients[ spin ].toViewConst(),
        scalarSame,
        vectorSame12 );
      
      internal::computeInnerIntegrals< internal::JA, true >(
        Ija[ spin ],
        r1Grid,
        r2Grid,
        r1OccupiedValues[ spin ].toViewConst(),
        r2OccupiedValues[ spin ].toViewConst(),
        r2OccupiedGradients[ spin ].toViewConst(),
        scalarSame,
        vectorSame21 );

      internal::computeInnerIntegrals< internal::AA, true >(
        IOppoaa[ spin ],
        r1Grid,
        r2Grid,
        r1OccupiedValues[ !spin ].toViewConst(),
        r2OccupiedValues[ !spin ].toViewConst(),
        r2OccupiedGradients[ !spin ].toViewConst(),
        scalarOppo,
        vectorOppo21 );
      
      internal::computeInnerIntegrals< internal::AA12, true >(
        IOppoaa12[ spin ],
        r1Grid,
        r2Grid,
        r1OccupiedValues[ !spin ].toViewConst(),
        r2OccupiedValues[ !spin ].toViewConst(),
        r2OccupiedGradients[ !spin ].toViewConst(),
        scalarOppo,
        vectorOppo12 );
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
            twoElectronContribution += 
              conj( r1Grid.basisValues( r1Idx, j ) ) * Iaa( spin, r1Idx ) * r1Grid.basisValues( r1Idx, i )
            + conj( r1Grid.basisValues( r1Idx, j ) ) * Iaa12( spin, r1Idx ).first * r1Grid.basisValues( r1Idx, i )
            + conj( r1Grid.basisValues( r1Idx, j ) ) * dot( Iaa12( spin, r1Idx ).second, r1Grid.basisGradients( r1Idx, i ) )
            - conj( r1Grid.basisValues( r1Idx, j ) ) * Iai( spin, r1Idx, i )
            - Ija( spin, r1Idx, j ) * r1Grid.basisValues( r1Idx, i )
            + conj( r1Grid.basisValues( r1Idx, j ) ) * IOppoaa( spin, r1Idx ) * r1Grid.basisValues( r1Idx, i )
            + conj( r1Grid.basisValues( r1Idx, j ) ) * IOppoaa12( spin, r1Idx ).first * r1Grid.basisValues( r1Idx, i )
            + conj( r1Grid.basisValues( r1Idx, j ) ) * dot( IOppoaa12( spin, r1Idx ).second, r1Grid.basisGradients( r1Idx, i ) );
          }

          fockOperator( spin, j, i ) = oneElectronTerms( j, i ) + twoElectronContribution / 2;
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
