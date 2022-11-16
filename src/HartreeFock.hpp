#pragma once

#include "AtomicBasis.hpp"
#include "integration/integrateAll.hpp"

#include "dense/common.hpp"

namespace tcscf
{

/**
 */
template< typename T, typename REAL, typename LAMBDA >
void precomputeIntegrand(
  ArrayView2d< T > const & values,
  integration::QMCGrid< REAL, 3 > const & r1Grid,
  integration::QMCGrid< REAL, 2 > const & r2Grid,
  LAMBDA && f )
{
  TCSCF_MARK_FUNCTION;

  LVARRAY_ERROR_IF_NE( values.size( 0 ), r1Grid.nGrid() );
  LVARRAY_ERROR_IF_NE( values.size( 1 ), r2Grid.nGrid() );

  ArrayView2d< REAL const > const r1Points = r1Grid.quadratureGrid.points.toViewConst();
  ArrayView1d< REAL const > const r1Weights = r1Grid.quadratureGrid.weights.toViewConst();

  ArrayView2d< REAL const > const r2Points = r2Grid.quadratureGrid.points.toViewConst();
  ArrayView1d< REAL const > const r2Weights = r2Grid.quadratureGrid.weights.toViewConst();

  forAll< DefaultPolicy< ParallelHost > >( r1Points.size( 1 ),
    [=] ( IndexType const r1Idx )
    {
      Cartesian< REAL > const r1 = { r1Points( 0, r1Idx ), r1Points( 1, r1Idx ), r1Points( 2, r1Idx ) };
      for( IndexType r2Idx = 0; r2Idx < r2Points.size( 1 ); ++r2Idx )
      {
        Cartesian< REAL > const r2 { r2Points( 0, r2Idx ), 0, r2Points( 1, r2Idx ) };
        values( r1Idx, r2Idx ) = f( r1, r2 ) * (r1Weights[ r1Idx ] * r2Weights[ r2Idx ]);
      }
    }
  );
}

/**
 */
template< typename T, typename REAL, typename LAMBDA >
void precomputeIntegrand(
  ArrayView2d< T > const & values,
  integration::QMCGrid< REAL, 3 > const & r1Grid,
  integration::QMCGrid< REAL, 3 > const & r2Grid,
  LAMBDA && f )
{
  TCSCF_MARK_FUNCTION;

  LVARRAY_ERROR_IF_NE( values.size( 0 ), r1Grid.nGrid() );
  LVARRAY_ERROR_IF_NE( values.size( 1 ), r2Grid.nGrid() );

  ArrayView2d< REAL const > const r1Points = r1Grid.quadratureGrid.points.toViewConst();
  ArrayView1d< REAL const > const r1Weights = r1Grid.quadratureGrid.weights.toViewConst();

  ArrayView2d< REAL const > const r2Points = r2Grid.quadratureGrid.points.toViewConst();
  ArrayView1d< REAL const > const r2Weights = r2Grid.quadratureGrid.weights.toViewConst();

  forAll< DefaultPolicy< ParallelHost > >( r1Points.size( 1 ),
    [=] ( IndexType const r1Idx )
    {
      Cartesian< REAL > const r1 = { r1Points( 0, r1Idx ), r1Points( 1, r1Idx ), r1Points( 2, r1Idx ) };
      for( IndexType r2Idx = 0; r2Idx < r2Points.size( 1 ); ++r2Idx )
      {
        Cartesian< REAL > const r2 { r2Points( 0, r2Idx ), r2Points( 1, r2Idx ), r2Points( 2, r2Idx ) };
        values( r1Idx, r2Idx ) = f( r1, r2 ) * (r1Weights[ r1Idx ] * r2Weights[ r2Idx ]);
      }
    }
  );
}

/**
 * 
 */
template< typename REAL >
Array3d< REAL > computeF(
  integration::QMCGrid< REAL, 3 > const & r1Grid,
  integration::QMCGrid< REAL, 2 > const & r2Grid,
  ArrayView2d< REAL const > const & scalarFunction )
{
  TCSCF_MARK_FUNCTION;

  using PolicyType = ParallelHost;
  using Real = REAL;

  IndexType const nBasis = r1Grid.nBasis();
  IndexType const nGridR1 = r1Grid.nGrid();
  IndexType const nGridR2 = r2Grid.nGrid();

  LVARRAY_ERROR_IF_NE( r1Grid.nBasis(), r2Grid.nBasis() );
  LVARRAY_ERROR_IF_NE( scalarFunction.size( 0 ), nGridR1 );
  LVARRAY_ERROR_IF_NE( scalarFunction.size( 1 ), nGridR2 );

  ArrayView2d< Real const > const r2BasisValues = r2Grid.basisValues.toViewConst();
  Array3d< Real > F( nGridR1, nBasis, nBasis );

  forAll< DefaultPolicy< PolicyType > >( nGridR1,
    [=, F=F.toView()] ( IndexType const r1Idx )
    {
      for( IndexType j = 0; j < nBasis; ++j )
      {
        for( IndexType i = j; i < nBasis; ++i )
        {
          Real answer {};
          for( IndexType r2Idx = 0; r2Idx < nGridR2; ++r2Idx )
          {
            answer += conj( r2BasisValues( r2Idx, j ) ) * scalarFunction( r1Idx, r2Idx ) * r2BasisValues( r2Idx, i );
          }

          F( r1Idx, j, i ) = 2 * pi< Real > * answer;
          F( r1Idx, i, j ) = conj( 2 * pi< Real > * answer );
        }
      }
    }
  );

  return F;
}

/**
 * 
 */
template< typename REAL >
Array3d< Cartesian< std::complex< REAL > > > computeV(
  integration::QMCGrid< REAL, 3 > const & r1Grid,
  integration::QMCGrid< REAL, 3 > const & r2Grid,
  ArrayView2d< Cartesian< REAL > const > const & vectorFunction )
{
  TCSCF_MARK_FUNCTION;

  using PolicyType = ParallelHost;
  using Real = REAL;
  using Complex = std::complex< Real >;

  LVARRAY_ERROR_IF_NE( r1Grid.nBasis(), r2Grid.nBasis() );

  IndexType const nBasis = r1Grid.nBasis();

  ArrayView2d< Complex const > const r2BasisValues = r2Grid.basisValues.toViewConst();

  IndexType const nGridR1 = r1Grid.nGrid();
  IndexType const nGridR2 = r2Grid.nGrid();
  Array3d< Cartesian< Complex > > V( nGridR1, nBasis, nBasis );

  forAll< DefaultPolicy< PolicyType > >( nGridR1,
    [=, V=V.toView()] ( IndexType const r1Idx )
    {
      for( IndexType j = 0; j < nBasis; ++j )
      {
        for( IndexType i = j; i < nBasis; ++i )
        {
          Cartesian< Complex > answer {};
          for( IndexType r2Idx = 0; r2Idx < nGridR2; ++r2Idx )
          {
            Complex const scale = conj( r2BasisValues( r2Idx, j ) ) * r2BasisValues( r2Idx, i );
            answer.scaledAdd( scale, vectorFunction( r1Idx, r2Idx ) );
          }

          V( r1Idx, j, i ) = answer;
          V( r1Idx, i, j ) = { conj( answer.x() ), conj( answer.y() ), conj( answer.z() ) };
        }
      }
    }
  );

  return V;
}

/**
 */
template< typename T >
struct RCSHartreeFock
{
  using Real = RealType< T >;

  /**
   */
  RCSHartreeFock( int const numElectrons, int const numBasisFunctions ):
    nElectrons{ numElectrons },
    basisSize{ numBasisFunctions },
    density( numBasisFunctions, numBasisFunctions ),
    fockOperator( numBasisFunctions, numBasisFunctions ),
    eigenvalues( numBasisFunctions ),
    eigenvectors( numBasisFunctions, numElectrons / 2 ),
    _support( 2 * numBasisFunctions )
  {
    LVARRAY_ERROR_IF_NE( numElectrons % 2, 0 );
  }

  /**
   */
  RCSHartreeFock( int const numSpinUp, int const numSpinDown, int const numBasisFunctions ):
    RCSHartreeFock( numSpinUp + numSpinDown, numBasisFunctions )
  {}

  /**
   */
  constexpr bool needsGradients() const
  { return false; }

  /**
   */
  int numberOfConvergenceLoops() const
  { return _iter + 1; }

  /**
   */
  RealType< T > compute(
    bool const orthogonal,
    ArrayView2d< T const > const & overlap,
    ArrayView2d< Real const > const & oneElectronTerms,
    integration::QMCGrid< Real, 3 > const & r1Grid,
    ArrayView3d< Real const > const & FjiSame );

  /**
   */
  Real highestOccupiedOrbitalEnergy() const
  { 
    return eigenvalues( nElectrons / 2 - 1 );
  }

  int const nElectrons;
  int const basisSize;
  Array2d< T > const density;
  Array2d< T, RAJA::PERM_JI > const fockOperator;
  Array1d< Real > const eigenvalues;
  Array2d< T, RAJA::PERM_JI > const eigenvectors;

private:
  int _iter = 0;
  LvArray::dense::ArrayWorkspace< T, LvArray::ChaiBuffer > _workspace;
  Array1d< int > _support;
};

/**
 */
template< typename T >
struct UOSHartreeFock
{
  using Real = RealType< T >;

  /**
   */
  UOSHartreeFock( int const numSpinUp, int const numSpinDown, int const numBasisFunctions ):
    nElectrons{ numSpinUp, numSpinDown },
    basisSize{ numBasisFunctions },
    density( 2, numBasisFunctions, numBasisFunctions ),
    fockOperator( 2, numBasisFunctions, numBasisFunctions ),
    eigenvalues( 2, numBasisFunctions ),
    eigenvectors( 2, numBasisFunctions, std::max( numSpinUp, numSpinDown ) ),
    _support( 2 * std::max( numSpinUp, numSpinDown ) )
  {}
  
  /**
   */
  constexpr bool needsGradients() const
  { return false; }

  /**
   */
  int numberOfConvergenceLoops() const
  { return _iter + 1; }

  /**
   */
  Real compute(
    bool const orthogonal,
    ArrayView2d< T const > const & overlap,
    ArrayView2d< Real const > const & oneElectronTerms,
    integration::QMCGrid< Real, 3 > const & r1Grid,
    ArrayView3d< Real const > const & Fji );

  /**
   */
  Real highestOccupiedOrbitalEnergy() const
  { 
    return std::max( eigenvalues( 0, nElectrons[ 0 ] - 1 ), eigenvalues( 1, nElectrons[ 1 ] - 1 ) );
  }

  CArray< int, 2 > const nElectrons;
  int const basisSize;
  Array3d< T > const density;
  Array3d< T, RAJA::PERM_IKJ > const fockOperator;
  Array2d< Real > const eigenvalues;
  Array3d< T, RAJA::PERM_IKJ > const eigenvectors;

private:
  int _iter = 0;
  LvArray::dense::ArrayWorkspace< T, LvArray::ChaiBuffer > _workspace;
  Array1d< int > _support;
};

/**
 */
template< typename T >
struct TCHartreeFock
{
  using Real = RealType< T >;
  using Complex = std::complex< Real >;

  /**
   */
  TCHartreeFock( int const numSpinUp, int const numSpinDown, int const numBasisFunctions ):
    nElectrons{ numSpinUp, numSpinDown },
    basisSize{ numBasisFunctions },
    density( 2, numBasisFunctions, numBasisFunctions ),
    fockOperator( 2, numBasisFunctions, numBasisFunctions ),
    occupiedOrbitalPseudoEnergy{ Array1d< Real >{ numSpinUp },
                                 Array1d< Real >{ numSpinDown } },
    occupiedOrbitals{ Array2d< Complex, RAJA::PERM_JI >{ basisSize, numSpinUp },
                      Array2d< Complex, RAJA::PERM_JI >{ basisSize, numSpinDown } },
    eigenvalues( 2, numBasisFunctions ),
    eigenvectors( 2, numBasisFunctions, numBasisFunctions )
  {}
  
  /**
   */
  constexpr bool needsGradients() const
  { return true; }

  /**
   */
  int numberOfConvergenceLoops() const
  { return _iter + 1; }

  /**
   */
  Real compute(
    bool const orthogonal,
    ArrayView2d< T const > const & overlap,
    ArrayView2d< Real const > const & oneElectronTerms,
    integration::QMCGrid< Real, 3 > const & r1Grid,
    ArrayView3d< Real const > const & FjiSame,
    ArrayView3d< Real const > const & FjiOppo,
    ArrayView3d< Cartesian< T > const > const & VjiSame,
    ArrayView3d< Cartesian< T > const > const & VjiOppo );

  /**
   */
  Real highestOccupiedOrbitalEnergy() const
  {
    return std::max( occupiedOrbitalPseudoEnergy[ 0 ][ nElectrons[ 0 ] - 1 ],
                     occupiedOrbitalPseudoEnergy[ 1 ][ nElectrons[ 1 ] - 1 ] );
  }

  CArray< int, 2 > const nElectrons;
  int const basisSize;
  Array3d< Complex > const density;
  Array3d< Complex, RAJA::PERM_IKJ > const fockOperator;

  CArray< Array1d< Real >, 2 > occupiedOrbitalPseudoEnergy;
  CArray< Array2d< Complex, RAJA::PERM_JI >, 2 > occupiedOrbitals;

private:

  int _iter = 0;
  LvArray::dense::ArrayWorkspace< Complex, LvArray::ChaiBuffer > _workspace;
  Array2d< Complex > const eigenvalues;
  Array3d< Complex, RAJA::PERM_IKJ > const eigenvectors;
};

} // namespace tcscf
