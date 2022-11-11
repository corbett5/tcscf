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

  forAll< DefaultPolicy< Serial > >( r1Points.size( 1 ),
    [=] ( IndexType const r1Idx )
    {
      Cartesian< REAL > const r1 = { r1Points( 0, r1Idx ), r1Points( 1, r1Idx ), r1Points( 2, r1Idx ) };
      for( IndexType r2Idx = 0; r2Idx < r2Points.size( 1 ); ++r2Idx )
      {
        Cartesian< REAL > const r2 { r2Points( 0, r2Idx ), 0, r2Points( 1, r2Idx ) };
        values( r1Idx, r2Idx ) = f( r2, r1 ) * (r1Weights[ r1Idx ] * r2Weights[ r2Idx ]);
        
        // T const diff = f( r2, r1 ) - f( r1, r2 );
        // if constexpr ( std::is_same_v< T, double > )
        // {
        //   LVARRAY_ERROR_IF_GT( std::abs( diff ), 1e-10 );
        // }
        // else
        // {
        //   LVARRAY_ERROR_IF_GT( std::abs( diff.x() ), 1e-10 );
        //   LVARRAY_ERROR_IF_GT( std::abs( diff.y() ), 1e-10 );
        //   LVARRAY_ERROR_IF_GT( std::abs( diff.z() ), 1e-10 );
        // }
      }
    }
  );
}

/**
 */
template< typename T, typename REAL, typename LAMBDA >
void precomputeIntegrand(
  ArrayView2d< T > const & values,
  integration::QMCGrid< REAL, 2 > const & r1Grid,
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

  forAll< DefaultPolicy< Serial > >( r1Points.size( 1 ),
    [=] ( IndexType const r1Idx )
    {
      Cartesian< REAL > const r1 = { r1Points( 0, r1Idx ), 0, r1Points( 1, r1Idx ) };
      for( IndexType r2Idx = 0; r2Idx < r2Points.size( 1 ); ++r2Idx )
      {
        Cartesian< REAL > const r2 { r2Points( 0, r2Idx ), r2Points( 1, r2Idx ), r2Points( 2, r2Idx ) };
        values( r1Idx, r2Idx ) = f( r2, r1 ) * (r1Weights[ r1Idx ] * r2Weights[ r2Idx ]);
        
        // T const diff = f( r2, r1 ) - f( r1, r2 );
        // if constexpr ( std::is_same_v< T, double > )
        // {
        //   LVARRAY_ERROR_IF_GT( std::abs( diff ), 1e-10 );
        // }
        // else
        // {
        //   LVARRAY_ERROR_IF_GT( std::abs( diff.x() ), 1e-10 );
        //   LVARRAY_ERROR_IF_GT( std::abs( diff.y() ), 1e-10 );
        //   LVARRAY_ERROR_IF_GT( std::abs( diff.z() ), 1e-10 );
        // }
      }
    }
  );
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
    integration::QMCGrid< Real, 2 > const & r2Grid,
    ArrayView2d< Real const > const & r12Inv );

  /**
   */
  Real compute(
    bool const orthogonal,
    ArrayView2d< T const > const & overlap,
    ArrayView2d< Real const > const & oneElectronTerms,
    ArrayView4d< T const > const & twoElectronTerms );
  
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
    integration::QMCGrid< Real, 2 > const & r2Grid,
    ArrayView2d< Real const > const & r12Inv );

  /**
   */
  Real compute(
    bool const orthogonal,
    ArrayView2d< T const > const & overlap,
    ArrayView2d< Real const > const & oneElectronTerms,
    ArrayView4d< T const > const & twoElectronTerms );

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
    ArrayView4d< T const > const & twoElectronTermsSameSpin,
    ArrayView4d< T const > const & twoElectronTermsOppositeSpin,
    integration::QMCGrid< Real, 3 > const & r1Grid,
    integration::QMCGrid< Real, 2 > const & r2Grid,
    ArrayView2d< Real const > const & scalarSame,
    ArrayView2d< Real const > const & scalarOppo,
    ArrayView2d< Cartesian< Real > const > const & vectorSame21,
    ArrayView2d< Cartesian< Real > const > const & vectorOppo21,
    ArrayView2d< Cartesian< Real > const > const & vectorSame12,
    ArrayView2d< Cartesian< Real > const > const & vectorOppo12 );

  /**
   */
  Real compute(
    bool const orthogonal,
    ArrayView2d< T const > const & overlap,
    ArrayView2d< Real const > const & oneElectronTerms,
    ArrayView4d< T const > const & twoElectronTermsSameSpin,
    ArrayView4d< T const > const & twoElectronTermsOppositeSpin );

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

  /**
   */
  void constructFockOperator(
    ArrayView2d< Real const > const & oneElectronTerms,
    ArrayView4d< T const > const & twoElectronTermsSameSpin,
    ArrayView4d< T const > const & twoElectronTermsOppositeSpin );
  
  /**
   */
  T calculateEnergy(
    ArrayView2d< Real const > const & oneElectronTerms,
    ArrayView4d< T const > const & twoElectronTermsSameSpin,
    ArrayView4d< T const > const & twoElectronTermsOppositeSpin ) const;

  /**
   */
  void getNewDensity();

  int _iter = 0;
  LvArray::dense::ArrayWorkspace< Complex, LvArray::ChaiBuffer > _workspace;
  Array2d< Complex > const eigenvalues;
  Array3d< Complex, RAJA::PERM_IKJ > const eigenvectors;
};

} // namespace tcscf
