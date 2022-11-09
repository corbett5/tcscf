#pragma once

#include "AtomicBasis.hpp"
#include "integration/integrateAll.hpp"

#include "dense/common.hpp"

namespace tcscf
{

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
    integration::QMCGrid< Real, 2 > const & r2Grid );

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
    integration::QMCGrid< Real, 2 > const & r2Grid );

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
