#pragma once

#include "AtomicBasis.hpp"

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
    eigenvectors( numBasisFunctions, numBasisFunctions ),
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
  Array2d< std::complex< Real > > const density;
  Array2d< std::complex< Real >, RAJA::PERM_JI > const fockOperator;
  Array1d< Real > const eigenvalues;
  Array2d< std::complex< Real >, RAJA::PERM_JI > const eigenvectors;

private:
  LvArray::dense::ArrayWorkspace< std::complex< Real >, LvArray::ChaiBuffer > _workspace;
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
    eigenvectors( 2, numBasisFunctions, numBasisFunctions ),
    _support( 2 * std::max( numSpinUp, numSpinDown ) )
  {}
  
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
  Array3d< std::complex< Real > > const density;
  Array3d< std::complex< Real >, RAJA::PERM_IKJ > const fockOperator;
  Array2d< Real > const eigenvalues;
  Array3d< std::complex< Real >, RAJA::PERM_IKJ > const eigenvectors;

private:

  /**
   */
  void constructFockOperator(
    ArrayView2d< Real const > const & oneElectronTerms,
    ArrayView4d< T const > const & twoElectronTerms );
  
  /**
   */
  T calculateEnergy( ArrayView2d< Real const > const & oneElectronTerms ) const;

  /**
   */
  void getNewDensity();

  LvArray::dense::ArrayWorkspace< std::complex< Real >, LvArray::ChaiBuffer > _workspace;
  Array1d< int > _support;
};

/**
 */
template< typename T >
struct TCHartreeFock
{
  using Real = RealType< T >;

  /**
   */
  TCHartreeFock( int const numSpinUp, int const numSpinDown, int const numBasisFunctions ):
    nElectrons{ numSpinUp, numSpinDown },
    basisSize{ numBasisFunctions },
    density( 2, numBasisFunctions, numBasisFunctions ),
    fockOperator( 2, numBasisFunctions, numBasisFunctions ),
    eigenvalues( 2, numBasisFunctions ),
    eigenvectors( 2, numBasisFunctions, numBasisFunctions ),
    _support( 2 * std::max( numSpinUp, numSpinDown ) )
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
    return std::max( eigenvalues( 0, nElectrons[ 0 ] - 1 ), eigenvalues( 1, nElectrons[ 1 ] - 1 ) );
  }

  CArray< int, 2 > const nElectrons;
  int const basisSize;
  Array3d< std::complex< Real > > const density;
  Array3d< std::complex< Real >, RAJA::PERM_IKJ > const fockOperator;
  Array2d< Real > const eigenvalues;
  Array3d< std::complex< Real >, RAJA::PERM_IKJ > const eigenvectors;

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

  LvArray::dense::ArrayWorkspace< std::complex< Real >, LvArray::ChaiBuffer > _workspace;
  Array1d< int > _support;
};

} // namespace tcscf
