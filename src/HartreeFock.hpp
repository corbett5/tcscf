#pragma once

#include "blasLapackInterface.hpp"

// TODO: get rid of this include and instead template AtomicRCSHartreeFock on the basis type
#include "HydrogenLikeBasis.hpp"

namespace tcscf
{

template< typename T >
struct RCSHartreeFock
{
  using Real = RealType< T >;

  RCSHartreeFock( int const numElectrons, int const numBasisFunctions ):
    nElectrons{ numElectrons },
    basisSize{ numBasisFunctions },
    density( basisSize, basisSize ),
    fockOperator( basisSize, basisSize ),
    eigenvalues( basisSize )
  {}
  
  void compute(
    bool const orthogonal,
    ArrayView2d< T const > const & overlap,
    ArrayView2d< T const > const & oneElectronTerms,
    ArrayView4d< T const > const & twoElectronTerms );

  int const nElectrons;
  int const basisSize;
  Array2d< T > const density;
  Array2d< T, RAJA::PERM_JI > const fockOperator;
  Array1d< Real > const eigenvalues;
};


struct AtomicRCSHartreeFock
{
  using Real = double;
  using BasisFunction = HydrogenLikeBasisFunction< double >;

  AtomicRCSHartreeFock(
    int const numElectrons,
    std::vector< BasisFunction > const & functions ):
    nElectrons{ numElectrons },
    basisFunctions( functions ),
    density( basisFunctions.size(), basisFunctions.size() ),
    fockOperator( basisFunctions.size(), basisFunctions.size() ),
    eigenvalues( basisFunctions.size() )
  {}

  void compute(
    ArrayView2d< Real const > const & oneElectronTerms,
    ArrayView4d< std::complex< double > const > const & twoElectronTerms );

  int const nElectrons;
  std::vector< BasisFunction > const basisFunctions;
  Array2d< std::complex< double > > const density;
  Array2d< std::complex< double >, RAJA::PERM_JI > const fockOperator;
  Array1d< Real > const eigenvalues;
};

} // namespace tcscf
