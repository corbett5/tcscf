#include  "../HartreeFock.hpp"
#include "../OchiBasis.hpp"


#include "testingCommon.hpp"

namespace tcscf::testing
{

TEST( HartreeFock, Helium )
{
  double const alpha = 1;
  int const Z = 2;

  std::vector< OchiBasisFunction< double > > basisFunctions = {
    OchiBasisFunction< double >{ alpha, 1, 0, +0 },
    OchiBasisFunction< double >{ alpha, 1, 1, -1 },
    OchiBasisFunction< double >{ alpha, 1, 1, +0 },
    OchiBasisFunction< double >{ alpha, 1, 1, +1 },
    OchiBasisFunction< double >{ alpha, 2, 0, +0 },
    OchiBasisFunction< double >{ alpha, 2, 1, -1 },
    OchiBasisFunction< double >{ alpha, 2, 1, +0 },
    OchiBasisFunction< double >{ alpha, 2, 1, +1 },
    // OchiBasisFunction< double >{ alpha, 2, 2, -2 },
    // OchiBasisFunction< double >{ alpha, 2, 2, -1 },
    // OchiBasisFunction< double >{ alpha, 2, 2, 0 },
    // OchiBasisFunction< double >{ alpha, 2, 2, +1 },
    // OchiBasisFunction< double >{ alpha, 2, 2, +2 }
  };

  IndexType const nBasis = basisFunctions.size();

  Array2d< std::complex< double > > const oneElectronTerms = computeCoreMatrix( Z, basisFunctions );
  Array4d< std::complex< double > > const twoElectronTerms = computeR12Matrix( basisFunctions );

  Array2d< std::complex< double > > density( nBasis, nBasis );
  Array2d< std::complex< double > > overlap;

  RCSHartreeFock< std::complex< double > > hfCalculator( 2, nBasis );
  hfCalculator.compute( true, overlap, oneElectronTerms, twoElectronTerms );
}

} // namespace tcscf::testing
