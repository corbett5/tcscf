#include "../integration/TreutlerAhlrichsLebedev.hpp"

#include "../setup.hpp"
#include "../caliperInterface.hpp"
#include "../RAJAInterface.hpp"

#include "../SlaterTypeOrbital.hpp"

#include "testingCommon.hpp"

namespace tcscf::testing
{

TEST( SlaterTypeOrbital, overlap )
{
  int order = 11;
  integration::TreutlerAhlrichsLebedev< double > integrator( 1.0, 1000, order );

  for( int n = 1; n < order / 2; ++n )
  {
    for( int np = 1; np <= n; ++np )
    {
      for( int l = 0; l <= np; ++l )
      {
        for( int m = -l; m <= l; ++m )
        {
          SlaterTypeOrbital< double > b1 { 1.524, n, l, m };
          SlaterTypeOrbital< double > b2 { 2.532, np, l, m };

          std::complex< double > value = integrate( integrator,
            [&] ( CArray< double, 3 > const & rThetaPhi )
            {
              double const r = rThetaPhi[ 0 ];
              double const theta = rThetaPhi[ 1 ];
              double const phi = rThetaPhi[ 2 ];
              return std::conj( b1( { r, theta, phi } ) ) * b2( { r, theta, phi } ) * std::pow( r, 2 );
            }
          );

          std::complex< double > const diff = value - overlap( b1, b2 );
          EXPECT_LT( std::abs( diff.real() ), 1e-14 );
          EXPECT_LT( std::abs( diff.imag() ), 1e-14 );
        }
      }
    }
  }
}

} // namespace tcscf::testing


int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );

  tcscf::CommandLineOptions options = tcscf::parseCommandLineOptions( argc, argv );
  tcscf::CaliperWrapper caliperWrapper( options.caliperArgs );

  int const result = RUN_ALL_TESTS();
  tcscf::printHighWaterMarks();

  return result;
}
