#include  "../LvArrayInterface.hpp"
#include "../blasLapackInterface.hpp"

#include "testingCommon.hpp"

namespace tcscf::testing
{

TEST(blasLapack, eigenvalues)
{
  tcscf::Array2d< double, RAJA::PERM_JI > test( 3, 3 );
  test(0, 0) = 3;
  test(1, 1) = 2;
  test(2, 2) = -4;


  tcscf::Array1d< std::complex< double > > eigenvalues( 3 );

  tcscf::eigenvalues( test, eigenvalues );

  EXPECT_DOUBLE_EQ( eigenvalues[ 0 ].real(), 3 );
  EXPECT_DOUBLE_EQ( eigenvalues[ 0 ].imag(), 0 );
  
  EXPECT_DOUBLE_EQ( eigenvalues[ 1 ].real(), 2 );
  EXPECT_DOUBLE_EQ( eigenvalues[ 1 ].imag(), 0 );
  
  EXPECT_DOUBLE_EQ( eigenvalues[ 2 ].real(), -4 );
  EXPECT_DOUBLE_EQ( eigenvalues[ 2 ].imag(), 0 );
}

TEST( blasLapack, hermitianEigendecomposition )
{
  tcscf::Array2d< std::complex< double >, RAJA::PERM_JI > matrix( 3, 3 );
  matrix(0, 0) = 3;
  matrix(1, 1) = 2;
  matrix(2, 2) = -4;

  tcscf::Array1d< double > eigenvalues( 3 );

  tcscf::hermitianEigendecomposition( matrix, eigenvalues );

  EXPECT_DOUBLE_EQ( eigenvalues[ 0 ], -4 );
  EXPECT_COMPLEX_EQ( matrix(0, 0), 0 );
  EXPECT_COMPLEX_EQ( matrix(0, 1), 0 );
  EXPECT_COMPLEX_EQ( matrix(0, 2), 1 );
  
  EXPECT_DOUBLE_EQ( eigenvalues[ 1 ], 2 );
  EXPECT_COMPLEX_EQ( matrix(1, 0), 0 );
  EXPECT_COMPLEX_EQ( matrix(1, 1), 1 );
  EXPECT_COMPLEX_EQ( matrix(1, 2), 0 );
  
  EXPECT_DOUBLE_EQ( eigenvalues[ 2 ], 3 );
  EXPECT_COMPLEX_EQ( matrix(2, 0), 1 );
  EXPECT_COMPLEX_EQ( matrix(2, 1), 0 );
  EXPECT_COMPLEX_EQ( matrix(2, 2), 0 );
}

} // namespace tcscf::testing
