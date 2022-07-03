#include  "../OchiBasis.hpp"
#include  "../HydrogenLikeBasis.hpp"

#include "testingCommon.hpp"

namespace tcscf::testing
{

TEST( OchiBasis, orthogonal )
{
  double const alpha = 1.0;

  for( int n1 = 0; n1 < 2; ++n1 )
  {
    for( int l1 = 0; l1 < 2; ++l1 )
    {
      for( int m1 = -l1; m1 <= l1; ++m1 )
      {
        for( int n2 = 0; n2 < 2; ++n2 )
        {
          for( int l2 = 0; l2 < 2; ++l2 )
          {
            for( int m2 = -l2; m2 <= l2; ++m2 )
            {
              OchiBasisFunction< double > b1( alpha, n1, l1, m1 );
              OchiBasisFunction< double > b2( alpha, n2, l2, m2 );

              std::complex< double > value = overlap( b1, b2 );
              bool const delta = (n1 == n2) && (l1 == l2) && (m1 == m2);
              EXPECT_COMPLEX_NEAR( value, delta, 1e-5 );
            }
          }
        }
      }
    }
  }
}

TEST( HydrogenLikeBasisFunction, orthogonal )
{
  int const Z = 1;

  for( int n1 = 0; n1 < 3; ++n1 )
  {
    for( int l1 = 0; l1 < n1; ++l1 )
    {
      for( int m1 = -l1; m1 <= l1; ++m1 )
      {
        for( int n2 = 0; n2 < 3; ++n2 )
        {
          for( int l2 = 0; l2 < n2; ++l2 )
          {
            for( int m2 = -l2; m2 <= l2; ++m2 )
            {
              HydrogenLikeBasisFunction< double > b1( Z, n1, l1, m1 );
              HydrogenLikeBasisFunction< double > b2( Z, n2, l2, m2 );

              std::complex< double > value = overlap( b1, b2 );
              bool const delta = (n1 == n2) && (l1 == l2) && (m1 == m2);
              EXPECT_COMPLEX_NEAR( value, delta, 1e-5 );
            }
          }
        }
      }
    }
  }
}

} // namespace tcscf::testing
