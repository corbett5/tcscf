#include <gtest/gtest.h>

#define EXPECT_COMPLEX_EQ( z1, z2 ) \
  EXPECT_DOUBLE_EQ( std::real( z1 ), std::real( z2 ) ); \
  EXPECT_DOUBLE_EQ( std::imag( z1 ), std::imag( z2 ) )

#define EXPECT_COMPLEX_NEAR( z1, z2, absError ) \
  EXPECT_NEAR( std::real( z1 ), std::real( z2 ), absError ); \
  EXPECT_NEAR( std::imag( z1 ), std::imag( z2 ), absError )

