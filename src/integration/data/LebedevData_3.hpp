#pragma once

// Lebedev quadrature points of order 3.
// The ordering of the data is { polar, azimuthal, weight }.
// Angles are in radians with the polar angle between 0 and pi
// and the azimuthal angle between -pi and pi.

static constexpr long double Lebedev_3[ 6 ][ 3 ] = {
  {  1.5707963267948966192313217L,  0.0000000000000000000000000L,  2.0943951023931996810986337L },
  {  1.5707963267948966192313217L,  3.1415926535897932384626434L,  2.0943951023931996810986337L },
  {  1.5707963267948966192313217L,  1.5707963267948966192313217L,  2.0943951023931996810986337L },
  {  1.5707963267948966192313217L, -1.5707963267948966192313217L,  2.0943951023931996810986337L },
  {  0.0000000000000000000000000L,  1.5707963267948966192313217L,  2.0943951023931996810986337L },
  {  3.1415926535897932384626434L,  1.5707963267948966192313217L,  2.0943951023931996810986337L },
};
