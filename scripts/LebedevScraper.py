import requests
from decimal import Decimal

orders = (3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25,
          27, 29, 31, 35, 41, 47, 53, 59, 65, 71, 77,
          83, 89, 95, 101, 107, 113, 119, 125, 131)

pi = Decimal('3.1415926535897932384626433832795028841971693993751058209749445923')

header ="""#pragma once

// Lebedev quadrature points of order {0}.
// The ordering of the data is {{ polar, azimuthal, weight }}.
// Angles are in radians with the polar angle between 0 and pi
// and the azimuthal angle between -pi and pi.

static constexpr long double Lebedev_{0}[ {1} ][ 3 ] = {{"""

body = """
#define REGISTER_LEBEDEV( N ) \
  RegisterLebedev Lebedev_register_ ## N ( N, Lebedev_ ## N )

static std::unordered_map< int, std::pair< int, long double const * > > lebedevCoefficients;

template< int N >
struct RegisterLebedev
{
  RegisterLebedev( int const order, long double const (&values)[ N ][ 3 ] )
  {
    LVARRAY_ERROR_IF_NE( lebedevCoefficients.count( order), 0 );
    lebedevCoefficients[ order ] = { N, &values[ 0 ][ 0 ] };
  }
};
"""

baseUrl = 'https://people.sc.fsu.edu/~jburkardt/datasets/sphere_lebedev_rule/lebedev_{:03}.txt'
for order in orders:
    lines = requests.get(baseUrl.format(order)).text.split('\n')

    with open(f'./tmp/LebedevData_{order}.hpp', 'w') as f:

        nPoints = 0
        for line in lines:
            if line:
                nPoints += 1
        
        print(header.format(order, nPoints), file=f)

        for line in lines:
            if line:
                azimuthal, polar, weight = line.split()
                azimuthal = Decimal(azimuthal) / 180 * pi
                polar = Decimal(polar) / 180 * pi
                weight = 4 * pi * Decimal(weight)
                print(f'  {{ {polar:28.25f}L, {azimuthal:28.25f}L, {weight:28.25f}L }},', file=f)

        print('};\n', file=f)


with open('./tmp/LebedevData.hpp', 'w') as f:
    print('#pragma once\n', file=f)

    for order in orders:
        print(f'#include "LebedevData_{order}.hpp"', file=f)

    print(body, file=f)

    for order in orders:
        print(f'REGISTER_LEBEDEV( {order} );', file=f)
    
    print('', file=f)
    
