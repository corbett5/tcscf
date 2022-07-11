#pragma once

#include "LebedevData_3.hpp"
#include "LebedevData_5.hpp"
#include "LebedevData_7.hpp"
#include "LebedevData_9.hpp"
#include "LebedevData_11.hpp"
#include "LebedevData_13.hpp"
#include "LebedevData_15.hpp"
#include "LebedevData_17.hpp"
#include "LebedevData_19.hpp"
#include "LebedevData_21.hpp"
#include "LebedevData_23.hpp"
#include "LebedevData_25.hpp"
#include "LebedevData_27.hpp"
#include "LebedevData_29.hpp"
#include "LebedevData_31.hpp"
#include "LebedevData_35.hpp"
#include "LebedevData_41.hpp"
#include "LebedevData_47.hpp"
#include "LebedevData_53.hpp"
#include "LebedevData_59.hpp"
#include "LebedevData_65.hpp"
#include "LebedevData_71.hpp"
#include "LebedevData_77.hpp"
#include "LebedevData_83.hpp"
#include "LebedevData_89.hpp"
#include "LebedevData_95.hpp"
#include "LebedevData_101.hpp"
#include "LebedevData_107.hpp"
#include "LebedevData_113.hpp"
#include "LebedevData_119.hpp"
#include "LebedevData_125.hpp"
#include "LebedevData_131.hpp"

#define REGISTER_LEBEDEV( N )   RegisterLebedev Lebedev_register_ ## N ( N, Lebedev_ ## N )

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

REGISTER_LEBEDEV( 3 );
REGISTER_LEBEDEV( 5 );
REGISTER_LEBEDEV( 7 );
REGISTER_LEBEDEV( 9 );
REGISTER_LEBEDEV( 11 );
REGISTER_LEBEDEV( 13 );
REGISTER_LEBEDEV( 15 );
REGISTER_LEBEDEV( 17 );
REGISTER_LEBEDEV( 19 );
REGISTER_LEBEDEV( 21 );
REGISTER_LEBEDEV( 23 );
REGISTER_LEBEDEV( 25 );
REGISTER_LEBEDEV( 27 );
REGISTER_LEBEDEV( 29 );
REGISTER_LEBEDEV( 31 );
REGISTER_LEBEDEV( 35 );
REGISTER_LEBEDEV( 41 );
REGISTER_LEBEDEV( 47 );
REGISTER_LEBEDEV( 53 );
REGISTER_LEBEDEV( 59 );
REGISTER_LEBEDEV( 65 );
REGISTER_LEBEDEV( 71 );
REGISTER_LEBEDEV( 77 );
REGISTER_LEBEDEV( 83 );
REGISTER_LEBEDEV( 89 );
REGISTER_LEBEDEV( 95 );
REGISTER_LEBEDEV( 101 );
REGISTER_LEBEDEV( 107 );
REGISTER_LEBEDEV( 113 );
REGISTER_LEBEDEV( 119 );
REGISTER_LEBEDEV( 125 );
REGISTER_LEBEDEV( 131 );

