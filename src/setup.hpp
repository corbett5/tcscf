#pragma once

#include <string>

namespace tcscf
{

struct CommandLineOptions
{
  int nMax;
  int lMax;
  double initialAlpha = -1;
  int r1GridSize = -1;
  int r2GridSize = -1;
  std::string caliperArgs = "";
  bool suppressMoveLogging = false;
};

// To use caliper pass the following "-c runtime-report,aggregate_across_ranks=false,max_column_width=200,profile.cuda"
CommandLineOptions parseCommandLineOptions( int argc, char ** argv );

void printHighWaterMarks();

} // namespace tcscf
