#pragma once

#include <string>

namespace tcscf
{

struct CommandLineOptions
{
  std::string caliperArgs = "";
  bool suppressMoveLogging = false;
};

// To use caliper pass the following "-c runtime-report,aggregate_across_ranks=false,max_column_width=200,profile.cuda"
CommandLineOptions parseCommandLineOptions( int argc, char ** argv );

void printHighWaterMarks();

} // namespace tcscf
