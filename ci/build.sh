#!/bin/bash

make -f ./tests/vcl.core/Makefile
make -f ./tests/vcl.components/Makefile
make -f ./tests/vcl.geometry/Makefile
make -f ./tests/vcl.math/Makefile

if [ "$compiler" = "g++-6" ] && [ "$build_type" = "Debug" ]
then
  ../ci/codecov.sh vcl_core_coverage
  ../ci/codecov.sh vcl_components_coverage
  ../ci/codecov.sh vcl_geometry_coverage
  ../ci/codecov.sh vcl_math_coverage
else
  ./bin/vcl_core_test
  ./bin/vcl_components_test
  ./bin/vcl_geometry_test
  ./bin/vcl_math_test
fi
