#!/bin/bash

make -f ./tests/vcl.core/Makefile
make -f ./tests/vcl.components/Makefile
make -f ./tests/vcl.geometry/Makefile
make -f ./tests/vcl.math/Makefile

if [ "$build_type" = "Debug" ]; then
  make -f ./tests/Makefile vcl_core_coverage
  make -f ./tests/Makefile vcl_components_coverage
  make -f ./tests/Makefile vcl_geometry_coverage
  make -f ./tests/Makefile vcl_math_coverage
else
  ./bin/vcl_core_test
  ./bin/vcl_components_test
  ./bin/vcl_geometry_test
  ./bin/vcl_math_test
fi
