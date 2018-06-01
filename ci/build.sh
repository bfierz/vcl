#!/bin/bash

make -f ./tests/vcl.core/Makefile
make -f ./tests/vcl.components/Makefile
make -f ./tests/vcl.geometry/Makefile
#make -f ./tests/vcl.graphics.opengl/Makefile
make -f ./tests/vcl.math/Makefile

if [ "$compiler" = "g++-6" ] && [ "$build_type" = "Debug" ]
then
  ../ci/codecov.sh vcl_core_coverage || exit 1
  ../ci/codecov.sh vcl_components_coverage || exit 1
  ../ci/codecov.sh vcl_geometry_coverage || exit 1
  #../ci/codecov.sh vcl_graphics_opengl_coverage || exit 1
  ../ci/codecov.sh vcl_math_coverage || exit 1
else
  ./bin/vcl_core_test || exit 1
  ./bin/vcl_components_test || exit 1
  ./bin/vcl_geometry_test || exit 1
  #./bin/vcl_graphics_opengl_test || exit 1
  ./bin/vcl_math_test || exit 1
fi
