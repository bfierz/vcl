#!/bin/bash

make vcl.core.test
make vcl.components.test
make vcl.geometry.test
make vcl.math.test

if [ "$compiler" = "g++-6" ] && [ "$build_type" = "Debug" ]
then
  ../ci/codecov.sh vcl.core.test.coverage || exit 1
  ../ci/codecov.sh vcl.components.test.coverage || exit 1
  ../ci/codecov.sh vcl.geometry.test.coverage || exit 1
  ../ci/codecov.sh vcl.math.test.coverage || exit 1
else
  pushd src/tests
  ctest --no-compress-output -T Test || /bin/true
  popd
fi
