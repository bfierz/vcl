#!/bin/bash

if [ $build_type = "Debug" ]; then
  # Uploading report to CodeCov
  bash <(curl -s https://codecov.io/bash) || echo "Codecov did not collect coverage reports"
fi
