# Cross-compile Windows x64 binaries on Linux with MinGW-w64 (posix threads).
# Test binaries run under Wine; for Direct3D 12, put the vkd3d-proton DLLs
# next to them (see doc in .devcontainer/Dockerfile).
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

set(TOOLCHAIN_TRIPLE x86_64-w64-mingw32)

set(CMAKE_C_COMPILER ${TOOLCHAIN_TRIPLE}-gcc-posix)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_TRIPLE}-g++-posix)
set(CMAKE_RC_COMPILER ${TOOLCHAIN_TRIPLE}-windres)

set(CMAKE_FIND_ROOT_PATH /usr/${TOOLCHAIN_TRIPLE})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Don't depend on the MinGW runtime DLLs at run time
set(CMAKE_EXE_LINKER_FLAGS_INIT "-static")

set(CMAKE_CROSSCOMPILING_EMULATOR wine)
