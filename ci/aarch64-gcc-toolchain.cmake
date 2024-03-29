# Source: https://github.com/vpetrigo/arm-cmake-toolchains
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

if(MINGW OR CYGWIN OR WIN32)
    set(UTIL_SEARCH_CMD where)
elseif(UNIX OR APPLE)
    set(UTIL_SEARCH_CMD which)
endif()

set(TOOLCHAIN_TRIPLE aarch64-linux-gnu)
set(TOOLCHAIN_PREFIX ${TOOLCHAIN_TRIPLE}-)

execute_process(
    COMMAND ${UTIL_SEARCH_CMD} ${TOOLCHAIN_PREFIX}gcc
    OUTPUT_VARIABLE BINUTILS_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
get_filename_component(ARM_TOOLCHAIN_DIR ${BINUTILS_PATH} DIRECTORY)

# Without that flag CMake is not able to pass test compilation check
if(${CMAKE_VERSION} VERSION_EQUAL "3.6.0" OR ${CMAKE_VERSION} VERSION_GREATER "3.6")
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
else()
    set(CMAKE_EXE_LINKER_FLAGS_INIT "--specs=nosys.specs")
endif()

set(CMAKE_C_COMPILER ${TOOLCHAIN_PREFIX}gcc)
set(CMAKE_ASM_COMPILER ${CMAKE_C_COMPILER})
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}g++)

set(CMAKE_OBJCOPY ${ARM_TOOLCHAIN_DIR}/${TOOLCHAIN_PREFIX}objcopy CACHE INTERNAL "objcopy tool")
set(CMAKE_SIZE_UTIL ${ARM_TOOLCHAIN_DIR}/${TOOLCHAIN_PREFIX}size CACHE INTERNAL "size tool")

# Theoretically, sysroot limits the search of libraries to the installation location of the cross-compilation SDK.
# For some reason SYSROOT is applied twice on Ubuntu, thus we ignore SYSROOT
# set(CMAKE_SYSROOT ${ARM_TOOLCHAIN_DIR}/../${TOOLCHAIN_TRIPLE})
set(CMAKE_SYSROOT /)
set(CMAKE_FIND_ROOT_PATH ${BINUTILS_PATH})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Configure the emulator
set(CMAKE_CROSSCOMPILING_EMULATOR qemu-aarch64 -L ${ARM_TOOLCHAIN_DIR}/../${TOOLCHAIN_TRIPLE})
