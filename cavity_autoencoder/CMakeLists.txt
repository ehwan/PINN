cmake_minimum_required(VERSION 3.5)

# for clangd auto completion
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

if( UNIX )
  set( CMAKE_CXX_COMPILER /usr/bin/clang++ )
  set( CMAKE_C_COMPILER /usr/bin/clang )
endif()

project( Cavity CXX )
find_package( Eigen3 )
add_executable( Cavity
  cavity.cpp
)
target_link_libraries(
  Cavity PUBLIC
  Eigen3::Eigen
)
set_target_properties(
  Cavity
  PROPERTIES
  CXX_STANDARD 17
)