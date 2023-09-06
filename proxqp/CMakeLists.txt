Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.0.1 # or a later release
)

FetchContent_MakeAvailable(Catch2)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(tests test.cpp)
target_link_libraries(tests 
    PRIVATE 
    Catch2::Catch2WithMain
    Eigen3::Eigen
    )
