CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0)


PROJECT(xgboostpp)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")

add_executable(test_xgb test.cpp)
INCLUDE_DIRECTORIES("/usr/local/lib/python3.6/dist-packages/xgboost/include/" "./")
LINK_DIRECTORIES("/usr/local/lib/python3.6/dist-packages/xgboost/lib")
find_library(XGBOOST xgboost HINTS "/usr/local/lib/python3.6/dist-packages/xgboost/lib")
target_link_libraries(test_xgb PUBLIC ${XGBOOST})
