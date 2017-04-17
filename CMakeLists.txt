cmake_minimum_required(VERSION 2.8)
project( CTPN )

file( GLOB LIB_SRCS "src/*.cpp" )
file( GLOB TEST_SRCS ${LIB_SRCS} "test/*.cpp" )
#file( GLOB JNI_SRCS  ${LIB_SRCS} "./jni/*.cpp" )

SET( OPENCV_Path "/home/hzqiaohan/.local" ) 
SET( OPENCV_INCLUDE "${OPENCV_Path}/include" "${OPENCV_Path}/include/opencv" "${OPENCV_Path}/include/opencv2" ) 
SET( OPENCV_LIB "${OPENCV_Path}/lib" ) 

SET( CAFFE_PATH "/home/hzqiaohan/OCR/CTPN/caffe")
SET( CAFFE_INC "${CAFFE_PATH}/include")
SET( CAFFE_LIB "${CAFFE_PATH}/build/lib")

INCLUDE_DIRECTORIES(${CAFFE_INC} ${OPENCV_INCLUDE} "src")
LINK_DIRECTORIES( ${CAFFE_LIB} ${OPENCV_LIB} "/usr/local/cuda/lib64" )

SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -D__STDC_CONSTANT_MACROS" )

add_executable( test ${TEST_SRCS} )
#target_link_libraries( IDrec -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_features2d -lopencv_legacy -lboost_filesystem -lboost_system )
target_link_libraries( test -lglog -lcaffe -lprotobuf -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_features2d -lopencv_legacy -lboost_filesystem -lboost_system)
#add_library( deskew SHARED ${LIB_SRCS}  )
#add_library( deskewjni SHARED ${JNI_SRCS}  )
