set(TARGET whisper-streaming-test)

add_executable(${TARGET}
    whisper-streaming-decoder.cpp
    test-streaming.cpp
)

include(DefaultTargetOptions)

target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/examples
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(${TARGET} PRIVATE common whisper ${CMAKE_THREAD_LIBS_INIT})

install(TARGETS ${TARGET} RUNTIME)
