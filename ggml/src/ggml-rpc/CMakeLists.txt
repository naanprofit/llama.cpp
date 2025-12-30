message(STATUS "Using RPC backend")

ggml_add_backend_library(ggml-rpc
                         ggml-rpc.cpp
                        )

if (GGML_RPC_RDMA AND NOT WIN32)
    find_library(RDMACM_LIB rdmacm)
    find_library(IBVERBS_LIB ibverbs)
    if (RDMACM_LIB AND IBVERBS_LIB)
        target_compile_definitions(ggml-rpc PRIVATE GGML_RPC_RDMA)
        target_link_libraries(ggml-rpc PRIVATE ${RDMACM_LIB} ${IBVERBS_LIB})
    else()
        message(WARNING "GGML_RPC_RDMA requested, but rdmacm/ibverbs not found; RDMA transport will be unavailable")
    endif()
endif()

if (WIN32)
    target_link_libraries(ggml-rpc PRIVATE ws2_32)
endif()
