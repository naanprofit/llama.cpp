#include "ggml-backend.h"
#include "ggml-rpc.h"
#include "ggml.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

struct roundtrip_params {
    std::string endpoint;
    size_t      bytes;
    size_t      n_threads;
    bool        allow_public;
};

static void set_env_flag(const char * name, const std::string & value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

static bool run_roundtrip(const roundtrip_params & params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!dev) {
        fprintf(stderr, "No CPU backend available for roundtrip test\n");
        return false;
    }

    set_env_flag("GGML_RPC_SINGLE_SHOT", "1");
    if (params.allow_public) {
        set_env_flag("GGML_RPC_ALLOW_ANY", "1");
    }
    set_env_flag("GGML_RPC_RDMA_BULK_MB", std::to_string(params.bytes / (1024 * 1024)));

    std::thread server_thread([&]() {
        ggml_backend_dev_t devs[1] = { dev };
        ggml_backend_rpc_start_server(params.endpoint.c_str(), nullptr, params.n_threads, 1, devs);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    ggml_init_params init_params = {
        /*.mem_size   =*/ggml_tensor_overhead(),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(init_params);
    if (!ctx) {
        fprintf(stderr, "failed to initialize ggml context\n");
        server_thread.join();
        return false;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_rpc_buffer_type(params.endpoint.c_str(), 0);
    if (!buft) {
        fprintf(stderr, "failed to create RPC buffer type\n");
        ggml_free(ctx);
        server_thread.join();
        return false;
    }

    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, params.bytes);
    if (!buffer) {
        fprintf(stderr, "failed to allocate RPC buffer of %zu bytes\n", params.bytes);
        ggml_free(ctx);
        server_thread.join();
        return false;
    }

    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, params.bytes);
    tensor->buffer       = buffer;
    tensor->data         = ggml_backend_buffer_get_base(buffer);

    std::vector<uint8_t> payload(params.bytes);
    for (size_t i = 0; i < params.bytes; ++i) {
        payload[i] = static_cast<uint8_t>(i & 0xff);
    }
    std::vector<uint8_t> roundtrip(params.bytes, 0);

    const auto start_set = std::chrono::steady_clock::now();
    ggml_backend_tensor_set(tensor, payload.data(), 0, params.bytes);
    const auto end_set = std::chrono::steady_clock::now();

    const auto start_get = std::chrono::steady_clock::now();
    ggml_backend_tensor_get(tensor, roundtrip.data(), 0, params.bytes);
    const auto end_get = std::chrono::steady_clock::now();

    const double mb    = static_cast<double>(params.bytes) / (1024.0 * 1024.0);
    const double set_s = std::chrono::duration<double>(end_set - start_set).count();
    const double get_s = std::chrono::duration<double>(end_get - start_get).count();

    size_t mismatches = 0;
    for (size_t i = 0; i < params.bytes; ++i) {
        if (payload[i] != roundtrip[i]) {
            ++mismatches;
            if (mismatches > 8) {
                break;
            }
        }
    }

    printf("Endpoint: %s\n", params.endpoint.c_str());
    printf("  bytes: %zu\n", params.bytes);
    printf("  set bandwidth: %.2f MiB/s\n", set_s > 0 ? mb / set_s : 0.0);
    printf("  get bandwidth: %.2f MiB/s\n", get_s > 0 ? mb / get_s : 0.0);
    printf("  status: %s\n", mismatches == 0 ? "OK" : "FAILED");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);

    server_thread.join();
    return mismatches == 0;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    std::string tcp_endpoint = "127.0.0.1:50052";
    std::string rdma_endpoint;
    size_t      bulk_mb      = 256;
    size_t      n_threads    = std::max(1U, std::thread::hardware_concurrency() / 2);
    bool        allow_public = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--tcp") {
            if (++i >= argc) {
                fprintf(stderr, "missing value for --tcp\n");
                return 1;
            }
            tcp_endpoint = argv[i];
        } else if (arg == "--rdma") {
            if (++i >= argc) {
                fprintf(stderr, "missing value for --rdma\n");
                return 1;
            }
            rdma_endpoint = argv[i];
        } else if (arg == "--bulk-mb") {
            if (++i >= argc) {
                fprintf(stderr, "missing value for --bulk-mb\n");
                return 1;
            }
            bulk_mb = static_cast<size_t>(std::stoul(argv[i]));
        } else if (arg == "--threads") {
            if (++i >= argc) {
                fprintf(stderr, "missing value for --threads\n");
                return 1;
            }
            n_threads = static_cast<size_t>(std::stoul(argv[i]));
        } else if (arg == "--allow-public") {
            allow_public = true;
        } else if (arg == "--help" || arg == "-h") {
            printf(
                "Usage: %s [--tcp tcp://host:port] [--rdma rdma://host:port] [--bulk-mb N] [--threads N] "
                "[--allow-public]\n",
                argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    const size_t     bytes  = bulk_mb * 1024ull * 1024ull;
    roundtrip_params params = { tcp_endpoint, bytes, n_threads, allow_public };

    bool ok = run_roundtrip(params);

#ifdef GGML_RPC_RDMA
    if (!rdma_endpoint.empty()) {
        roundtrip_params rdma_params = { rdma_endpoint, bytes, n_threads, allow_public };
        ok                           = run_roundtrip(rdma_params) && ok;
    }
#else
    if (!rdma_endpoint.empty()) {
        fprintf(stderr, "RDMA endpoint provided but GGML_RPC_RDMA is not enabled in this build\n");
        ok = false;
    }
#endif

    return ok ? 0 : 1;
}
