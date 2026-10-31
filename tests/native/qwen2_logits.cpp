#include "llama.h"

#include <charconv>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

static bool decode(
        llama_context * context, const std::vector<llama_token> & tokens,
        int begin, int count, int vocabulary, std::vector<std::vector<float>> & output) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    batch.n_tokens = count;
    for (int i = 0; i < count; ++i) {
        batch.token[i] = tokens[begin + i];
        batch.pos[i] = begin + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = true;
    }
    const int status = llama_decode(context, batch);
    llama_batch_free(batch);
    if (status != 0) {
        std::cerr << "native decode failed: " << status << '\n';
        return false;
    }
    for (int i = 0; i < count; ++i) {
        const float * logits = llama_get_logits_ith(context, i);
        if (!logits) {
            std::cerr << "native logits missing\n";
            return false;
        }
        for (int j = 0; j < vocabulary; ++j) {
            if (!std::isfinite(logits[j])) {
                std::cerr << "native logits nonfinite\n";
                return false;
            }
        }
        output.emplace_back(logits, logits + vocabulary);
    }
    return true;
}

static int run(int argc, char ** argv) {
    if (argc < 4 || argc > 19 ||
            (std::strcmp(argv[2], "full") != 0 && std::strcmp(argv[2], "incremental") != 0)) {
        std::cerr << "usage: native-qwen2 MODEL full|incremental TOKEN... (1-16 tokens)\n";
        return 1;
    }
    std::vector<llama_token> tokens;
    for (int i = 3; i < argc; ++i) {
        llama_token token;
        const char * end = argv[i] + std::strlen(argv[i]);
        const auto parsed = std::from_chars(argv[i], end, token);
        if (parsed.ec != std::errc() || parsed.ptr != end || token < 0) {
            std::cerr << "invalid token id\n";
            return 1;
        }
        tokens.push_back(token);
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.use_mmap = false;
    model_params.check_tensors = true;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(argv[1], model_params), llama_model_free);
    if (!model) {
        std::cerr << "native model load failed\n";
        return 2;
    }
    const int vocabulary = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    for (const auto token : tokens) {
        if (token >= vocabulary) {
            std::cerr << "token id exceeds vocabulary\n";
            return 1;
        }
    }

    llama_context_params params = llama_context_default_params();
    params.n_ctx = 32;
    params.n_batch = 16;
    params.n_ubatch = 16;
    params.n_seq_max = 1;
    params.n_threads = 1;
    params.n_threads_batch = 1;
    params.type_k = GGML_TYPE_F32;
    params.type_v = GGML_TYPE_F32;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.offload_kqv = false;
    params.op_offload = false;
    std::unique_ptr<llama_context, decltype(&llama_free)> context(
        llama_init_from_model(model.get(), params), llama_free);
    if (!context) {
        std::cerr << "native context creation failed\n";
        return 3;
    }

    std::vector<std::vector<float>> logits;
    const bool full = std::strcmp(argv[2], "full") == 0;
    const int count = full ? static_cast<int>(tokens.size()) : 1;
    for (int begin = 0; begin < static_cast<int>(tokens.size()); begin += count) {
        if (!decode(context.get(), tokens, begin, count, vocabulary, logits)) {
            return 4;
        }
    }
    std::cout << std::setprecision(9)
              << "{\"native_revision\":\"" << IGBUNDLE_LLAMA_REVISION
              << "\",\"mode\":\"" << argv[2] << "\",\"n_vocab\":" << vocabulary << ",\"tokens\":[";
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::cout << (i ? "," : "") << tokens[i];
    }
    std::cout << "],\"logits\":[";
    for (size_t i = 0; i < logits.size(); ++i) {
        std::cout << (i ? ",[" : "[");
        for (size_t j = 0; j < logits[i].size(); ++j) {
            std::cout << (j ? "," : "") << logits[i][j];
        }
        std::cout << ']';
    }
    std::cout << "]}\n";
    return 0;
}

int main(int argc, char ** argv) {
    llama_backend_init();
    ggml_backend_load_all();
    const int status = run(argc, argv);
    llama_backend_free();
    return status;
}
