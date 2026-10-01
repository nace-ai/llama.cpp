#pragma once

#include "llama.h"
#include "server-common.h"

#include <cstdint>
#include <string>
#include <vector>

// Fork-only System One encoder. Not an upstream llama-server route.
struct edlm_encoded_group {
    int32_t decide = -1;
    std::vector<int32_t> options;
};

struct edlm_question_meta {
    std::string id;
    std::string type;
    std::vector<std::string> keys;
    std::vector<std::string> legend;
};

struct edlm_encoded {
    std::vector<llama_token> tokens;
    std::vector<int32_t> segments;
    std::vector<edlm_encoded_group> groups;
    std::vector<edlm_question_meta> meta;
};

// Throws std::runtime_error when the request is invalid or does not fit.
edlm_encoded edlm_encode_systemone(const llama_vocab * vocab, const json & body);

json edlm_format_answers(const std::vector<std::vector<float>> & logits, const std::vector<edlm_question_meta> & meta);
