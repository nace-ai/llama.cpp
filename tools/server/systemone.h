#pragma once

#include "llama.h"
#include "server-common.h"

#include <cstdint>
#include <string>
#include <vector>

// POST /v1/systemone. Protocol only. The loaded architecture scores the encoded record.
constexpr int SYSTEMONE_MAX_OPTIONS = 255;
constexpr int SYSTEMONE_MAX_STATE = 8192;
constexpr int SYSTEMONE_MAX_BRANCH = 8192;
constexpr int SYSTEMONE_MAX_PACKED = 16384;

struct systemone_group {
    int32_t decide = -1;
    std::vector<int32_t> options;
};

struct systemone_question {
    std::string id;
    std::string type;
    std::vector<std::string> keys;
    std::vector<std::string> legend;
};

struct systemone_encoded {
    std::vector<llama_token> tokens;
    std::vector<int32_t> segments;
    std::vector<llama_pos> positions;
    std::vector<systemone_group> groups;
    std::vector<systemone_question> meta;
    int32_t n_state = 0;
};

// Throws std::runtime_error when the request is invalid or does not fit.
systemone_encoded systemone_encode(const llama_vocab * vocab, const json & body);

json systemone_format_answers(const std::vector<std::vector<float>> & logits, const std::vector<systemone_question> & meta);

// Tokens of a Python json.dumps of the answers. Not generated tokens.
int systemone_output_tokens(const llama_vocab * vocab, const json & answers);
