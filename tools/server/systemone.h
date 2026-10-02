#pragma once

#include "llama.h"
#include "server-common.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

// POST /v1/systemone. Protocol only. The loaded architecture scores the encoded record.
// The position window is 32768. The recommended serving context is 16384.
constexpr int SYSTEMONE_MAX_OPTIONS = 255;
constexpr int SYSTEMONE_CONTEXT_LENGTH = 32768;
constexpr int SYSTEMONE_RECOMMENDED_CONTEXT = 16384;
constexpr int SYSTEMONE_MAX_STATE = SYSTEMONE_RECOMMENDED_CONTEXT;
constexpr int SYSTEMONE_MAX_BRANCH = SYSTEMONE_RECOMMENDED_CONTEXT;
constexpr int SYSTEMONE_MAX_PACKED = SYSTEMONE_RECOMMENDED_CONTEXT;
constexpr int SYSTEMONE_POSITION_LIMIT = SYSTEMONE_CONTEXT_LENGTH;

// Missing or invalid keeps fallback. Never above the position window.
inline int systemone_max(const char * name, int fallback) {
    const char * env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return fallback;
    }
    char * end = nullptr;
    const long value = std::strtol(env, &end, 10);
    if (end == env || *end != '\0' || value < 1 || value > SYSTEMONE_POSITION_LIMIT) {
        return fallback;
    }
    return (int) value;
}

// SYSTEMONE_CONTEXT sets every cap. A specific SYSTEMONE_MAX_* overrides that one cap.
inline int systemone_limit(const char * specific) {
    const char * env = std::getenv(specific);
    if (env != nullptr && env[0] != '\0') {
        return systemone_max(specific, SYSTEMONE_RECOMMENDED_CONTEXT);
    }
    return systemone_max("SYSTEMONE_CONTEXT", SYSTEMONE_RECOMMENDED_CONTEXT);
}

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
