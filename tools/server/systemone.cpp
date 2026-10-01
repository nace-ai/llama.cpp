#include "systemone.h"

#include "common.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace {

const char * kDelims[] = {
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
};

std::string render_value(const json & value, int indent);

std::string shield_specials(const std::string & text) {
    static const std::regex re("<\\|([A-Za-z0-9_]+)\\|>");
    return std::regex_replace(text, re, "<\xC2\xA6$1\xC2\xA6>");
}

std::string render_scalar(const json & value) {
    if (value.is_null()) {
        return "";
    }
    if (value.is_string()) {
        return value.get<std::string>();
    }
    if (value.is_boolean()) {
        return value.get<bool>() ? "True" : "False";
    }
    if (value.is_number_integer()) {
        return std::to_string(value.get<long long>());
    }
    if (value.is_number()) {
        std::ostringstream out;
        out << value.get<double>();
        return out.str();
    }
    return render_value(value, 0);
}

std::string render_value(const json & value, int indent) {
    const std::string pad(indent * 2, ' ');
    if (value.is_array()) {
        std::string out;
        for (size_t i = 0; i < value.size(); ++i) {
            if (i) {
                out += '\n';
            }
            std::string child = render_value(value.at(i), indent + 1);
            size_t cut = 0;
            while (cut < child.size() && (child[cut] == ' ' || child[cut] == '\t')) {
                ++cut;
            }
            out += pad + "- " + child.substr(cut);
        }
        return out;
    }
    if (value.is_object()) {
        std::string out;
        bool first = true;
        for (const auto & [key, child] : value.items()) {
            if (!first) {
                out += '\n';
            }
            first = false;
            if (child.is_object() || child.is_array()) {
                out += pad + key + ":\n" + render_value(child, indent + 1);
            } else {
                out += pad + key + ": " + render_scalar(child);
            }
        }
        return out;
    }
    return render_scalar(value);
}

llama_token one_special(const llama_vocab * vocab, const char * text) {
    const std::vector<llama_token> ids = common_tokenize(vocab, text, false, true);
    if (ids.size() != 1) {
        throw std::runtime_error(std::string("delimiter is not one token: ") + text);
    }
    return ids[0];
}

std::vector<llama_token> user_tokens(const llama_vocab * vocab, const std::string & text) {
    if (text.empty()) {
        return {};
    }
    return common_tokenize(vocab, shield_specials(text), false, false);
}

std::string option_text(const std::string & name, const json * desc) {
    if (desc == nullptr || desc->is_null()) {
        return name;
    }
    const std::string rendered = render_value(*desc, 0);
    if (rendered.empty()) {
        return name;
    }
    return name + ": " + rendered;
}

double round4(double value) {
    return std::round(value * 10000.0) / 10000.0;
}

std::vector<double> softmax(const std::vector<float> & logits) {
    if (logits.empty()) {
        throw std::runtime_error("question has no options");
    }
    double peak = logits[0];
    for (float logit : logits) {
        peak = std::max(peak, (double) logit);
    }
    std::vector<double> probs(logits.size());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp((double) logits[i] - peak);
        sum += probs[i];
    }
    if (sum == 0.0) {
        throw std::runtime_error("pointer logits could not be normalized");
    }
    for (double & prob : probs) {
        prob /= sum;
    }
    return probs;
}

}  // namespace

systemone_encoded systemone_encode(const llama_vocab * vocab, const json & body) {
    if (!body.is_object() || !body.contains("state") || !body.contains("questions") || !body.at("questions").is_object()) {
        throw std::runtime_error("request needs state and questions");
    }
    if (body.at("questions").size() == 0) {
        throw std::runtime_error("request needs at least one question");
    }

    llama_token delim[5];
    for (int i = 0; i < 5; ++i) {
        delim[i] = one_special(vocab, kDelims[i]);
    }

    const std::vector<llama_token> state_tokens = user_tokens(vocab, render_value(body.at("state"), 0));
    if ((int) state_tokens.size() + 1 > SYSTEMONE_MAX_STATE) {
        throw std::runtime_error("state exceeds 8192 tokens");
    }

    systemone_encoded out;
    out.tokens.push_back(delim[0]);
    out.tokens.insert(out.tokens.end(), state_tokens.begin(), state_tokens.end());
    out.segments.assign(out.tokens.size(), 0);
    out.n_state = (int32_t) out.tokens.size();
    out.positions.resize(out.tokens.size());
    for (int32_t i = 0; i < out.n_state; ++i) {
        out.positions[i] = i;
    }

    int question_id = 0;
    for (const auto & [qid, question] : body.at("questions").items()) {
        if (!question.is_object() || !question.contains("type") || !question.at("type").is_string()) {
            throw std::runtime_error("each question needs a type");
        }
        const std::string type = question.at("type").get<std::string>();
        systemone_question meta;
        meta.id = qid;
        meta.type = type;

        std::vector<std::string> options;
        if (type == "noul") {
            const json * criteria = question.contains("criteria") ? &question.at("criteria") : nullptr;
            const json * no = nullptr;
            const json * yes = nullptr;
            if (criteria && criteria->is_object()) {
                if (criteria->contains("false")) {
                    no = &criteria->at("false");
                }
                if (criteria->contains("true")) {
                    yes = &criteria->at("true");
                }
            }
            options.push_back(option_text("no", no));
            options.push_back(option_text("yes", yes));
            meta.keys = {"false", "true"};
        } else if (type == "choice") {
            if (!question.contains("criteria") || !question.at("criteria").is_object() || question.at("criteria").size() == 0) {
                throw std::runtime_error("choice needs criteria");
            }
            for (const auto & [name, desc] : question.at("criteria").items()) {
                options.push_back(option_text(name, &desc));
                meta.keys.push_back(name);
            }
        } else if (type == "score") {
            if (!question.contains("criteria") || !question.at("criteria").is_array() || question.at("criteria").size() == 0) {
                throw std::runtime_error("score needs criteria");
            }
            for (size_t i = 0; i < question.at("criteria").size(); ++i) {
                const std::string text = render_value(question.at("criteria").at(i), 0);
                options.push_back(text);
                meta.keys.push_back(std::to_string(i));
                meta.legend.push_back(text);
            }
        } else {
            throw std::runtime_error("question type must be choice, noul, or score");
        }
        if (options.empty() || (int) options.size() > SYSTEMONE_MAX_OPTIONS) {
            throw std::runtime_error("a question needs 1 to 255 options");
        }

        json instructions = nullptr;
        if (question.contains("instructions")) {
            instructions = question.at("instructions");
        }
        std::vector<llama_token> branch;
        branch.push_back(delim[1]);
        const std::vector<llama_token> instr = user_tokens(vocab, render_value(instructions, 0));
        branch.insert(branch.end(), instr.begin(), instr.end());
        std::vector<int> ends;
        for (const std::string & option : options) {
            branch.push_back(delim[2]);
            const std::vector<llama_token> words = user_tokens(vocab, option);
            branch.insert(branch.end(), words.begin(), words.end());
            branch.push_back(delim[3]);
            ends.push_back((int) branch.size() - 1);
        }
        branch.push_back(delim[4]);
        if ((int) branch.size() > SYSTEMONE_MAX_BRANCH - (int) out.n_state) {
            throw std::runtime_error("question branch exceeds the 8192-token row limit");
        }

        ++question_id;
        const int base = (int) out.tokens.size();
        const int32_t p0 = out.n_state;
        out.tokens.insert(out.tokens.end(), branch.begin(), branch.end());
        out.segments.insert(out.segments.end(), branch.size(), question_id);
        for (int32_t i = 0; i < (int32_t) branch.size(); ++i) {
            out.positions.push_back(p0 + i);
        }
        systemone_group group;
        group.decide = base + (int) branch.size() - 1;
        for (int end : ends) {
            group.options.push_back(base + end);
        }
        out.groups.push_back(std::move(group));
        out.meta.push_back(std::move(meta));
    }
    if (out.positions.size() != out.tokens.size() || out.segments.size() != out.tokens.size()) {
        throw std::runtime_error("encoder layout mismatch");
    }
    return out;
}

json systemone_format_answers(const std::vector<std::vector<float>> & logits, const std::vector<systemone_question> & meta) {
    if (logits.size() != meta.size()) {
        throw std::runtime_error("pointer readout did not return one group per question");
    }
    json answers = json::object();
    for (size_t q = 0; q < meta.size(); ++q) {
        const std::vector<double> probs = softmax(logits[q]);
        if (probs.size() != meta[q].keys.size()) {
            throw std::runtime_error("pointer readout width does not match the question");
        }
        size_t best = 0;
        for (size_t i = 1; i < probs.size(); ++i) {
            if (probs[i] > probs[best]) {
                best = i;
            }
        }
        json item = json::object();
        item["type"] = meta[q].type;
        if (meta[q].type == "noul") {
            item["noul"] = round4(probs[1]);
        } else if (meta[q].type == "choice") {
            json dist = json::object();
            for (size_t i = 0; i < probs.size(); ++i) {
                dist[meta[q].keys[i]] = round4(probs[i]);
            }
            const double uniform = 1.0 / (double) probs.size();
            const double confidence = probs.size() == 1 ? 1.0 : (probs[best] - uniform) / (1.0 - uniform);
            item["choice"] = meta[q].keys[best];
            item["confidence"] = round4(confidence);
            item["probabilities"] = std::move(dist);
        } else {
            double score = 0.0;
            double spread = 0.0;
            json dist = json::object();
            json legend = json::object();
            for (size_t i = 0; i < probs.size(); ++i) {
                score += (double) i * probs[i];
                spread += probs[i] * std::fabs((double) i - (double) best);
                dist[meta[q].keys[i]] = round4(probs[i]);
                if (i < meta[q].legend.size()) {
                    legend[meta[q].keys[i]] = meta[q].legend[i];
                }
            }
            const double confidence = probs.size() == 1 ? 1.0 : 1.0 - spread / (double) (probs.size() - 1);
            item["score"] = round4(score);
            item["legend"] = std::move(legend);
            item["probabilities"] = std::move(dist);
            item["confidence"] = round4(confidence);
        }
        answers[meta[q].id] = std::move(item);
    }
    return answers;
}

static std::string py_float(double value) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.4f", value);
    std::string text(buf);
    const auto dot = text.find('.');
    if (dot == std::string::npos) {
        return text + ".0";
    }
    while (text.size() > dot + 2 && text.back() == '0') {
        text.pop_back();
    }
    return text;
}

static void append_py_string(std::string & out, const std::string & text) {
    out.push_back('"');
    for (size_t i = 0; i < text.size();) {
        const unsigned char c = (unsigned char) text[i];
        if (c == '"') { out += "\\\""; ++i; continue; }
        if (c == '\\') { out += "\\\\"; ++i; continue; }
        if (c == '\b') { out += "\\b"; ++i; continue; }
        if (c == '\f') { out += "\\f"; ++i; continue; }
        if (c == '\n') { out += "\\n"; ++i; continue; }
        if (c == '\r') { out += "\\r"; ++i; continue; }
        if (c == '\t') { out += "\\t"; ++i; continue; }
        uint32_t cp = c;
        size_t n = 1;
        bool ok = true;
        if (c >= 0x80) {
            if ((c & 0xE0) == 0xC0 && i + 1 < text.size()) {
                cp = ((uint32_t) (c & 0x1F) << 6) | ((unsigned char) text[i + 1] & 0x3F);
                n = 2;
                ok = (c & 0xFE) != 0xC0;
            } else if ((c & 0xF0) == 0xE0 && i + 2 < text.size()) {
                cp = ((uint32_t) (c & 0x0F) << 12) | (((unsigned char) text[i + 1] & 0x3F) << 6) | ((unsigned char) text[i + 2] & 0x3F);
                n = 3;
                ok = cp >= 0x800;
            } else if ((c & 0xF8) == 0xF0 && i + 3 < text.size()) {
                cp = ((uint32_t) (c & 0x07) << 18) | (((unsigned char) text[i + 1] & 0x3F) << 12) |
                     (((unsigned char) text[i + 2] & 0x3F) << 6) | ((unsigned char) text[i + 3] & 0x3F);
                n = 4;
                ok = cp >= 0x10000 && cp <= 0x10FFFF;
            } else {
                ok = false;
            }
        }
        if (!ok) {
            out += "\\ufffd";
            ++i;
            continue;
        }
        if (cp < 0x20 || cp >= 0x80) {
            char esc[16];
            if (cp >= 0x10000) {
                const uint32_t u = cp - 0x10000;
                std::snprintf(esc, sizeof(esc), "\\u%04x\\u%04x", 0xD800 + (u >> 10), 0xDC00 + (u & 0x3FF));
            } else {
                std::snprintf(esc, sizeof(esc), "\\u%04x", cp);
            }
            out += esc;
        } else {
            out.push_back((char) cp);
        }
        i += n;
    }
    out.push_back('"');
}

static void append_py_json(std::string & out, const json & value) {
    if (value.is_null()) {
        out += "null";
    } else if (value.is_boolean()) {
        out += value.get<bool>() ? "true" : "false";
    } else if (value.is_string()) {
        append_py_string(out, value.get<std::string>());
    } else if (value.is_number()) {
        out += py_float(value.get<double>());
    } else if (value.is_array()) {
        out.push_back('[');
        for (size_t i = 0; i < value.size(); ++i) {
            if (i) {
                out += ", ";
            }
            append_py_json(out, value.at(i));
        }
        out.push_back(']');
    } else if (value.is_object()) {
        out.push_back('{');
        bool first = true;
        for (const auto & [key, child] : value.items()) {
            if (!first) {
                out += ", ";
            }
            first = false;
            append_py_string(out, key);
            out += ": ";
            append_py_json(out, child);
        }
        out.push_back('}');
    } else {
        throw std::runtime_error("answer could not be serialized");
    }
}

int systemone_output_tokens(const llama_vocab * vocab, const json & answers) {
    std::string text;
    append_py_json(text, answers);
    return (int) common_tokenize(vocab, text, false, false).size();
}
