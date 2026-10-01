#include "edlm-systemone.h"

#include "common.h"

#include <cmath>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace {

constexpr int kMaxOptions = 255;
constexpr int kMaxState = 8192;
constexpr int kMaxBranch = 8192;
constexpr int kMaxPacked = 16384;

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

edlm_encoded edlm_encode_systemone(const llama_vocab * vocab, const json & body) {
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
    if ((int) state_tokens.size() + 1 > kMaxState) {
        throw std::runtime_error("state exceeds 8192 tokens");
    }

    edlm_encoded out;
    out.tokens.push_back(delim[0]);
    out.tokens.insert(out.tokens.end(), state_tokens.begin(), state_tokens.end());
    out.segments.assign(out.tokens.size(), 0);

    int question_id = 0;
    for (const auto & [qid, question] : body.at("questions").items()) {
        if (!question.is_object() || !question.contains("type") || !question.at("type").is_string()) {
            throw std::runtime_error("each question needs a type");
        }
        const std::string type = question.at("type").get<std::string>();
        edlm_question_meta meta;
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
        if (options.empty() || (int) options.size() > kMaxOptions) {
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
        if ((int) branch.size() > kMaxBranch - (int) out.segments.size()) {
            throw std::runtime_error("question branch exceeds the 8192-token row limit");
        }
        if ((int) out.tokens.size() + (int) branch.size() > kMaxPacked) {
            throw std::runtime_error("request exceeds 16384 tokens");
        }

        ++question_id;
        const int base = (int) out.tokens.size();
        out.tokens.insert(out.tokens.end(), branch.begin(), branch.end());
        out.segments.insert(out.segments.end(), branch.size(), question_id);
        edlm_encoded_group group;
        group.decide = base + (int) branch.size() - 1;
        for (int end : ends) {
            group.options.push_back(base + end);
        }
        out.groups.push_back(std::move(group));
        out.meta.push_back(std::move(meta));
    }
    return out;
}

json edlm_format_answers(const std::vector<std::vector<float>> & logits, const std::vector<edlm_question_meta> & meta) {
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
