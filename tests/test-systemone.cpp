// Test protocol helpers without loading model weights or exposing them as server APIs.
#include "../tools/server/systemone.cpp"
#include "testing.h"

#include <cstring>
#include <iostream>
#include <limits>
#include <utility>

static void assert_invalid(testing & t, const std::string & request, const std::string & message) {
    try {
        systemone_encode(nullptr, json::parse(request));
        t.assert_true("invalid request was rejected before tokenization", false);
    } catch (const std::runtime_error & e) {
        t.assert_equal("validation error", message, std::string(e.what()));
    }
}

int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--render-double-bits") {
        std::string line;
        while (std::getline(std::cin, line)) {
            const uint64_t bits = std::stoull(line, nullptr, 16);
            double value;
            static_assert(sizeof(value) == sizeof(bits), "expected IEEE binary64");
            std::memcpy(&value, &bits, sizeof(value));
            std::cout << render_float(value) << '\n';
        }
        return 0;
    }

    testing t;
    t.test("Python-compatible numeric rendering", [](testing & t) {
        const std::vector<std::pair<std::string, std::string>> cases = {
            {"1", "1"}, {"1.0", "1.0"}, {"0.0", "0.0"}, {"-0.0", "-0.0"},
            {"0.1", "0.1"}, {"1.23456789", "1.23456789"},
            {"1e-4", "0.0001"}, {"1e-5", "1e-05"}, {"1e15", "1000000000000000.0"},
            {"1e16", "1e+16"}, {"-1.2345678901234567e100", "-1.2345678901234567e+100"},
            {"5e-324", "5e-324"}, {"1.7976931348623157e308", "1.7976931348623157e+308"},
            {"-9223372036854775808", "-9223372036854775808"},
            {"9223372036854775808", "9223372036854775808"},
            {"18446744073709551615", "18446744073709551615"},
        };
        for (const auto & [input, expected] : cases) {
            t.assert_equal(input, expected, render_value(json::parse(input), 0));
        }
        try {
            render_float(std::numeric_limits<double>::infinity());
            t.assert_true("non-finite number was rejected", false);
        } catch (const std::runtime_error & e) {
            t.assert_equal("non-finite number", std::string("numbers must be finite"), std::string(e.what()));
        }
    });

    t.test("rendering preserves object order and empty descriptions", [](testing & t) {
        const auto state = json::parse(R"({"z":1.23456789,"a":1.0,"items":[true,null]})");
        t.assert_equal("nested state", std::string("z: 1.23456789\na: 1.0\nitems:\n  - True\n  - "), render_value(state, 0));
        for (const auto & text : {"{}", "[]"}) {
            const auto desc = json::parse(text);
            t.assert_equal(text, std::string("option: "), option_text("option", &desc));
        }
        for (const auto & text : {"null", "\"\""}) {
            const auto desc = json::parse(text);
            t.assert_equal(text, std::string("option"), option_text("option", &desc));
        }
        const auto desc = json::parse("false");
        t.assert_equal("false description", std::string("option: False"), option_text("option", &desc));
    });

    t.test("batch wrappers are always rejected", [](testing & t) {
        const std::string message = "requests batch wrapper is not supported; send one state and questions object";
        assert_invalid(t, R"({"requests":[]})", message);
        assert_invalid(t, R"({"state":"x","questions":{"q":{"type":"noul"}},"requests":[]})", message);
        assert_invalid(t, R"({"state":"x","questions":{"q":{"type":"noul"}},"requests":null})", message);
    });

    t.test("noul criteria must be an object or null", [](testing & t) {
        for (const auto & criteria : {"[]", "\"yes\"", "1", "false"}) {
            assert_invalid(t, std::string(R"({"state":"x","questions":{"q":{"type":"noul","criteria":)") + criteria + "}}}", "noul criteria must be an object or null");
        }
        for (const auto & criteria : {"null", "{}", R"({"true":"yes","false":"no"})"}) {
            validate_request(json::parse(std::string(R"({"state":"x","questions":{"q":{"type":"noul","criteria":)") + criteria + "}}}"));
            t.assert_true(std::string("accepted ") + criteria, true);
        }
    });

    t.test("score legend is an ordered array", [](testing & t) {
        systemone_question question;
        question.id = "rating";
        question.type = "score";
        question.keys = {"0", "1"};
        question.legend = {"low", "high"};
        const auto answers = systemone_format_answers({{0.0f, 0.0f}}, {question});
        const auto & answer = answers.at("rating");
        t.assert_true("legend is array", answer.at("legend").is_array());
        t.assert_equal("legend order", std::string(R"(["low","high"])"), answer.at("legend").dump());
        t.assert_equal("expected score", 0.5, answer.at("score").get<double>());
        t.assert_equal("probability keys", std::string(R"({"0":0.5,"1":0.5})"), answer.at("probabilities").dump());
    });
    return t.summary();
}
