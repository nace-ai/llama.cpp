#include "models.h"

#include "llama-impl.h"
#include "llama-batch.h"

#include "gguf.h"

#include <cinttypes>
#include <cmath>
#include <cstdint>

// Qwen3 blocks, no KV cache. Missing segments: every token in a sequence can see every other token.
// With segments: state (0) is bidirectional; branch k sees state and earlier tokens of branch k only.
struct llm_graph_input_edlm : public llm_graph_input_attn_no_cache {
    llm_graph_input_edlm(const llama_hparams & hparams, const llama_cparams & cparams) :
        llm_graph_input_attn_no_cache(hparams, cparams) {}

    void set_input(const llama_ubatch * ubatch) override;
};

static int32_t edlm_src(const llama_ubatch * ubatch, int32_t i) {
    if (ubatch->data && (size_t) i < ubatch->data->batch_idxs.size()) {
        return ubatch->data->batch_idxs[i];
    }
    return i;
}

void llm_graph_input_edlm::set_input(const llama_ubatch * ubatch) {
    const int64_t n_tokens = ubatch->n_tokens;
    const bool have_seg = cparams.edlm && cparams.edlm->seg.size() == (size_t) n_tokens;
    if (!have_seg) {
        if (cparams.edlm && !cparams.edlm->seg.empty()) {
            LLAMA_LOG_WARN("%s: edlm segments %zu != n_tokens %" PRId64 ", using a bidirectional mask\n",
                    __func__, cparams.edlm->seg.size(), n_tokens);
        }
        llm_graph_input_attn_no_cache::set_input(ubatch);
        return;
    }

    const auto fill = [&](auto * data, int64_t ne) {
        using T = std::remove_reference_t<decltype(*data)>;
        std::fill(data, data + ne, llama_cast<T>(-INFINITY));
        const auto & seg = cparams.edlm->seg;
        for (int64_t i1 = 0; i1 < n_tokens; ++i1) {
            const llama_seq_id s1 = ubatch->seq_id[i1][0];
            const int32_t src1 = edlm_src(ubatch, (int32_t) i1);
            if (src1 < 0 || (size_t) src1 >= seg.size()) {
                continue;
            }
            const int32_t g1 = seg[(size_t) src1];
            for (int64_t i0 = 0; i0 < n_tokens; ++i0) {
                if (ubatch->seq_id[i0][0] != s1) {
                    continue;
                }
                const int32_t src0 = edlm_src(ubatch, (int32_t) i0);
                if (src0 < 0 || (size_t) src0 >= seg.size()) {
                    continue;
                }
                const int32_t g0 = seg[(size_t) src0];
                const bool diag = src0 == src1;
                const bool state = g0 == 0 && g1 == 0;
                const bool same = g0 == 0 || g0 == g1;
                const bool causal = src0 <= src1;
                if (diag || (g0 >= 0 && g1 >= 0 && (state || (causal && same)))) {
                    data[i1 * n_tokens + i0] = llama_cast<T>(0.0f);
                }
            }
        }
    };

    GGML_ASSERT(self_kq_mask);
    GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask->buffer));
    if (self_kq_mask->type == GGML_TYPE_F16) {
        fill((ggml_fp16_t *) self_kq_mask->data, ggml_nelements(self_kq_mask));
    } else {
        fill((float *) self_kq_mask->data, ggml_nelements(self_kq_mask));
    }
}

void llama_model_edlm::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer()) {
        case 28: type = hparams.n_embd == 1024 ? LLM_TYPE_0_6B : LLM_TYPE_1_7B; break;
        case 36: type = hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_8B; break;
        case 40: type = LLM_TYPE_14B; break;
        case 64: type = LLM_TYPE_32B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
    hparams.causal_attn = false;
}

void llama_model_edlm::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    const int64_t tid = gguf_find_tensor(ml.metadata, "pointer.q.weight");
    if (tid >= 0) {
        const int64_t * ne = gguf_get_tensor_ne(ml.metadata, tid);
        const int64_t n_in = ne[0];
        const int64_t dp = ne[1];
        edlm_q    = create_tensor(tn(LLM_TENSOR_EDLM_POINTER_Q, "weight"), {n_in, dp}, 0);
        edlm_q_b  = create_tensor(tn(LLM_TENSOR_EDLM_POINTER_Q, "bias"),   {dp}, TENSOR_NOT_REQUIRED);
        edlm_k    = create_tensor(tn(LLM_TENSOR_EDLM_POINTER_K, "weight"), {n_in, dp}, 0);
        edlm_k_b  = create_tensor(tn(LLM_TENSOR_EDLM_POINTER_K, "bias"),   {dp}, TENSOR_NOT_REQUIRED);
        edlm_temp = create_tensor(tn(LLM_TENSOR_EDLM_POINTER_TEMP, "weight"), {1}, TENSOR_NOT_REQUIRED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_edlm::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_edlm::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();

    auto inp_owned = std::make_unique<llm_graph_input_edlm>(hparams, cparams);
    const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;
    inp_owned->self_kq_mask = ggml_new_tensor_4d(ctx0, type_mask, n_tokens, n_tokens, 1, 1);
    ggml_set_input(inp_owned->self_kq_mask);
    cb(inp_owned->self_kq_mask, "self_kq_mask", -1);
    inp_owned->self_kq_mask_cnv = inp_owned->self_kq_mask;
    auto * inp_attn = (llm_graph_input_edlm *) res->add_input(std::move(inp_owned));

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        {
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, model.layers[il].ffn_up_s,
                model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
                model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
