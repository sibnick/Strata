#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/session.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/prefill/prefill.hpp"

#include <cstdio>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr uint64_t MiB = 1ull << 20;
constexpr uint64_t GiB = 1ull << 30;

uint64_t allocation_bytes(uint64_t bytes) {
    return bytes < MiB ? bytes : (bytes + 2 * MiB - 1) / (2 * MiB) * (2 * MiB);
}

uint64_t tensor_bytes(const strata::TensorInfo& tensor) {
    int elements = 0, bytes = 0;
    if (tensor.shape.empty() || !strata::block_geometry(tensor.type, elements, bytes) ||
        tensor.shape[0] % (uint64_t) elements != 0)
        throw std::runtime_error("unsupported tensor geometry: " + tensor.name);
    uint64_t count = 1;
    for (uint64_t dimension : tensor.shape) {
        if (!dimension || count > std::numeric_limits<uint64_t>::max() / dimension)
            throw std::runtime_error("invalid tensor extent: " + tensor.name);
        count *= dimension;
    }
    count /= (uint64_t) elements;
    if (count > std::numeric_limits<uint64_t>::max() / (uint64_t) bytes)
        throw std::runtime_error("tensor byte count overflow: " + tensor.name);
    return count * (uint64_t) bytes;
}

int layer_of(const std::string& name) {
    if (name.rfind("blk.", 0) != 0) return -1;
    return std::stoi(name.substr(4));
}

bool expert_tensor(const std::string& name) {
    return name.ends_with(".ffn_gate_exps.weight") || name.ends_with(".ffn_up_exps.weight") ||
           name.ends_with(".ffn_down_exps.weight");
}

void emit(int boundary, int stage, const char* component, uint64_t bytes) {
    std::printf("%d,%d,%s,%llu\n", boundary, stage, component, (unsigned long long) bytes);
}

void self_test() {
    const strata::core::ModelGeometry geometry;
    strata::core::qsa_set_kv_int8(true);
    const uint64_t expected[3][2] = {
        {2961460736, 1038760448}, {2964729344, 1035491840}, {2967997952, 1032223232},
    };
    for (int boundary = 36; boundary <= 38; ++boundary) {
        for (int stage = 0; stage < 2; ++stage) {
            const int lo = stage == 0 ? 0 : boundary, hi = stage == 0 ? boundary : 48;
            if (strata::core::session_bytes(geometry, 262144, 10, lo, hi) != expected[boundary - 36][stage])
                throw std::runtime_error("INT8 session allocation contract changed");
        }
    }
    if (allocation_bytes(MiB - 1) != MiB - 1 || allocation_bytes(MiB) != 2 * MiB ||
        allocation_bytes(2 * MiB + 1) != 4 * MiB)
        throw std::runtime_error("allocation rounding failed");
    std::puts("split-memory sizing self-test passed (no model or CUDA allocations)");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        std::string pack, native, mtp, draft_vocab;
        std::vector<int64_t> stage_chunks;
        int64_t context = 262144, chunk = 256, additional_reserve_mib = 1024;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--self-test") {
                self_test();
                return 0;
            }
            if (option == "--help") {
                std::printf("usage: strata-split-memory --pack DIR --native SHARD1 --mtp DIR "
                            "[--context N] [--chunk N] [--prefill-stage-chunks N,N] "
                            "[--draft-vocab FILE] [--additional-reserve-mib N]\n"
                            "UD-Q4_K_XL, INT8 KV, K36/K37/K38, owned prefill buffers (96-blob ring),\n"
                            "MTP spec=2 with its default 32768-cell window; no batch, KV streaming or vision.\n"
                            "CSV byte estimates, not a speed prediction or a fit guarantee. Reads headers/index only;\n"
                            "WeightTable metadata loading initializes CUDA and uses bounded staging buffers.\n"
                            "No model weights, expert cache or full-context session are allocated.\n"
                            "--self-test checks session bytes and allocation rounding without CUDA allocations.\n");
                return 0;
            }
            if (i + 1 == argc) throw std::runtime_error("missing value for " + option);
            const std::string value = argv[++i];
            if (option == "--pack") pack = value;
            else if (option == "--native") native = value;
            else if (option == "--mtp") mtp = value;
            else if (option == "--draft-vocab") draft_vocab = value;
            else if (option == "--prefill-stage-chunks") {
                size_t start = 0;
                while (start < value.size()) {
                    const size_t end = value.find(',', start);
                    const std::string token = value.substr(
                        start, end == std::string::npos ? std::string::npos : end - start);
                    int64_t n = 0;
                    const auto parsed = std::from_chars(token.data(), token.data() + token.size(), n);
                    if (token.empty() || parsed.ec != std::errc() || parsed.ptr != token.data() + token.size() ||
                        n <= 0 || n > 262144)
                        throw std::runtime_error("--prefill-stage-chunks expects positive integer sizes");
                    stage_chunks.push_back(n);
                    if (end == std::string::npos) break;
                    start = end + 1;
                }
                if (value.empty() || value.back() == ',')
                    throw std::runtime_error("--prefill-stage-chunks has an empty size");
            }
            else if (option == "--additional-reserve-mib") {
                size_t end = 0;
                additional_reserve_mib = std::stoll(value, &end);
                if (end != value.size() || additional_reserve_mib < 0)
                    throw std::runtime_error("--additional-reserve-mib must be a non-negative integer");
            }
            else if (option == "--context" || option == "--chunk") {
                size_t end = 0;
                const int64_t n = std::stoll(value, &end);
                if (end != value.size() || n <= 0 || n > 262144)
                    throw std::runtime_error(option + " must be an integer from 1 to 262144");
                (option == "--context" ? context : chunk) = n;
            } else throw std::runtime_error("unknown option: " + option);
        }
        if (pack.empty() || native.empty() || mtp.empty() || chunk > context ||
            (!stage_chunks.empty() &&
             (stage_chunks.size() != 2 || stage_chunks.front() != chunk ||
              stage_chunks[0] > context || stage_chunks[1] > context)))
            throw std::runtime_error("--pack, --native, --mtp and chunk <= context are required");

        const auto shards = strata::gguf_split_paths(native);
        strata::core::ModelGeometry geometry;
        std::string error;
        std::vector<uint64_t> experts(48, 0);
        uint64_t payload = 0, ple = 0, embedding = 0, head = 0;
        std::set<std::string> tensors;
        for (size_t i = 0; i < shards.size(); ++i) {
            const strata::GgufFile file(shards[i]);
            if (i == 0) {
                error = strata::check_engine_architecture(file);
                if (!error.empty()) throw std::runtime_error(error);
                error = strata::read_geometry(file, geometry);
                if (!error.empty()) throw std::runtime_error(error);
            }
            for (const auto& tensor : file.tensors()) {
                if (!tensors.insert(tensor.name).second)
                    throw std::runtime_error("duplicate tensor: " + tensor.name);
                const uint64_t bytes = tensor_bytes(tensor);
                const uint64_t size = file.file_size() - file.data_start();
                if (tensor.offset > size || bytes > size - tensor.offset)
                    throw std::runtime_error("truncated tensor: " + tensor.name);
                payload += bytes;
                if (expert_tensor(tensor.name)) {
                    const int layer = layer_of(tensor.name);
                    if (layer < 0 || layer >= 48) throw std::runtime_error("invalid expert layer");
                    experts[(size_t) layer] += bytes;
                }
                if (tensor.name == "per_layer_token_embd.weight") ple = bytes;
                if (tensor.name == "token_embd.weight") embedding = bytes;
                if (tensor.name == "output.weight") head = bytes;
            }
        }
        if (geometry.n_layers != 48 || geometry.n_expert != 512 || !ple || !embedding || !head)
            throw std::runtime_error("expected the 48-layer, 512-expert Flash-Next model with PLE, embedding and head");
        if (!strata::kernels::cpu::expert_layout_load(pack, geometry.n_layers, geometry.n_expert, error))
            throw std::runtime_error(error);
        const auto& layout = strata::kernels::cpu::expert_layout();
        for (size_t layer = 0; layer < experts.size(); ++layer)
            if (!experts[layer] || experts[layer] != layout.blob_bytes((int64_t) layer) * geometry.n_expert)
                throw std::runtime_error("pack/GGUF expert sizes differ at layer " + std::to_string(layer));

        std::set<std::string> names;
        std::ifstream index(pack + "/index.txt");
        if (!index) throw std::runtime_error("cannot open pack index");
        std::string line;
        while (std::getline(index, line)) {
            if (line.empty() || line[0] == '#') continue;
            names.insert(line.substr(0, line.find_first_of(" \t")));
        }
        strata::core::WeightTable table;
        // Skip every payload: reuse the loader's index validation and canonical metadata.
        if (!table.load(pack, nullptr, 0, error, &names)) throw std::runtime_error(error);
        std::set<std::string> native_names;
        if (!strata::core::NativeDense::served_names(shards, true, native_names, error) ||
            !strata::core::NativeDense::keep_unquantized_ple_key(pack, native_names, error))
            throw std::runtime_error(error);
        native_names.insert("token_embd.weight");
        native_names.insert("output.weight");
        strata::core::qsa_set_kv_int8(true);
        strata::prefill::Prefill::set_ring_override(96);
        const uint64_t mtp_dense = std::filesystem::file_size(mtp + "/dense.bin");
        const uint64_t mtp_experts = std::filesystem::file_size(mtp + "/experts.bin");
        const uint64_t mtp_files = mtp_dense + mtp_experts;
        const std::string vocab_path = draft_vocab.empty() ? mtp + "/draft_vocab.bin" : draft_vocab;
        const uint64_t vocab_bytes = std::filesystem::file_size(vocab_path);
        if (!vocab_bytes || vocab_bytes % 4 || vocab_bytes / 4 > (uint64_t) geometry.n_vocab)
            throw std::runtime_error("invalid MTP draft vocabulary size");
        const uint64_t draft_head = allocation_bytes(vocab_bytes / 4 * (head / geometry.n_vocab));
        // These are explicit allowances, not exact workspace sizes. The MTP allowance includes its own KV.
        const uint64_t verify_reserve = 96 * MiB;
        const uint64_t mtp_runtime_reserve = 384 * MiB;
        const uint64_t projection_padding_reserve = 64 * MiB;
        const uint64_t cuda_reserve = (uint64_t) additional_reserve_mib * MiB;
        std::fprintf(stderr, "INT8 KV; context=%lld chunk=%lld; owned prefill, not borrowed expert slots.\n"
                             "Verify/MTP-runtime/additional CUDA allowances are reserves, not measured allocations.\n"
                             "Sizing only: this command neither loads the model nor enforces a host-RAM floor.\n",
                     (long long) context, (long long) chunk);
        if (!stage_chunks.empty())
            std::fprintf(stderr, "Per-stage prefill chunks: stage0=%lld stage1=%lld; additional CUDA reserve=%lld MiB.\n",
                         (long long) stage_chunks[0], (long long) stage_chunks[1],
                         (long long) additional_reserve_mib);
        std::printf("split,stage,component,bytes\n");
        emit(0, -1, "gguf_payload", payload);
        emit(0, -1, "ple_direct_file_not_resident_ram", ple);
        emit(0, -1, "mapped_token_embedding", embedding);
        for (int layer = 0; layer < 48; ++layer)
            emit(0, -1, ("layer_" + std::to_string(layer) + "_expert_payload").c_str(), experts[(size_t) layer]);
        for (const int boundary : {36, 37, 38}) {
            for (int stage = 0; stage < 2; ++stage) {
                const int lo = stage == 0 ? 0 : boundary, hi = stage == 0 ? boundary : 48;
                uint64_t expert_bytes = 0;
                for (int layer = lo; layer < hi; ++layer) expert_bytes += experts[(size_t) layer];
                std::set<std::string> skip = native_names;
                for (const auto& [name, ref] : table.all()) {
                    (void) ref;
                    const int layer = layer_of(name);
                    if (layer < 0 || name.find("ple") != std::string::npos ||
                        (stage == 0 && name.ends_with(".ffn_gate_inp.weight"))) continue;
                    if (layer < lo || layer >= hi) skip.insert(name);
                }
                uint64_t canonical = 0, projections = 0;
                if (!strata::core::WeightTable::pool_bytes(pack, canonical, error, &skip) ||
                    !strata::core::NativeDense::weight_bytes_for(shards, table, true, lo, hi, projections, error))
                    throw std::runtime_error(error);
                std::vector<strata::core::QsaState> qsa((size_t) geometry.n_qsa_layers());
                strata::core::SessionState session;
                session.qsa_ord0 = lo / geometry.qsa_interval;
                session.qsa_states = qsa.data();
                for (auto& state : qsa) state.max_cells = context;
                const uint64_t state = strata::core::session_bytes(geometry, context, geometry.n_expert_used, lo, hi);
                const int64_t stage_chunk = stage_chunks.empty() ? chunk : stage_chunks[(size_t) stage];
                const uint64_t prefill =
                    strata::prefill::Prefill::bytes_needed_owned(geometry, session, stage_chunk, true);
                const uint64_t output = stage == 1 ? allocation_bytes(head) : 0;
                const uint64_t draft = stage == 1 ? allocation_bytes(mtp_dense) + allocation_bytes(mtp_experts) + draft_head +
                                                     allocation_bytes(vocab_bytes) + mtp_runtime_reserve : 0;
                emit(boundary, stage, "expert_payload", expert_bytes);
                emit(boundary, stage, "canonical_arena", canonical);
                emit(boundary, stage, "native_projection_allocation_estimate", projections);
                emit(boundary, stage, "native_projection_padding_reserve", projection_padding_reserve);
                emit(boundary, stage, "session_arena", state);
                emit(boundary, stage, "owned_prefill_allocation_estimate", prefill);
                emit(boundary, stage, "output_head_allocation_estimate", output);
                emit(boundary, stage, "mtp_files_payload", stage == 1 ? mtp_files : 0);
                emit(boundary, stage, "mtp_total_with_runtime_reserve", draft);
                emit(boundary, stage, "verify_reserve", verify_reserve);
                emit(boundary, stage, "additional_cuda_and_safety_reserve", cuda_reserve);
                emit(boundary, stage, "required_free_before_model_load",
                     expert_bytes + allocation_bytes(canonical) + projections + projection_padding_reserve + allocation_bytes(state) +
                     prefill + output + draft + verify_reserve + cuda_reserve);
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "strata-split-memory: %s\n", error.what());
        return 1;
    }
}
