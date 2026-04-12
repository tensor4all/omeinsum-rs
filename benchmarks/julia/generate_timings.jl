#!/usr/bin/env julia

using BenchmarkTools
using JSON
using OMEinsum

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DATA_DIR = joinpath(ROOT, "benchmarks", "data")
const BINARY_OUTPUT = joinpath(DATA_DIR, "binary_timings.json")
const NETWORK_OUTPUT = joinpath(DATA_DIR, "network_timings.json")

struct BinaryScenario
    name::String
    rank_a::Int
    rank_b::Int
    num_contracted::Int
    num_batch::Int
end

const BINARY_SCENARIOS = [
    BinaryScenario("matmul_10x10", 10, 10, 5, 0),
    BinaryScenario("batched_matmul_8x8_batch_4", 8, 8, 4, 4),
    BinaryScenario("high_d_12x12_contract_6", 12, 12, 6, 0),
    BinaryScenario("high_d_15x15_contract_7", 15, 15, 7, 0),
    BinaryScenario("high_d_18x18_contract_8", 18, 18, 8, 0),
    BinaryScenario("high_d_20x20_contract_9", 20, 20, 9, 0),
    BinaryScenario("high_d_12x12_contract_4_batch_4", 12, 12, 4, 4),
    BinaryScenario("high_d_15x15_contract_5_batch_5", 15, 15, 5, 5),
]

const NETWORK_SCENARIOS = [
    ("small", joinpath(ROOT, "benches", "network_small.json"), 10, 1),
    ("medium", joinpath(ROOT, "benches", "network_medium.json"), 10, 1),
    ("large", joinpath(ROOT, "benches", "network_large.json"), 10, 1),
    ("3reg_150", joinpath(ROOT, "benches", "network_3reg_150.json"), 1, 1),
]

function binary_indices(s::BinaryScenario)
    num_left_a = s.rank_a - s.num_contracted - s.num_batch
    num_right_b = s.rank_b - s.num_contracted - s.num_batch

    next_idx = 1
    left_indices = collect(next_idx:(next_idx + num_left_a - 1))
    next_idx += num_left_a

    contracted_indices = collect(next_idx:(next_idx + s.num_contracted - 1))
    next_idx += s.num_contracted

    right_indices = collect(next_idx:(next_idx + num_right_b - 1))
    next_idx += num_right_b

    batch_indices = collect(next_idx:(next_idx + s.num_batch - 1))

    ixs_a = vcat(left_indices, contracted_indices, batch_indices)
    ixs_b = vcat(contracted_indices, right_indices, batch_indices)
    ixs_c = vcat(left_indices, right_indices, batch_indices)

    return ixs_a, ixs_b, ixs_c
end

function prepare_binary_case(s::BinaryScenario)
    ixs_a, ixs_b, ixs_c = binary_indices(s)
    shape_a = ntuple(_ -> 2, s.rank_a)
    shape_b = ntuple(_ -> 2, s.rank_b)

    numel_a = 2^s.rank_a
    numel_b = 2^s.rank_b

    data_a = reshape(Float32.(collect(0:(numel_a - 1))) .* 0.001f0, shape_a)
    data_b = reshape(Float32.(collect(0:(numel_b - 1))) .* 0.001f0, shape_b)

    code = OMEinsum.EinCode((Tuple(ixs_a), Tuple(ixs_b)), Tuple(ixs_c))
    size_dict = Dict(i => 2 for i in unique(vcat(ixs_a, ixs_b, ixs_c)))
    return code, data_a, data_b, size_dict
end

function json_to_nested(node)
    if node["isleaf"]
        return OMEinsum.DynamicNestedEinsum{Int}(node["tensorindex"] + 1)
    end

    eins = node["eins"]
    args = [json_to_nested(child) for child in node["args"]]
    ixs = [Vector{Int}(ix) for ix in eins["ixs"]]
    iy = Vector{Int}(eins["iy"])
    code = OMEinsum.DynamicEinCode(ixs, iy)
    return OMEinsum.DynamicNestedEinsum(args, code)
end

function prepare_network_case(path::String)
    network = JSON.parsefile(path)
    bond_dim = network["bond_dim"]
    edges = network["edges"]

    tensors = tuple([fill(Float32(0.5^0.4), bond_dim, bond_dim) for _ in edges]...)
    size_dict = Dict{Int,Int}()
    for edge in edges
        size_dict[edge[1]] = bond_dim
        size_dict[edge[2]] = bond_dim
    end

    code = json_to_nested(network["tree"])
    return code, tensors, size_dict
end

function binary_timings()
    timings = Dict{String,Int}()
    for scenario in BINARY_SCENARIOS
        println("Binary: ", scenario.name)
        code, a, b, size_dict = prepare_binary_case(scenario)
        bench = @benchmark OMEinsum.einsum($code, ($a, $b), $size_dict) samples=100 evals=1
        timings[scenario.name] = minimum(bench).time
    end
    timings
end

function network_timings()
    timings = Dict{String,Int}()
    for (name, path, samples, evals) in NETWORK_SCENARIOS
        println("Network: ", name)
        code, tensors, size_dict = prepare_network_case(path)
        bench = @benchmark OMEinsum.einsum($code, $tensors, $size_dict) samples=samples evals=evals
        timings[name] = minimum(bench).time
    end
    timings
end

function write_compact_json(path::String, data)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON.print(io, data)
    end
end

write_compact_json(BINARY_OUTPUT, binary_timings())
write_compact_json(NETWORK_OUTPUT, network_timings())
