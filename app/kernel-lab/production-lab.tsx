"use client";

import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { KERNEL_GROUPS, KERNEL_OPS, type KernelOp, TOTAL_TRITON_KERNELS } from "../kernel-ops";
import { KERNEL_SIGNATURES } from "../kernel-signatures";
import { KERNEL_CALL_BINDINGS } from "../kernel-call-bindings";
import { KERNEL_SOURCE_LOADERS, VISUAL_SOURCE, type KernelSourceSegment, type SourcePhase } from "../kernel-source-segments";

type ArgKind = "input" | "output" | "inout" | "index" | "stride" | "shape" | "block" | "flag" | "scalar" | "helper";
type Coord = { lane: number; local: number[]; global: number[]; flat: number; valid: boolean };

const STEPS = [
  ["GRID", "启动 Grid"], ["PROGRAM", "program 领任务"], ["OFFSET", "lane 算地址"],
  ["LOAD", "逐变量 load"], ["UB", "UB 工作集"], ["COMPUTE", "片上计算"],
  ["STORE", "逐变量 store"], ["TORCH", "Torch 对照"],
] as const;

const SOURCE_ROOT = `https://github.com/sgl-project/sgl-kernel-npu/blob/${VISUAL_SOURCE.commit}/python/sgl_kernel_npu/sgl_kernel_npu`;
const SGLANG_ROOT = "https://github.com/sgl-project/sglang/blob/main/python/sglang";
const PHASE_STEP: Record<SourcePhase, number> = { program:1, offset:2, load:3, compute:5, store:6 };
const PHASE_TEXT: Record<SourcePhase, string> = { program:"PROGRAM / TASK", offset:"OFFSET / MASK", load:"GM → UB", compute:"UB COMPUTE", store:"UB → GM" };

type ArgMeaning = { meaning: string; shape: string; caller: string };
type LabProfile = { labels: string[]; shape: number[]; block: number[]; persistent: boolean; persistentMode?: "strided" | "contiguous"; programCap?: number };
type UbItem = { name: string; source: string; formula: string; elements: number; bytesPerElement: number; bytes: number; phase: "load" | "compute" };

const EXACT_ARG_MEANINGS: Record<string, Partial<ArgMeaning>> = {
  hidden_states: { meaning:"MLP/GEMM 产生的 gate-up 融合 hidden states", shape:"[tokens, 2 × intermediate_size]", caller:"模型 MLP 激活输入" },
  hidden_states_ptr: { meaning:"当前 token 的模型 hidden states", shape:"[tokens, hidden_size]", caller:"MoE 路由前的 hidden_states" },
  hidden_state_ptr: { meaning:"当前 token / draft step 的 hidden state", shape:"[batch, hidden_size] 或 [batch, hidden_size, steps]", caller:"模型层传入的 hidden_states" },
  input1_ptr: { meaning:"已路由专家分支的 hidden states", shape:"[tokens, hidden_size]", caller:"routed_input" },
  input2_ptr: { meaning:"共享专家或残差分支的 hidden states", shape:"[tokens, hidden_size]", caller:"shared_input" },
  input_ptr: { meaning:"该算子的主输入张量；具体内容见当前算子标题与上方输入说明", shape:"由 wrapper 展平为 [rows, feature]，或保持源码声明的多维布局", caller:"Python wrapper 的 input / x / fused projection 输出" },
  x_ptr: { meaning:"算子的主输入 x；在激活/归一化中通常就是 hidden states", shape:"通常 [tokens, hidden_size]；门控激活时为 [tokens, 2 × intermediate_size]", caller:"wrapper 的 x 或 hidden_states" },
  q_ptr: { meaning:"Query 张量", shape:"[batch 或 tokens, q_heads, head_dim]（部分 kernel 展平末两维）", caller:"注意力层由 q_proj / 融合 QKV 投影得到的 q" },
  Q: { meaning:"decode 当前步的 Query", shape:"[batch, q_heads, qk_head_dim]", caller:"注意力后端传给 decode_gqa / decode_mla 的 q" },
  query: { meaning:"注意力 Query", shape:"[tokens, q_heads, head_dim]", caller:"模型 attention 的 query states" },
  query_ptr: { meaning:"注意力或索引器使用的 Query", shape:"[batch/tokens, heads, head_dim]", caller:"attention/indexer 后端的 query states" },
  k_ptr: { meaning:"Key 输出或工作张量", shape:"[batch 或 tokens, kv_heads, head_dim]", caller:"融合 QKV 输入拆出的 k；随后写入或读取 KV cache" },
  v_ptr: { meaning:"Value 输出或工作张量", shape:"[batch 或 tokens, kv_heads, value_dim]", caller:"融合 QKV 输入拆出的 v；随后写入或读取 KV cache" },
  K_Buffer: { meaning:"分页 Key cache", shape:"[num_pages, page_size, kv_heads, key_dim]", caller:"请求 KV cache pool 中的 k_buffer" },
  V_Buffer: { meaning:"分页 Value cache", shape:"[num_pages, page_size, kv_heads, value_dim]", caller:"请求 KV cache pool 中的 v_buffer" },
  K_NOPE_Buffer: { meaning:"MLA 中不含 RoPE 部分的分页 Key cache", shape:"[num_pages, page_size, kv_heads, kv_lora_rank]", caller:"MLA KV cache 的 compressed/nope 部分" },
  K_ROPE_Buffer: { meaning:"MLA 中携带位置编码的 Key cache", shape:"[num_pages, page_size, kv_heads, rope_dim]", caller:"MLA KV cache 的 RoPE 部分" },
  k_cache: { meaning:"分页 Key cache", shape:"[num_pages, page_size, kv_heads, head_dim]", caller:"请求 KV cache pool" },
  v_cache: { meaning:"分页 Value cache", shape:"[num_pages, page_size, kv_heads, head_dim]", caller:"请求 KV cache pool" },
  k_cache_ptr: { meaning:"分页 Key cache", shape:"[num_pages, page_size, kv_heads, head_dim]", caller:"请求 KV cache pool" },
  v_cache_ptr: { meaning:"分页 Value cache", shape:"[num_pages, page_size, kv_heads, head_dim]", caller:"请求 KV cache pool" },
  Att_Out: { meaning:"decode attention 的输出", shape:"[batch, q_heads, value_dim]", caller:"attention 后端预分配的 att_out" },
  attn_out: { meaning:"attention 输出 hidden states", shape:"[tokens, q_heads, value_dim]", caller:"注意力层输出缓冲区" },
  block_table: { meaning:"逻辑 KV page 到物理 cache page 的映射表", shape:"[batch, max_num_pages]", caller:"调度器为每个请求维护的 block table" },
  block_tables: { meaning:"逻辑 KV page 到物理 cache page 的映射表", shape:"[batch, max_num_pages]", caller:"调度器为每个请求维护的 block tables" },
  block_table_ptr: { meaning:"逻辑 KV page 到物理 cache page 的映射表", shape:"[batch, max_num_pages]", caller:"注意力后端生成的 block_table" },
  kv_seq_lens: { meaning:"每个请求当前可见的 KV token 数", shape:"[batch]", caller:"forward batch 的序列长度元数据" },
  seq_lens: { meaning:"每个请求的实际序列长度", shape:"[batch]", caller:"forward batch / scheduler 的 seq_lens" },
  seq_lens_ptr: { meaning:"每个请求的实际序列长度", shape:"[batch]", caller:"forward batch / scheduler 的 seq_lens" },
  req_to_token_ptr: { meaning:"request slot 与逻辑 token 位置到物理 token/cache slot 的映射", shape:"[request_pool_size, max_context_len]", caller:"req_to_token_pool.req_to_token" },
  req_pool_indices_ptr: { meaning:"batch 中每个请求对应的 request-pool slot", shape:"[batch]", caller:"forward_batch.req_pool_indices" },
  group_list_ptr: { meaning:"各 expert 的 token 分组边界或计数", shape:"[num_experts] 或 [num_experts + 1]", caller:"MoE dispatch 产生的 group_list" },
  weight_ptr: { meaning:"逐 hidden/channel 使用的模型权重", shape:"[hidden_size]；卷积权重另见当前 wrapper", caller:"模块参数 weight" },
  bias_ptr: { meaning:"逐 hidden/channel 广播的偏置", shape:"[hidden_size]", caller:"模块参数 bias" },
  residual_ptr: { meaning:"与主分支相加的残差 hidden states", shape:"[tokens, hidden_size]", caller:"Transformer block 的 residual" },
  output_ptr: { meaning:"当前算子的输出张量", shape:"通常与主输入同 shape；降维/拆分算子按上方输出说明", caller:"Python wrapper 预分配的 output" },
  out_ptr: { meaning:"当前算子的输出或量化输出", shape:"由 wrapper 根据算子语义分配；通常 [tokens, output_hidden]", caller:"Python wrapper 的 out" },
  scale_ptr: { meaning:"量化或归一化产生/使用的 scale", shape:"常见 [tokens]、[groups] 或 [hidden_size]", caller:"量化元数据或模型 scale 参数" },
  logits_ptr: { meaning:"采样前的词表 logits", shape:"[batch, vocab_size]", caller:"LM head 输出 logits" },
  argmax_ptr: { meaning:"每个请求最大 logit 对应的 token id", shape:"[batch]", caller:"采样器输出 token ids" },
  prob_ptr: { meaning:"被选中 argmax token 的 softmax 概率", shape:"[batch]", caller:"采样器输出概率" },
  conv_state_ptr: { meaning:"Mamba/线性注意力的卷积历史缓存", shape:"[num_cache_lines, channels, state_len]", caller:"mamba cache pool 的 conv_state" },
  conv_states_ptr: { meaning:"Mamba/线性注意力的卷积历史缓存", shape:"[num_cache_lines, channels, state_len]", caller:"mamba cache pool 的 conv states" },
  conv_state_indices_ptr: { meaning:"当前 batch 每个请求选择的卷积 cache line", shape:"[batch]", caller:"forward metadata 的 mamba_cache_indices" },
  pos_ptr: { meaning:"每个 token/row 的绝对 position id", shape:"[tokens]", caller:"模型 forward 的 positions" },
  sin_ptr: { meaning:"当前 position 对应的 RoPE sin", shape:"[tokens, rope_dim] 或可广播 cache", caller:"rotary embedding 生成的 position_sin" },
  cos_ptr: { meaning:"当前 position 对应的 RoPE cos", shape:"[tokens, rope_dim] 或可广播 cache", caller:"rotary embedding 生成的 position_cos" },
  cos_sin_cache_ptr: { meaning:"按 position 索引的半宽 RoPE cos/sin cache", shape:"[max_seq_len, rope_dim]", caller:"rotary embedding 的全局 position cache" },
  in_qkv_ptr: { meaning:"融合投影产生的 Q、可选 gate、K、V 连续输入", shape:"[tokens, q_size + gate_size + 2 × kv_size]", caller:"Qwen3-VL / 多模态 attention 的融合 QKV 投影输出 qkv" },
  cos_sin_ptr: { meaning:"temporal、height、width 三套 MRoPE cos/sin 表；每套最后一维前半是 cos、后半是 sin", shape:"[3, num_tokens, rope_dim]", caller:"多模态 rotary embedding 根据文本/高/宽 position ids 生成的 cos_sin" },
  out_q_ptr: { meaning:"完成分 head RMSNorm 与 MRoPE 后的 Query", shape:"[tokens, num_q_heads × head_size]", caller:"attention 计算使用的 q_output" },
  out_k_ptr: { meaning:"完成分 head RMSNorm 与 MRoPE 后的 Key", shape:"[tokens, num_kv_heads × head_size]", caller:"attention / KV cache 使用的 k_output" },
  out_v_ptr: { meaning:"从融合输入拆出的 Value；不做 RMSNorm 或 MRoPE", shape:"[tokens, num_kv_heads × head_size]", caller:"attention / KV cache 使用的 v_output" },
  out_gate_ptr: { meaning:"融合输入中可选的 Q gate 分支", shape:"[tokens, num_q_heads × head_size]；无 gate 时末维为 0", caller:"启用 gated attention 时返回的 gate_output" },
  predicts: { meaning:"最终被接受或重新采样的 token 写入位置", shape:"[num_prediction_slots]", caller:"speculative decoding 的输出 token buffer" },
  candidates: { meaning:"draft 模型提出的候选 token id；chain 按列前进，tree 按 child/sibling 索引遍历", shape:"[batch, num_draft_tokens]", caller:"speculative decoding draft candidates" },
  retrive_index: { meaning:"每个候选节点对应的输出/隐藏状态槽位索引", shape:"[batch, num_draft_tokens]", caller:"draft tree/chain 的 retrieve index（沿用上游拼写 retrive）" },
  retrive_next_token: { meaning:"tree 中当前节点第一个 child 的节点下标；-1 表示没有 child", shape:"[batch, num_draft_tokens]", caller:"tree speculative decoding 的 child 链" },
  retrive_next_sibling: { meaning:"tree 中当前节点下一个 sibling 的节点下标；-1 表示 sibling 链结束", shape:"[batch, num_draft_tokens]", caller:"tree speculative decoding 的 sibling 链" },
  uniform_samples: { meaning:"每个 draft 节点用于接受/拒绝判断的 [0,1) 随机数", shape:"[batch, num_draft_tokens]", caller:"采样器预生成的 acceptance uniforms" },
  uniform_samples_for_final_sampling: { meaning:"拒绝后从 residual 分布抽最终 token 的每请求随机数", shape:"[batch]", caller:"采样器预生成的 final-sampling uniforms" },
  target_probs: { meaning:"target 模型在每个 draft/tree 行上的完整词表概率", shape:"[batch, num_draft_tokens, vocab_size]", caller:"target logits softmax 后的概率" },
  draft_probs: { meaning:"chain 中是 draft 模型词表概率；target-only tree 中复用为 rejected probability scratch", shape:"[batch, draft_rows, vocab_size]", caller:"draft probability 或 zeros_like(target_probs) scratch" },
  rejected_probs: { meaning:"target-only tree 中记录被拒绝 sibling 的 target probability", shape:"[batch, num_draft_tokens, vocab_size]", caller:"wrapper 传入并先清零的 draft_probs scratch" },
  metadata: { meaning:"跨三次 kernel launch 传递的每请求小型状态：概率行、输出槽以及是否全部接受", shape:"chain=[batch,3]；target-only tree=[batch,2]", caller:"speculative sampling wrapper 内部分配的 int64 metadata" },
  block_sums: { meaning:"每个请求、每个 2048-token 词表块的 residual probability 总和", shape:"[batch, ceil(vocab_size/2048)]", caller:"wrapper 内部分配，供最终两级 CDF 采样" },
  accept_index: { meaning:"按接受顺序记录被保留的 draft/retrieve 槽位", shape:"chain=[batch,num_draft_tokens]；tree=[batch,max_tree_depth]", caller:"speculative decoding 输出的 accepted indices" },
  accept_token_num: { meaning:"每个请求实际接受的 draft token 数量", shape:"[batch]", caller:"speculative decoding 输出的 accepted count" },
};

function outputArg(name: string) {
  const n = name.toLowerCase();
  if (/^(out|o|y|output|att_out|gated_output|zero_result|accept_index|accept_token_num|predicts|argmax|prob|v_new|ht|ad|ai)$/.test(n)) return true;
  if (/(^|_)(out|output|dst|result|mean|rstd|variance|candidate_scores|candidate_indices|topk_indices|mixed_qkvz|mixed_ba)(_|$)/.test(n)) return true;
  return /^(q_ptr|k_ptr|v_ptr|gate_ptr|scale_ptr)$/.test(n) && false;
}

function pointerArg(name: string) {
  const n = name.toLowerCase();
  if (/(stride|block|size|num_|^num|dim|head|batch|total|scale$|eps|beta$|threshold|limit|width|len$|^t$|^h$|^k$|^v$|^b$)/.test(n) && !n.endsWith("_ptr")) return false;
  return n.endsWith("_ptr") || /(^|_)(cache|buffer|table|indices|index|lens|offsets)$/.test(n) || ["q","k","v","w","x","y","o","g","a","b","h","h0","ht","beta","query","sinks","predicts","candidates","target_predict","hidden_states","gated_output"].includes(n);
}

function classifyArg(name: string, loads: string[] = [], stores: string[] = []): ArgKind {
  const n = name.toLowerCase();
  const isPointer = pointerArg(name);
  if (isPointer && loads.includes(name) && stores.includes(name)) return "inout";
  if (isPointer && stores.includes(name)) return "output";
  if (isPointer && loads.includes(name) && /(index|indices|table|lens|length|offset|loc|pool|seqlen|cum_seq)/.test(n)) return "index";
  if (isPointer && loads.includes(name)) return "input";
  if (n === "x" && name.length === 1) return "input";
  if (outputArg(name)) return "output";
  if (pointerArg(name) && /(index|indices|table|lens|length|offset|loc|pool|seqlen|cum_seq)/.test(n)) return "index";
  if (pointerArg(name)) return "input";
  if (n.startsWith("stride") || n.includes("_stride")) return "stride";
  if (/^(block|blk|bt$|bk$|bv$|bd$|bhv$|mbs$|nbd$|bd$|topk$|rows_per_iter|dim_block)/.test(n) || n.includes("block_size") || n.endsWith("_block")) return "block";
  if (/^(has_|use_|is_|do_|need_|save_|store_|head_first|reverse|bias$|norms$|run_|guarded|fill_only|prefetch)/.test(n)) return "flag";
  if (name === "_builder") return "helper";
  if (/^(t|b|h|k|v|d|n|m|bs|hv|hg|nb|nt)$/.test(n) || /(size|dim|rows|cols|heads|tokens|batch|seq_len|num_|total|page_size|max_|steps|width|rank)/.test(n)) return "shape";
  return "scalar";
}

const KIND_TEXT: Record<ArgKind, string> = {
  input: "输入张量指针", output: "输出 / 更新目标", inout: "原地读取并更新", index: "索引 / 长度张量", stride: "元素步长",
  shape: "形状 / 循环边界", block: "编译期 Block", flag: "编译期分支", scalar: "标量参数", helper: "编译器 helper",
};

function shapeFromName(name: string, kind: ArgKind, op: KernelOp): string {
  const n = name.toLowerCase();
  if (kind === "stride" || kind === "shape" || kind === "block" || kind === "flag" || kind === "scalar" || kind === "helper") return "标量";
  if (/qk_var/.test(n)) return "[tokens, 2]（Q/K 各一个方差）";
  if (/mean|rstd|variance/.test(n)) return "[rows] 或 [rows, 1]";
  if (/scale|offset/.test(n) && !/state|stride/.test(n)) return "[rows]、[groups] 或可广播 shape";
  if (/expert_(indices|scales)|topk_(idx|indices)|candidate_(indices|scores)/.test(n)) return "[tokens, top_k]";
  if (/accept_index/.test(n)) return "[batch, num_draft_tokens]";
  if (/accept_token_num/.test(n)) return "[batch]";
  if (/predict|candidate|retrive/.test(n)) return "[batch, num_draft_tokens] 或展平后的等价布局";
  if (/state|snapshot|cache/.test(n)) return op.group === "Mamba / Cache" ? "[cache_slots, layers/heads, hidden/channel, state/window]（以 wrapper stride 为准）" : "缓存布局；维度由当前 wrapper 的 stride 参数给出";
  if (/weight|bias/.test(n)) return "[hidden/head/channel dim]，按对应维广播";
  if (/^(q|k|v|o|g|w|h|h0|ht|a|b|beta)$/.test(n)) return "源码采用的多维工作张量；维度符号见本 kernel 的 shape 参数";
  return "见 Python wrapper 的实参 shape；kernel 可能把它展平后寻址";
}

function meaningFromName(name: string, kind: ArgKind, op: KernelOp, callerExpression?: string, wrapper?: string, wrapperShape?: string): ArgMeaning {
  const exact = EXACT_ARG_MEANINGS[name];
  const n = name.toLowerCase();
  let meaning = exact?.meaning;
  if (!meaning) {
    if (kind === "stride") meaning = `${name.replace(/^stride_/, "").replace(/_stride$/, "")} 这一维跨一格对应的元素步长（不是字节数）`;
    else if (kind === "block") meaning = `当前 program 在 ${name.replace(/^block(_size)?_?/i, "")} 维上的编译期 tile 大小`;
    else if (kind === "flag") meaning = `控制 ${name.replace(/_/g, " ").toLowerCase()} 代码路径是否启用的编译期条件`;
    else if (kind === "shape") meaning = `${name.replace(/_/g, " ").toLowerCase()} 的真实尺寸、数量或循环边界`;
    else if (kind === "index") meaning = /len|seqlen/.test(n) ? "每个 request / sequence 的有效长度" : /table/.test(n) ? "逻辑位置到物理 cache/page 的映射表" : "选择 request、token、page、expert 或 cache slot 的整数索引";
    else if (/weight/.test(n)) meaning = "该算子使用的模型权重";
    else if (/bias/.test(n)) meaning = "该算子使用的广播偏置";
    else if (/scale|alpha|beta/.test(n)) meaning = "计算、attention 或量化使用的缩放系数";
    else if (/eps|epsilon/.test(n)) meaning = "归一化数值稳定项，加入 variance / norm 后避免除零";
    else if (kind === "output") meaning = "Python wrapper 为当前算子分配的结果张量";
    else if (kind === "inout") meaning = "既提供旧值又接收更新值的状态或缓存张量";
    else if (kind === "input") meaning = `当前 ${op.title} 的输入/中间张量；名字 ${name} 与 wrapper 调用位置一一对应`;
    else meaning = `参与 ${op.title} 地址计算、边界判断或数学公式的标量`;
  }
  return {
    meaning,
    shape: wrapperShape ?? exact?.shape ?? shapeFromName(name, kind, op),
    caller: callerExpression ? `${wrapper ?? "Python wrapper"} 中传入 ${callerExpression}` : exact?.caller ?? (kind === "block" || kind === "flag" ? "Python wrapper 启动 kernel 时传入的 constexpr" : kind === "stride" || kind === "shape" ? "由实参 tensor.shape / stride 推导后传入" : kind === "scalar" ? "模型配置、算子参数或 wrapper 计算值" : `sgl-kernel-npu 的 ${op.module} Python wrapper；上层模型语义见本算子输入说明`),
  };
}

function profileFor(op: KernelOp, kernel = op.kernels[0]): LabProfile {
  const persistent = /persistent/i.test(op.grid);
  if (op.id === "muladd") return { labels:["batch_rows","unused grid axis","hidden_size"], shape:[48,1,4096], block:[1,1,4096], persistent:true, programCap:40 };
  if (op.id === "qkvmrope") return { labels:["token / core task","Q/K head","head lane"], shape:[12,4,128], block:[1,4,128], persistent:true, persistentMode:"contiguous", programCap:8 };
  if (op.id === "chain_sample" || op.id === "tree_target") {
    if (kernel.includes("block_sum")) return { labels:["request","vocab token","scalar"], shape:[4,4096,1], block:[1,2048,1], persistent:true, programCap:4 };
    if (kernel.includes("sample_kernel")) return { labels:["request","vocab block sums","selected block lane"], shape:[4,2,2048], block:[1,2,2048], persistent:false };
    return { labels:["request","draft / tree step","scalar"], shape:[4,6,1], block:[1,6,1], persistent:false };
  }
  if (op.group === "Attention" || op.group === "Indexer") return { labels:["batch / Q-block","head / chunk","D / page lane"], shape:[4,6,16], block:[1,2,8], persistent };
  if (op.group === "FLA / KDA") return { labels:["sequence / chunk","head","K/V lane"], shape:[8,4,16], block:[1,1,8], persistent };
  if (op.group === "Mamba / Cache") return { labels:["request / layer","channel block","window / tail"], shape:[8,8,8], block:[1,2,4], persistent };
  if (op.group === "Norm / RoPE") return { labels:["token / row","head / column group","hidden lane"], shape:[8,4,32], block:[1,1,8], persistent };
  if (op.group === "MoE / Sample") return { labels:["token / request","expert / head","hidden / top-k"], shape:[8,4,16], block:[1,1,8], persistent };
  return { labels:["row / token","expert / group","hidden lane"], shape:[8,2,32], block:[1,1,8], persistent };
}

function symbolValue(symbol: string, shape: number[], block: number[]) {
  const s = symbol.replace(/[()]/g, "").trim();
  if (/^\d+$/.test(s)) return Number(s);
  if (/^(BLOCK_M|BLOCK_L|ROWS_PER_ITER|BT|MINIBLOCK_SIZE)$/i.test(s)) return block[0];
  if (/^(BLOCK_N|BLOCK_H|BH)$/i.test(s)) return block[1];
  if (/^(BLOCK_SIZE|BLOCK_C|COL_BLOCK_SIZE|BLOCK_D|BLOCK_V|BLOCK_K|BK|BV|BD)$/i.test(s)) return block[2];
  if (/^(batch_size|batch_rows|M|L|T)$/i.test(s)) return shape[0];
  if (/^(num_heads|heads|H)$/i.test(s)) return shape[1];
  if (/^(hidden_size|output_dim|C|D|K|V|N)$/i.test(s)) return shape[2];
  const division = s.match(/^([A-Za-z_]\w*|\d+)\s*\/\/\s*([A-Za-z_]\w*|\d+)$/);
  if (division) return Math.max(1, Math.floor(symbolValue(division[1], shape, block) / symbolValue(division[2], shape, block)));
  const subtraction = s.match(/^([A-Za-z_]\w*|\d+)\s*-\s*([A-Za-z_]\w*|\d+)$/);
  if (subtraction) return Math.max(1, symbolValue(subtraction[1], shape, block) - symbolValue(subtraction[2], shape, block));
  return 1;
}

function formulaElements(formula: string | null, shape: number[], block: number[], fallback: number) {
  if (!formula || formula === "1") return formula === "1" ? 1 : fallback;
  const value = formula.split("×").reduce((product, factor) => product * symbolValue(factor, shape, block), 1);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function pointerFromLoad(segment: KernelSourceSegment) {
  return segment.variables.find(name => /(_ptr$|cache|buffer|table|indices)/i.test(name))
    ?? segment.code.match(/tl\.load\(\s*([A-Za-z_]\w*)/)?.[1]
    ?? "source pointer";
}

function ubModel(segments: KernelSourceSegment[], shape: number[], block: number[], exactMulAdd: boolean) {
  if (exactMulAdd) {
    const elements = 4096;
    const items: UbItem[] = [
      { name:"routed_values", source:"input1_ptr", formula:"BLOCK_SIZE = 4096", elements, bytesPerElement:2, bytes:8192, phase:"load" },
      { name:"shared_values", source:"input2_ptr", formula:"BLOCK_SIZE = 4096", elements, bytesPerElement:2, bytes:8192, phase:"load" },
      { name:"buffered_values", source:"routed_values × factor + shared_values", formula:"BLOCK_SIZE = 4096", elements, bytesPerElement:4, bytes:16384, phase:"compute" },
    ];
    return { items, peakBytes:32768, exact:true, formula:"8 KiB routed + 8 KiB shared + 16 KiB FP32 result = 32 KiB" };
  }
  const loads: UbItem[] = [];
  const seenLoads = new Set<string>();
  for (const segment of segments.filter(item => item.phase === "load" && item.result)) {
    if (seenLoads.has(segment.result!)) continue;
    seenLoads.add(segment.result!);
    const source = pointerFromLoad(segment);
    const elements = formulaElements(segment.tileFormula, shape, block, block[0] * block[1] * block[2]);
    const bytesPerElement = /(idx|index|offset|table|length|lens)/i.test(`${source} ${segment.result}`) ? 4 : 2;
    loads.push({ name:segment.result!, source, formula:segment.tileFormula ?? "scalar", elements, bytesPerElement, bytes:elements * bytesPerElement, phase:"load" });
  }
  const computes: UbItem[] = [];
  const seenComputes = new Set<string>();
  for (const segment of segments.filter(item => item.phase === "compute" && item.result && item.tileFormula && item.tileFormula !== "1")) {
    if (seenComputes.has(segment.result!)) continue;
    seenComputes.add(segment.result!);
    const elements = formulaElements(segment.tileFormula, shape, block, block[0] * block[1] * block[2]);
    computes.push({ name:segment.result!, source:`L${segment.lineStart} compute result`, formula:segment.tileFormula!, elements, bytesPerElement:4, bytes:elements * 4, phase:"compute" });
  }
  const largestTemporary = computes.reduce((largest, item) => item.bytes > largest.bytes ? item : largest, { bytes:0 } as UbItem);
  const peakBytes = loads.reduce((sum, item) => sum + item.bytes, 0) + largestTemporary.bytes;
  return { items:[...loads, ...computes], peakBytes, exact:false, formula:`Σ 当前 load tile + 最大 FP32 compute tile = ${peakBytes} B` };
}

function stableNumber(text: string) {
  let h = 17;
  for (const c of text) h = (h * 31 + c.charCodeAt(0)) % 997;
  return h;
}

function sampleValue(name: string, kind: ArgKind, shape: number[], block: number[], lane = 0) {
  const n = name.toLowerCase();
  if (kind === "flag") return "true";
  if (kind === "stride") {
    if (/(_d|dim|col|token)$/.test(n)) return "1 element";
    if (/(_h|head|window)$/.test(n)) return `${shape[2]} elements`;
    return `${shape[1] * shape[2]} elements`;
  }
  if (kind === "block") return String(/(_d|_n|_c|hidden|col|tail|block$)/.test(n) ? block[2] : block[0]);
  if (kind === "shape") {
    if (/batch|rows|tokens|^b$|^m$/.test(n)) return String(shape[0]);
    if (/head|^h$/.test(n)) return String(shape[1]);
    return String(shape[2]);
  }
  if (kind === "scalar") return /eps/.test(n) ? "1e-6" : /scale|alpha|beta/.test(n) ? "0.125" : /limit|threshold/.test(n) ? "7.0" : "1";
  if (kind === "helper") return "compiler supplied";
  const raw = ((stableNumber(name) % 19) - 9) / 4 + lane * .125;
  return raw.toFixed(3);
}

function addressFormula(name: string, kind: ArgKind, coord: Coord) {
  if (kind === "index") return `${name}[task] → physical=${(coord.flat * 7 + 3) % 23} → base + physical × row_stride + lane`;
  if (kind === "output" || kind === "inout") return `${name} + (${coord.global[0]}×stride₀ + ${coord.global[1]}×stride₁ + ${coord.global[2]})`;
  return `${name} + pid·BLOCK + lane = base + ${coord.flat} elements`;
}

function splitCompute(text: string) {
  return text.split(/；|→/).map(x => x.trim()).filter(Boolean);
}

function unflattenTask(task: number, grid: number[]) {
  return [Math.floor(task / (grid[1] * grid[2])), Math.floor(task / grid[2]) % grid[1], task % grid[2]];
}

function persistentTask(program: number, round: number, totalTasks: number, programs: number, mode: "strided" | "contiguous" = "strided") {
  if (mode === "strided") {
    const task = program + round * programs;
    return task < totalTasks ? task : null;
  }
  const base = Math.floor(totalTasks / programs);
  const extra = totalTasks % programs;
  const count = base + (program < extra ? 1 : 0);
  const start = program < extra ? program * (base + 1) : extra * (base + 1) + (program - extra) * base;
  return round < count ? start + round : null;
}

function KernelProductionLab() {
  const params = useSearchParams();
  const initialId = params.get("op") || KERNEL_OPS[0].id;
  const initialOp = KERNEL_OPS.find(x => x.id === initialId) ?? KERNEL_OPS[0];
  const [selectedId, setSelectedId] = useState(initialOp.id);
  const [kernelIndex, setKernelIndex] = useState(() => Math.max(0, initialOp.kernels.indexOf(params.get("kernel") || "")));
  const [group, setGroup] = useState("全部");
  const [query, setQuery] = useState("");
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [round, setRound] = useState(0);
  const selected = KERNEL_OPS.find(x => x.id === selectedId) ?? KERNEL_OPS[0];
  const [shape, setShape] = useState(() => profileFor(selected, selected.kernels[Math.min(kernelIndex, selected.kernels.length - 1)]).shape);
  const [block, setBlock] = useState(() => profileFor(selected, selected.kernels[Math.min(kernelIndex, selected.kernels.length - 1)]).block);
  const [pid, setPid] = useState([0,0,0]);
  const [variable, setVariable] = useState("");
  const [segmentId, setSegmentId] = useState("");
  const [sourceModule, setSourceModule] = useState<{ module: string; kernels: Record<string, KernelSourceSegment[]> }>({ module:"", kernels:{} });
  const kernel = selected.kernels[Math.min(kernelIndex, selected.kernels.length - 1)];
  const signature = KERNEL_SIGNATURES[selected.module]?.[kernel];
  const callBinding = KERNEL_CALL_BINDINGS[selected.module]?.[kernel];
  const sourceSegments = sourceModule.module === selected.module ? sourceModule.kernels[kernel] ?? [] : [];
  const activeSegment = sourceSegments.find(item => item.id === segmentId);
  const args = useMemo(() => signature?.args ?? [], [signature]);
  const profile = profileFor(selected, kernel);

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => setStep(s => s === STEPS.length - 1 ? 0 : s + 1), 1100);
    return () => window.clearInterval(timer);
  }, [playing]);

  useEffect(() => {
    let cancelled = false;
    const loader = KERNEL_SOURCE_LOADERS[selected.module];
    if (loader) loader().then(kernels => { if (!cancelled) setSourceModule({ module:selected.module, kernels }); });
    return () => { cancelled = true; };
  }, [selected.module]);

  const argRows = args.map(name => {
    const kind = classifyArg(name, signature?.loads, signature?.stores);
    return { name, kind, ...meaningFromName(name, kind, selected, callBinding?.args[name], callBinding?.wrapper, callBinding?.shapes[name]), loaded:(signature?.loads.includes(name) ?? false) && pointerArg(name), stored:(signature?.stores.includes(name) ?? false) && pointerArg(name) };
  });
  const pointerRows = argRows.filter(x => ["input","output","inout","index"].includes(x.kind));
  const inputs = pointerRows.filter(x => x.kind !== "output");
  const outputs = pointerRows.filter(x => x.kind === "output" || x.kind === "inout");
  const selectedVar = argRows.find(x => x.name === variable) ?? pointerRows[0] ?? argRows[0];
  const grid = shape.map((n,i) => Math.max(1, Math.ceil(n / Math.max(1, block[i]))));
  const logicalTasks = grid[0] * grid[1] * grid[2];
  const physicalPrograms = profile.persistent ? Math.min(profile.programCap ?? 8, logicalTasks) : logicalTasks;
  const persistentRounds = profile.persistent ? Math.ceil(logicalTasks / physicalPrograms) : 1;
  const maxRound = persistentRounds - 1;
  const directFlatPid = (pid[0] * grid[1] + pid[1]) * grid[2] + pid[2];
  const physicalPid = profile.persistent ? pid[0] : directFlatPid;
  const assignedPersistentTask = profile.persistent ? persistentTask(physicalPid, round, logicalTasks, physicalPrograms, profile.persistentMode) : null;
  const task = profile.persistent ? assignedPersistentTask ?? logicalTasks : directFlatPid;
  const taskIsValid = profile.persistent ? assignedPersistentTask !== null : task < logicalTasks;
  const taskPid = profile.persistent ? taskIsValid ? unflattenTask(task, grid) : [0,0,0] : pid;
  const tileElements = block[0] * block[1] * block[2];
  const shownLanes = Math.min(32, tileElements);
  const coords: Coord[] = Array.from({ length: shownLanes }, (_, lane) => {
    const l2 = lane % block[2];
    const l1 = Math.floor(lane / block[2]) % block[1];
    const l0 = Math.floor(lane / (block[2] * block[1]));
    const local = [l0,l1,l2];
    const global = local.map((v,i) => taskPid[i] * block[i] + v);
    const valid = taskIsValid && global.every((v,i) => v < shape[i]);
    return { lane, local, global, valid, flat:(global[0] * shape[1] + global[1]) * shape[2] + global[2] };
  });
  const activeCoord = coords.find(x => x.valid) ?? coords[0];
  const ub = ubModel(sourceSegments, shape, block, selected.id === "muladd" && kernel === "mul_add_kernel" && shape[2] === 4096 && block[2] === 4096);
  const sourceUrl = `${SOURCE_ROOT}/${selected.module}#L${signature?.line ?? 1}`;
  const visible = KERNEL_OPS.filter(op => (group === "全部" || op.group === group) && `${op.title} ${op.module} ${op.kernels.join(" ")}`.toLowerCase().includes(query.toLowerCase()));

  const chooseOp = (op: KernelOp) => {
    const p = profileFor(op, op.kernels[0]);
    setSelectedId(op.id); setKernelIndex(0); setShape(p.shape); setBlock(p.block); setPid([0,0,0]); setRound(0); setStep(0); setPlaying(false); setVariable(""); setSegmentId("");
    window.history.replaceState(null, "", `/kernel-lab?op=${encodeURIComponent(op.id)}`);
  };
  const changeAxis = (setter: (v: number[]) => void, values: number[], axis: number, value: number) => {
    setRound(0);
    setter(values.map((x,i) => i === axis ? Math.max(1,value) : x));
  };
  const displayPrograms = Array.from({ length: Math.min(48, profile.persistent ? physicalPrograms : logicalTasks) }, (_, flat) => {
    if (profile.persistent) return { flat, p:[flat,0,0] };
    return { flat, p:[Math.floor(flat/(grid[1]*grid[2])), Math.floor(flat/grid[2])%grid[1], flat%grid[2]] };
  });
  const chooseSegment = (segment: KernelSourceSegment) => {
    setSegmentId(segment.id); setStep(PHASE_STEP[segment.phase]); setPlaying(false);
    const matchingVariable = segment.variables.find(name => args.includes(name));
    if (matchingVariable) setVariable(matchingVariable);
  };

  return <main className="production-lab-page variable-semantics-v2">
    <header className="pl-topbar">
      <Link href="/#cases">← 返回教程</Link>
      <div><b>Triton Ascend · Production Kernel Lab</b><span>{KERNEL_OPS.length} modules / {TOTAL_TRITON_KERNELS} JIT kernels · source cutoff {VISUAL_SOURCE.shortCommit}</span></div>
      <Link href="/profiling">Profiling 实测 ↗</Link>
    </header>

    <aside className="pl-sidebar">
      <div className="pl-search"><label>搜索真实算子或 kernel<input value={query} onChange={e => setQuery(e.target.value)} placeholder="rmsnorm / cache / decode" /></label></div>
      <div className="pl-groups">{KERNEL_GROUPS.map(g => <button key={g} className={group === g ? "active" : ""} onClick={() => setGroup(g)}>{g}</button>)}</div>
      <div className="pl-op-list">{visible.map(op => <button key={op.id} className={selected.id === op.id ? "active" : ""} onClick={() => chooseOp(op)}><small>{op.module}</small><b>{op.title}</b><span>{op.kernels.length} JIT</span></button>)}</div>
    </aside>

    <div className="pl-main">
      <section className="pl-title">
        <div><span>{selected.group} · {selected.module}</span><h1>{selected.title}</h1><p>{selected.input}</p></div>
        <div className="pl-title-actions"><a href={sourceUrl} target="_blank" rel="noreferrer">在 GitHub 打开该 kernel ↗</a><a href={`${VISUAL_SOURCE.repository}/commit/${VISUAL_SOURCE.commit}`} target="_blank" rel="noreferrer">可视化截止 commit {VISUAL_SOURCE.shortCommit} ↗</a><code>{profile.persistent ? "PERSISTENT GRID" : "DIRECT GRID"}</code></div>
      </section>

      <section className="pl-kernel-picker"><b>本模块 JIT kernel</b>{selected.kernels.map((name,i) => <button key={name} className={i === kernelIndex ? "active" : ""} onClick={() => { const p = profileFor(selected, name); setKernelIndex(i); setShape(p.shape); setBlock(p.block); setPid([0,0,0]); setRound(0); setStep(0); setVariable(""); setSegmentId(""); window.history.replaceState(null,"",`/kernel-lab?op=${selected.id}&kernel=${encodeURIComponent(name)}`); }}>{name}</button>)}</section>

      <section className="pl-signature"><span>SOURCE SIGNATURE · line {signature?.line ?? "?"} · {sourceSegments.length} executable statements · commit {VISUAL_SOURCE.shortCommit}</span><code>def {kernel}({args.join(", ")})</code><p>变量清单来自该 <code>@triton.jit</code> 签名；下方每一段可执行源码都可点击，并会跳到对应的 Grid / Offset / Load / Compute / Store 可视化阶段。</p></section>

      <section className="pl-controls">
        <div><span>教学输入 shape</span>{profile.labels.map((label,i) => <label key={label}>{label}<input type="number" min="1" value={shape[i]} onChange={e => changeAxis(setShape,shape,i,+e.target.value)} /></label>)}</div>
        <div><span>当前 program 的 BLOCK / tile</span>{profile.labels.map((label,i) => <label key={label}>BLOCK_{i}<input type="number" min="1" value={block[i]} onChange={e => changeAxis(setBlock,block,i,+e.target.value)} /></label>)}</div>
        <div className="pl-derived"><p><span>源码 Grid</span><code>{selected.grid}</code></p><p><span>教学展开</span><code>grid = ({grid.join(", ")}) · {logicalTasks} tasks</code></p><p><span>当前 Tile</span><code>{block.join(" × ")} = {tileElements} lanes</code></p></div>
      </section>

      <nav className="pl-steps" aria-label="模拟步骤">{STEPS.map(([key,label],i) => <button key={key} className={step === i ? "active" : step > i ? "done" : ""} onClick={() => {setStep(i);setPlaying(false);setSegmentId("")}}><span>{String(i+1).padStart(2,"0")}</span><b>{key}</b><small>{label}</small></button>)}</nav>

      <section className="pl-step-summary"><div><span>当前阶段</span><strong>{STEPS[step][1]}</strong></div><p>{activeSegment?.explanation ?? [selected.grid, `program ${taskPid.join(",")} 正在处理 task ${task}`, `lane → local → global → flat offset；尾块由 mask 保护`, selected.load, `${selected.tile}；当前片上工作集 ${(ub.peakBytes/1024).toFixed(2)} KiB`, selected.compute, selected.store, selected.torch][step]}</p></section>

      <section className={`pl-grid-panel ${step <= 1 ? "focus" : ""}`}>
        <header><div><span>GRID / PROGRAM MAP</span><h2>Grid 怎样拆成并行 task</h2></div><p>{profile.persistent ? profile.persistentMode === "contiguous" ? <><code>每个 pid 连续领取一段 task</code>；前 {logicalTasks % physicalPrograms || physicalPrograms} 个 program 每个处理 {Math.ceil(logicalTasks / physicalPrograms)} 个 task，其余处理 {Math.floor(logicalTasks / physicalPrograms)} 个；P{physicalPid} 当前 k={round} → {taskIsValid ? `task ${task}` : "idle"}。</> : <><code>task = pid + k × P</code>；P={physicalPrograms}，当前 k={round}，所以 task={physicalPid}+{round}×{physicalPrograms}={task}。</> : <>每个 program 直接领取一个逻辑 tile；program 坐标乘 BLOCK 得到 tile 起点。</>}</p></header>
        {profile.persistent && <div className="pl-round">
          <label htmlFor="persistent-round">persistent 轮次 k</label>
          <button type="button" onClick={() => setRound(current => Math.max(0, current - 1))} disabled={round === 0} aria-label="上一轮">−</button>
          <input id="persistent-round" type="range" min={0} max={maxRound} step={1} value={round} onInput={e => setRound(Number(e.currentTarget.value))} onChange={e => setRound(Number(e.currentTarget.value))} aria-label="persistent 轮次 k" data-round-max={maxRound} />
          <button type="button" onClick={() => setRound(current => Math.min(maxRound, current + 1))} disabled={round === maxRound} aria-label="下一轮">+</button>
          <output htmlFor="persistent-round">k = <b>{round}</b> / {maxRound}</output>
          <span>共 {persistentRounds} 轮；可拖动滑块、点 ± 或用方向键。第 {round} 轮只有仍分配到 task 的 program 工作。</span>
        </div>}
        <div className="pl-program-map">{displayPrograms.map(item => {
          const assignedTask = profile.persistent ? persistentTask(item.flat, round, logicalTasks, physicalPrograms, profile.persistentMode) : directFlatPid;
          const assignedPid = unflattenTask(assignedTask ?? 0, grid);
          const idle = profile.persistent && assignedTask === null;
          return <button key={item.flat} className={`${profile.persistent ? physicalPid === item.flat ? "active" : "" : pid.every((x,i)=>x===item.p[i]) ? "active" : ""} ${idle ? "idle" : ""}`} onClick={() => !idle && setPid(item.p)} disabled={idle}><b>P{item.flat}</b><span>{profile.persistent ? idle ? "idle" : `task ${assignedTask}` : `[${item.p.join(",")}]`}</span><small>{idle ? "本轮没有剩余 task" : `origin [${(profile.persistent ? assignedPid : item.p).map((x,i)=>x*block[i]).join(",")}]`}</small></button>;
        })}</div>
        {!profile.persistent && logicalTasks > 48 && <p className="pl-clipped">只绘制前 48 个 program；完整 Grid 共 {logicalTasks} 个，计算规则相同。</p>}
      </section>

      <section className={`pl-offset-panel ${step === 2 ? "focus" : ""}`}>
        <header><div><span>BLOCK / LANE ADDRESSING</span><h2>当前 program 内每个 lane 的坐标</h2></div><code>global[d] = pid[d] × BLOCK[d] + local[d]</code></header>
        <div className="pl-offset-formulas"><p><span>program 坐标</span><b>[{taskPid.join(", ")}]</b></p><p><span>tile 起点</span><b>[{taskPid.map((x,i)=>x*block[i]).join(", ")}]</b></p><p><span>展平公式</span><b>(g₀ × shape₁ + g₁) × shape₂ + g₂</b></p><p><span>mask</span><b>g₀&lt;S₀ &amp;&amp; g₁&lt;S₁ &amp;&amp; g₂&lt;S₂</b></p></div>
        <div className="pl-lane-table"><div><b>LANE</b><b>LOCAL COORD</b><b>GLOBAL COORD</b><b>FLAT OFFSET</b><b>MASK</b></div>{coords.map(c => <div key={c.lane} className={c.valid ? "valid" : "masked"}><code>{c.lane}</code><code>[{c.local.join(",")}]</code><code>[{c.global.join(",")}]</code><strong>{c.flat}</strong><b>{String(c.valid).toUpperCase()}</b></div>)}</div>
        {tileElements > shownLanes && <p className="pl-clipped">当前 tile 共 {tileElements} lanes，为保持可读性展示前 {shownLanes} 个；其余 lane 按同一公式继续展开。</p>}
      </section>

      <section className={`pl-variable-panel ${step === 3 || step === 6 ? "focus" : ""}`}>
        <header><div><span>ALL KERNEL ARGUMENTS</span><h2>每个变量是什么、什么 shape、从模型哪里传入</h2></div><p><b>{args.length}</b> 个签名变量 · 指针 {pointerRows.length} · 标量/形状/stride {args.length-pointerRows.length}</p></header>
        <div className="pl-variable-grid">{argRows.map((arg,i) => <button key={`${arg.name}-${i}`} className={`${arg.kind} ${selectedVar?.name === arg.name ? "active" : ""}`} onClick={() => setVariable(arg.name)}><span>{String(i+1).padStart(2,"0")} · {KIND_TEXT[arg.kind]} · {arg.loaded ? "tl.load" : "no load"}{arg.stored ? " + tl.store" : ""}</span><code>{arg.name}</code><p className="pl-arg-meaning">{arg.meaning}</p><dl><div><dt>SHAPE</dt><dd>{arg.shape}</dd></div><div><dt>调用时</dt><dd>{arg.caller}</dd></div></dl></button>)}</div>
        {selectedVar && <div className="pl-variable-trace"><div><span>选中变量</span><code>{selectedVar.name}</code><b>{KIND_TEXT[selectedVar.kind]}</b></div><p><span>变量含义</span><b>{selectedVar.meaning}</b></p><p><span>实际 / 符号 shape</span><code>{selectedVar.shape}</code></p><p><span>模型调用时对应</span><b>{selectedVar.caller}</b></p><p><span>当前 lane 地址</span><code>{addressFormula(selectedVar.name,selectedVar.kind,activeCoord)}</code></p></div>}
      </section>

      <section className={`pl-memory-panel ${step >= 3 && step <= 6 ? "focus" : ""}`}>
        <header><div><span>GM → UB / REG → GM</span><h2>真实变量搬运与计算流水</h2></div><p>蓝色是 load，黄绿色是片上存活值，橙色是 store。每列都是当前 program 的一轮，不乘整个 Grid。</p></header>
        <div className="pl-memory-flow">
          <div className={`pl-mem-column gm ${step===3 ? "active" : ""}`}><span>GLOBAL MEMORY · INPUT</span>{inputs.map((arg,i)=><button key={`${arg.name}-${i}`} onClick={()=>setVariable(arg.name)} className={selectedVar?.name===arg.name?"selected":""}><code>{arg.name}</code><small>{arg.kind === "index" ? "先取索引，再二次寻址" : addressFormula(arg.name,arg.kind,activeCoord)}</small><b>{coords.slice(0,8).map(c=>c.valid?sampleValue(arg.name,arg.kind,shape,block,c.lane):"×").join("  ")}</b></button>)}</div>
          <div className={`pl-flow-arrow load ${step===3 ? "active" : ""}`}><b>tl.load</b><span>mask + other</span><i>→</i></div>
          <div className={`pl-mem-column ub ${step===4 || step===5 ? "active" : ""}`}><span>UNIFIED BUFFER / REG</span>{ub.items.map((item,i)=><div className={item.phase === "compute" ? "acc" : ""} key={`${item.name}-${i}`}><code>{item.name}</code><small>{item.formula} → {item.elements.toLocaleString()} elements × {item.bytesPerElement} B = {(item.bytes/1024).toFixed(2)} KiB</small><b>{item.phase === "load" ? `来自 ${item.source}` : `计算结果：${item.source}`}</b></div>)}{!ub.items.length && <div><code>scalar / compiler value</code><small>该 helper 没有显式 tile load；片上值由调用方传入。</small></div>}<strong>{ub.exact ? "编译产物核对" : "源码级工作集上界"} = {(ub.peakBytes/1024).toFixed(2)} KiB</strong></div>
          <div className={`pl-compute ${step===5 ? "active" : ""}`}><span>COMPUTE</span>{splitCompute(selected.compute).map((x,i)=><div key={`${x}-${i}`}><b>{i+1}</b><code>{x}</code></div>)}</div>
          <div className={`pl-flow-arrow store ${step===6 ? "active" : ""}`}><b>tl.store</b><span>same mask</span><i>→</i></div>
          <div className={`pl-mem-column out ${step===6 ? "active" : ""}`}><span>GLOBAL MEMORY · OUTPUT / STATE</span>{(outputs.length ? outputs : [{name:"return / compiler value",kind:"output" as ArgKind}]).map((arg,i)=><button key={`${arg.name}-${i}`} onClick={()=>setVariable(arg.name)}><code>{arg.name}</code><small>{addressFormula(arg.name,arg.kind,activeCoord)}</small><b>{coords.slice(0,8).map(c=>c.valid?`y${c.lane}`:"skip").join("  ")}</b></button>)}</div>
        </div>
        <div className="pl-memory-controls" aria-label="搬运播放与单步控制">
          <div><span>TRANSFER PLAYBACK</span><b>{String(step + 1).padStart(2,"0")} / {String(STEPS.length).padStart(2,"0")} · {STEPS[step][1]}</b><small>控制紧跟搬运图；可连续播放，也可逐阶段观察 Grid、Load、UB、Compute 和 Store。</small></div>
          <div className="pl-play">
            <button onClick={() => { setStep(Math.max(0,step-1)); setPlaying(false); }} disabled={step===0} aria-label="单步后退">← 单步后退</button>
            <button className="play" onClick={() => setPlaying(!playing)} aria-label={playing ? "暂停播放" : "播放搬运"}>{playing ? "暂停播放" : "播放搬运"}</button>
            <button onClick={() => { setStep(Math.min(STEPS.length-1,step+1)); setPlaying(false); }} disabled={step===STEPS.length-1} aria-label="单步前进">单步前进 →</button>
          </div>
        </div>
        <div className="pl-ub-ledger"><b>UB 逐项计算</b><code>{ub.formula}</code><span>{ub.exact ? <>209 编译缓存的 <code>.ascend.stack.size.record = 0x8000 = 32 KiB</code>，与上面逐项计算完全一致。</> : <>这是源码显式 load tile 加最大 FP32 中间 tile 的教学上界；编译器可能通过生命周期复用降低占用，也可能因对齐或双缓冲增加占用。</>}</span></div>
        <div className="pl-source-map-head"><div><span>SOURCE → VISUAL STAGE</span><h2>每段代码如何驱动可视化</h2></div><p>以下列出本 kernel 的全部 {sourceSegments.length} 段可执行语句。点击任意一段，页面会切换到对应阶段；行号和链接固定在 <code>{VISUAL_SOURCE.shortCommit}</code>。</p></div>
        <div className="pl-source-phase-legend">{Object.entries(PHASE_TEXT).map(([phase,label]) => <span className={phase} key={phase}>{label}</span>)}</div>
        <div className="pl-source-segments">{sourceSegments.map((segment,index) => <article key={segment.id} className={`${segment.phase} ${segment.id === segmentId ? "active" : ""}`}><button className="pl-source-code" onClick={() => chooseSegment(segment)}><span><b>{String(index+1).padStart(2,"0")} · {PHASE_TEXT[segment.phase]}</b><i>L{segment.lineStart}{segment.lineEnd !== segment.lineStart ? `–${segment.lineEnd}` : ""}</i></span><pre><code>{segment.code}</code></pre><p>{segment.explanation}</p>{segment.tileFormula && <small>tile 公式：{segment.tileFormula} → {formulaElements(segment.tileFormula,shape,block,tileElements).toLocaleString()} elements</small>}</button><a href={`${SOURCE_ROOT}/${selected.module}#L${segment.lineStart}`} target="_blank" rel="noreferrer">打开固定版本源码 ↗</a></article>)}</div>
      </section>

      <section className={`pl-torch-panel ${step===7 ? "focus" : ""}`}>
        <div><span>TRITON 过程</span><p><b>Grid</b>{selected.grid}</p><p><b>Load</b>{selected.load}</p><p><b>Compute</b>{selected.compute}</p><p><b>Store</b>{selected.store}</p></div>
        <div><span>PYTORCH 语义参考</span><pre><code>{selected.torch}</code></pre><p>PyTorch 对照保证数学语义；它不会复现 Triton 的 program 数量、地址向量、UB 生命周期或 NPU 并行映射。</p><p className="pl-call-sources"><a href={sourceUrl} target="_blank" rel="noreferrer">sgl-kernel-npu wrapper / kernel ↗</a><a href={`${SGLANG_ROOT}/srt`} target="_blank" rel="noreferrer">SGLang 模型与后端调用点 ↗</a></p></div>
      </section>

      <p className="pl-boundary"><b>模拟边界：</b>坐标、mask、stride 公式和源码签名是可核对的；页面使用缩小后的教学 shape 展开。最终 GM 合并访存、UB/寄存器实际分配、Vector/Cube 映射、流水与双缓冲由 Triton‑Ascend 编译器和具体 NPU 决定。</p>
    </div>
  </main>;
}

export default KernelProductionLab;
