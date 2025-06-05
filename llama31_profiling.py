import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.profiler
import os
import gc
import time

# --- 配置参数 ---
# 为了演示，我们使用一个较小的模型。将其更改为 LLaMA 3.1 70B 进行实际分析。
# MODEL_NAME = "meta-llama/Meta-Llama-3.1-70B-Instruct"
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B" # 替换为实际模型或使用此模型进行测试

DEVICE = "xpu" if torch.xpu.is_available() else "cpu"
DTYPE = torch.bfloat16 # 70B模型推荐使用bfloat16以节省内存和加速

PROMPT = "Once upon a time, in a land far, far away,"
MAX_TOKENS_FOR_FIRST = 1  # 只生成第一个token
MAX_TOKENS_FOR_REST = 10  # 为"rest tokens"阶段生成多个token (总共会生成这么多，不包括第一个)
NUM_WARMUP_RUNS = 2
PROFILE_OUTPUT_DIR = f"logs/profile_output_{MODEL_NAME.split('/')[-1]}"

# --- 辅助函数 ---
def setup_model_and_tokenizer(model_name, dtype, device):
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model {model_name} to {device} with dtype {dtype}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        # low_cpu_mem_usage=True, # For very large models to reduce CPU RAM usage during loading
    )
    model = model.to(device)
    model.eval() # 设置为评估模式
    print("Model and tokenizer loaded.")
    return model, tokenizer

def generate_and_profile(
    model,
    tokenizer,
    inputs,
    max_new_tokens,
    profile_name,
    profile_dir,
    use_cache=True,
    is_rest_token_phase=False,
    past_key_values=None
):
    """
    执行生成并进行性能分析。
    对于 "rest token" 阶段，我们会传入 past_key_values。
    """
    print(f"\nProfiling: {profile_name}, Max new tokens: {max_new_tokens}")
    os.makedirs(profile_dir, exist_ok=True)
    trace_file = os.path.join(profile_dir, f"{profile_name.replace(' ', '_').lower()}.json")

    # 确保输入在正确的设备上
    current_input_ids = inputs["input_ids"].to(model.device)
    current_attention_mask = inputs["attention_mask"].to(model.device)

    # 如果是rest_token_phase，则input_ids应该是上一步生成的最后一个token
    # past_key_values 和 attention_mask 也需要相应地设置
    if is_rest_token_phase:
        # `current_input_ids` 应该是上一步输出的最后一个token
        # `current_attention_mask` 应该被扩展
        # `past_key_values` 由调用者提供
        pass # 假设调用者已正确准备了 inputs for rest phase

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.XPU]
    schedule = torch.profiler.schedule(
            wait=0,
            warmup=0,
            active=1)
    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        record_shapes=True,  # 关键：捕获tensor shapes
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir, worker_name=profile_name)
    ) as prof:
        with torch.no_grad(): # 推理时不需要梯度
             # 使用 record_function 来标记代码块，这在分析 trace 时很有用
            with torch.profiler.record_function(f"generate_tokens_{profile_name}"):
                outputs = model.generate(
                    current_input_ids,
                    attention_mask=current_attention_mask,
                    max_new_tokens=max_new_tokens,
                    use_cache=use_cache,
                    pad_token_id=tokenizer.eos_token_id,
                    past_key_values=past_key_values,
                    return_dict_in_generate=True # 为了获取past_key_values
                )
    
    print(f"Profiling data for '{profile_name}' saved.")
    print(f"Trace (for Perfetto/Tensorboard): {os.path.join(profile_dir, profile_name + '.pt.trace.json')}") # Tensorboard handler creates this
    
    # 打印关键指标
    # group_by_input_shape 对于分析 OP 的不同 Tensor Shape 调用非常有用
    # group_by_stack_n 可以帮助追溯 OP 的调用来源
    print(f"\n--- Top XPU time ops for {profile_name} (grouped by input shape) ---")
    try:
        # 使用 self_xpu_time_total 排序，如果不存在，则使用 xpu_time_total 或 self_cpu_time_total
        sort_key = "self_xpu_time_total"
        if not any(e.self_xpu_time_total > 0 for e in prof.key_averages()):
            sort_key = "xpu_time_total"
        if not any(e.xpu_time_total > 0 for e in prof.key_averages()):
             sort_key = "self_cpu_time_total" # Fallback if no XPU time

        print(prof.key_averages(group_by_input_shape=True).table(sort_by=sort_key, row_limit=15))
    except Exception as e:
        print(f"Could not print key_averages: {e}")
        print("Printing raw key_averages:")
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))
    


    # 清理显存
    del outputs
    if 'kv_cache' in locals(): del kv_cache
    gc.collect()
    if DEVICE == "xpu":
        torch.xpu.empty_cache()
    
    # 返回 generate 的输出，以便用于 "rest tokens" 阶段
    # 需要从 model.generate 获取 past_key_values
    # For this script's structure, we'll re-generate the first token to get KV cache
    # if needed, rather than passing it complexly through `main`.
    # However, the `model.generate` output can contain it.
    # For simplicity in this example, we'll have separate profiling runs.
    # The "rest tokens" run will implicitly contain its own "first token" part.
    # A more precise "rest token" profiling would involve:
    # 1. Generate 1 token, get past_key_values.
    # 2. Profile generation of N tokens *using* those past_key_values.
    # This is implemented in the `run_mode_profile` function.

def run_mode_profile(model_to_profile, tokenizer, mode_name, prompt_text, profile_base_dir):
    print(f"\n{'='*20} Profiling {mode_name} Mode {'='*20}")
    
    # 准备通用输入
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids_prompt = inputs["input_ids"].to(model_to_profile.device)
    attention_mask_prompt = inputs["attention_mask"].to(model_to_profile.device)

    # --- 1. First Token Profiling ---
    profile_name_first = f"{mode_name} First Token"
    profile_dir_first = os.path.join(profile_base_dir, mode_name.lower().replace(' ', '_'), "first_token")
    
    # Warmup for first token (important for compiled model especially)
    print(f"Warmup for {profile_name_first}...")
    with torch.no_grad():
        for _ in range(NUM_WARMUP_RUNS):
            _ = model_to_profile.generate(
                input_ids_prompt,
                attention_mask=attention_mask_prompt,
                max_new_tokens=MAX_TOKENS_FOR_FIRST,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True 
            )
            if DEVICE == "xpu": torch.xpu.synchronize() # ensure warmup completes
    if DEVICE == "xpu": torch.xpu.empty_cache()
    gc.collect()
    
    # Actual profiling for first token
    first_token_inputs = {"input_ids": input_ids_prompt, "attention_mask": attention_mask_prompt}
    generate_and_profile(
        model_to_profile, tokenizer, first_token_inputs,
        max_new_tokens=MAX_TOKENS_FOR_FIRST,
        profile_name=profile_name_first,
        profile_dir=profile_dir_first,
        use_cache=True, # KV cache is built during first token generation
        is_rest_token_phase=False
    )
    
    # 获取第一次生成的结果，用于准备 "rest tokens" 的输入
    with torch.no_grad():
        outputs_after_first = model_to_profile.generate(
            input_ids_prompt,
            attention_mask=attention_mask_prompt,
            max_new_tokens=MAX_TOKENS_FOR_FIRST,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True
        )
    past_key_values_from_first = outputs_after_first.past_key_values
    # `input_ids` for rest phase should be the last token generated
    rest_phase_input_ids = outputs_after_first.sequences[:, -1:]
    # `attention_mask` for rest phase needs to be extended
    # Concatenate original attention mask with a new mask for the generated token(s)
    # For a single next token, it's a tensor of ones with shape (batch_size, 1)
    ones_for_new_tokens = torch.ones((rest_phase_input_ids.shape[0], rest_phase_input_ids.shape[1]), dtype=torch.long, device=model_to_profile.device)
    rest_phase_attention_mask = torch.cat([attention_mask_prompt, ones_for_new_tokens], dim=1)
    
    if DEVICE == "xpu": torch.xpu.empty_cache()
    gc.collect()
    time.sleep(2) # Small pause

    # --- 2. Rest Tokens Profiling ---
    # This profiles the generation of subsequent tokens, using the KV cache from the first token.
    profile_name_rest = f"{mode_name} Rest Tokens"
    profile_dir_rest = os.path.join(profile_base_dir, mode_name.lower().replace(' ', '_'), "rest_tokens")

    # Warmup for rest tokens (using KV cache)
    print(f"Warmup for {profile_name_rest}...")
    with torch.no_grad():
        for _ in range(NUM_WARMUP_RUNS):
            _ = model_to_profile.generate(
                rest_phase_input_ids, # Only the last token
                attention_mask=rest_phase_attention_mask, # Extended attention mask
                max_new_tokens=MAX_TOKENS_FOR_REST,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                past_key_values=past_key_values_from_first, # Crucial: use KV cache
                return_dict_in_generate=True
            )
            if DEVICE == "xpu": torch.xpu.synchronize()
    if DEVICE == "xpu": torch.xpu.empty_cache()
    gc.collect()

    # Actual profiling for rest tokens
    rest_tokens_inputs = {"input_ids": rest_phase_input_ids, "attention_mask": rest_phase_attention_mask}
    generate_and_profile(
        model_to_profile, tokenizer, rest_tokens_inputs,
        max_new_tokens=MAX_TOKENS_FOR_REST,
        profile_name=profile_name_rest,
        profile_dir=profile_dir_rest,
        use_cache=True,
        is_rest_token_phase=True,
        past_key_values=past_key_values_from_first
    )
    
    if DEVICE == "xpu": torch.xpu.empty_cache()
    gc.collect()
    time.sleep(2) # Small pause

def main():
    print(f"Using device: {DEVICE}")
    if DEVICE == "xpu":
        print(f"XPU Name: {torch.xpu.get_device_name(0)}")
    
    # 创建输出目录
    os.makedirs(PROFILE_OUTPUT_DIR, exist_ok=True)

    # 加载模型和分词器
    # 注意：对于70B模型，这一步可能非常耗时且需要大量内存
    try:
        model, tokenizer = setup_model_and_tokenizer(MODEL_NAME, DTYPE, DEVICE)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have enough VRAM/RAM, network access to Hugging Face, and correct model name.")
        print("For LLaMA models, you might need to log in via `huggingface-cli login` and accept license terms.")
        return

    # --- Eager Mode Profiling ---
    # `model` is already in eager mode by default
    run_mode_profile(model, tokenizer, "Eager", PROMPT, PROFILE_OUTPUT_DIR)

    # 清理，准备编译模式
    # 对于非常大的模型，可能需要重新加载模型以确保干净的编译环境，
    # 但这里我们尝试直接编译。
    if DEVICE == "xpu": torch.xpu.empty_cache()
    gc.collect()

    # --- Torch.compile Mode Profiling ---
    print(f"\n{'='*20} Compiling model with torch.compile (IPEX backend) {'='*20}")
    print("Compilation can take a significant amount of time...")
    
    # 确保模型在目标设备上再编译
    if model.device.type != DEVICE:
        model = model.to(DEVICE)

    # 使用 IPEX 作为 torch.compile 的后端
    # `torch.compile`会返回一个新的、优化过的模型对象
    try:
        # IPEX backend options can be passed via `options` argument if needed.
        # e.g., options={"ipex_optimize": True} or specific optimization flags.
        # For now, default IPEX backend behavior is usually good.
        compiled_model = torch.compile(model)
        # compiled_model = torch.compile(model, backend="ipex", mode="max-autotune") # Alternative mode
        print("Model compilation finished.")
    except Exception as e:
        print(f"Error during torch.compile: {e}")
        print("torch.compile with IPEX backend might have specific requirements or limitations.")
        return

    run_mode_profile(compiled_model, tokenizer, "TorchCompile", PROMPT, PROFILE_OUTPUT_DIR)

    print(f"\n{'='*20} Profiling Complete {'='*20}")
    print(f"All profile data saved in: {PROFILE_OUTPUT_DIR}")
    print("You can view the .json traces using Perfetto UI (chrome://tracing) or import into TensorBoard.")

if __name__ == "__main__":
    main()