import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.profiler
import os
import gc
import time

# --- 配置参数 ---
# MODEL_NAME = "meta-llama/Meta-Llama-3.1-70B-Instruct" # 实际分析时使用
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B" # 测试用
DEVICE = "xpu" if torch.xpu.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE != "cpu" else torch.float32 # CPU上bfloat16支持可能不佳

PROMPT = "Once upon a time, in a land far, far away,"
MAX_TOKENS_FOR_FIRST = 1
MAX_TOKENS_FOR_REST = 10
NUM_WARMUP_RUNS = 2
DEFAULT_PROFILE_OUTPUT_DIR = f"logs/profile_output_{MODEL_NAME.split('/')[-1]}"

class LLMProfiler:
    def __init__(self, model_name, dtype, device, prompt,
                 max_tokens_first, max_tokens_rest, num_warmup, profile_base_dir):
        self.model_name = model_name
        self.dtype = dtype
        self.device = device
        self.prompt = prompt
        self.max_tokens_first = max_tokens_first
        self.max_tokens_rest = max_tokens_rest
        self.num_warmup = num_warmup
        self.profile_base_dir = profile_base_dir

        os.makedirs(self.profile_base_dir, exist_ok=True)

        self.tokenizer = None
        self.model = None # Eager model
        self.compiled_model = None

        self._setup_model_and_tokenizer()

    def _setup_model_and_tokenizer(self):
        print(f"Loading tokenizer for {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print(f"Loading model {self.model_name} to {self.device} with dtype {self.dtype}...")
        # Note: low_cpu_mem_usage is more relevant for very large models during from_pretrained
        # For 8B, it might not be strictly necessary but doesn't hurt.
        use_low_cpu_mem = "70B" in self.model_name # Heuristic
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=use_low_cpu_mem 
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Eager model and tokenizer loaded.")

    def _cleanup_memory(self):
        gc.collect()
        if self.device == "xpu":
            torch.xpu.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

    def _profile_generation_step(
        self,
        model_to_profile,
        inputs,
        max_new_tokens,
        profile_name,
        profile_dir_step,
        use_cache=True,
        past_key_values=None
    ):
        print(f"\nProfiling: {profile_name}, Max new tokens: {max_new_tokens}")
        os.makedirs(profile_dir_step, exist_ok=True)

        current_input_ids = inputs["input_ids"].to(model_to_profile.device)
        current_attention_mask = inputs["attention_mask"].to(model_to_profile.device)

        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.device == "xpu":
            activities.append(torch.profiler.ProfilerActivity.XPU)
        elif self.device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        
        schedule = torch.profiler.schedule(wait=0, warmup=0, active=1)

        with torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            record_shapes=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir_step, worker_name=profile_name)
        ) as prof:
            with torch.no_grad():
                with torch.profiler.record_function(f"generate_tokens_{profile_name}"):
                    outputs = model_to_profile.generate(
                        current_input_ids,
                        attention_mask=current_attention_mask,
                        max_new_tokens=max_new_tokens,
                        use_cache=use_cache,
                        pad_token_id=self.tokenizer.eos_token_id,
                        past_key_values=past_key_values,
                        return_dict_in_generate=True
                    )
        
        print(f"Profiling data for '{profile_name}' saved.")
        # The tensorboard_trace_handler creates a .pt.trace.json file
        # The exact name might vary slightly based on profiler version or if worker_name is complex
        # Let's try to find it or construct a likely name
        trace_file_name = f"{profile_name.replace(' ', '_')}.pt.trace.json" # Common pattern
        if not os.path.exists(os.path.join(profile_dir_step, trace_file_name)):
             # Try finding the most recent .json file if the naming is unexpected
            json_files = [f for f in os.listdir(profile_dir_step) if f.endswith('.json')]
            if json_files:
                trace_file_name = sorted(json_files, key=lambda f: os.path.getmtime(os.path.join(profile_dir_step, f)), reverse=True)[0]
        print(f"Trace (for Perfetto/TensorBoard): {os.path.join(profile_dir_step, trace_file_name)}")

        print(f"\n--- Top {self.device.upper()} time ops for {profile_name} (grouped by input shape) ---")
        try:
            sort_key = f"self_{self.device}_time_total" if self.device in ["xpu", "cuda"] else "self_cpu_time_total"
            if not any(getattr(e, sort_key, 0) > 0 for e in prof.key_averages()):
                sort_key = f"{self.device}_time_total" if self.device in ["xpu", "cuda"] else "cpu_time_total"
            if not any(getattr(e, sort_key, 0) > 0 for e in prof.key_averages()):
                 sort_key = "self_cpu_time_total"

            print(prof.key_averages(group_by_input_shape=True).table(sort_by=sort_key, row_limit=15))
        except Exception as e:
            print(f"Could not print key_averages with preferred sort key: {e}")
            print("Printing raw key_averages (sorted by self_cpu_time_total):")
            print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))
        
        self._cleanup_memory()
        # We don't return outputs from profiling step to keep it focused.
        # KV cache and next inputs are handled by the calling method.

    def _warmup_run(self, model_to_warmup, input_ids, attention_mask, max_new_tokens, past_key_values=None):
        with torch.no_grad():
            for _ in range(self.num_warmup):
                _ = model_to_warmup.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    past_key_values=past_key_values,
                    return_dict_in_generate=True
                )
                if self.device == "xpu": torch.xpu.synchronize()
                elif self.device == "cuda": torch.cuda.synchronize()
        self._cleanup_memory()

    def _get_kv_cache_and_next_inputs(self, model_for_kv, input_ids_prompt, attention_mask_prompt):
        """Generates one token to get past_key_values and prepares inputs for the next step."""
        with torch.no_grad():
            outputs_after_first = model_for_kv.generate(
                input_ids_prompt,
                attention_mask=attention_mask_prompt,
                max_new_tokens=self.max_tokens_first, # Generate one token
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True
            )
        
        past_key_values = outputs_after_first.past_key_values
        # Input for rest phase is the last token generated
        next_input_ids = outputs_after_first.sequences[:, -1:]
        
        # Attention mask for rest phase needs to be extended
        ones_for_new_token = torch.ones((next_input_ids.shape[0], 1), dtype=torch.long, device=model_for_kv.device)
        next_attention_mask = torch.cat([attention_mask_prompt, ones_for_new_token], dim=1)
        
        return past_key_values, next_input_ids, next_attention_mask

    def _run_profile_for_model_variant(self, model_to_profile, mode_name: str):
        print(f"\n{'='*20} Profiling {mode_name} Mode {'='*20}")
        
        inputs_prompt = self.tokenizer(self.prompt, return_tensors="pt")
        input_ids_prompt = inputs_prompt["input_ids"].to(model_to_profile.device)
        attention_mask_prompt = inputs_prompt["attention_mask"].to(model_to_profile.device)

        # --- 1. First Token Profiling ---
        profile_name_first = f"{mode_name} First Token"
        profile_dir_first = os.path.join(self.profile_base_dir, mode_name.lower().replace(' ', '_'), "first_token")
        
        print(f"Warmup for {profile_name_first}...")
        self._warmup_run(model_to_profile, input_ids_prompt, attention_mask_prompt, self.max_tokens_first)
        
        first_token_inputs = {"input_ids": input_ids_prompt, "attention_mask": attention_mask_prompt}
        self._profile_generation_step(
            model_to_profile, first_token_inputs, self.max_tokens_first,
            profile_name_first, profile_dir_first, use_cache=True
        )
        time.sleep(1) # Small pause

        # --- Get KV Cache and inputs for Rest Tokens phase ---
        # This generation is NOT profiled, it's just to setup the next stage
        past_key_values_from_first, rest_phase_input_ids, rest_phase_attention_mask = \
            self._get_kv_cache_and_next_inputs(model_to_profile, input_ids_prompt, attention_mask_prompt)
        
        self._cleanup_memory()
        time.sleep(1)

        # --- 2. Rest Tokens Profiling ---
        profile_name_rest = f"{mode_name} Rest Tokens"
        profile_dir_rest = os.path.join(self.profile_base_dir, mode_name.lower().replace(' ', '_'), "rest_tokens")

        print(f"Warmup for {profile_name_rest}...")
        self._warmup_run(
            model_to_profile, rest_phase_input_ids, rest_phase_attention_mask,
            self.max_tokens_rest, past_key_values=past_key_values_from_first
        )

        rest_tokens_inputs = {"input_ids": rest_phase_input_ids, "attention_mask": rest_phase_attention_mask}
        self._profile_generation_step(
            model_to_profile, rest_tokens_inputs, self.max_tokens_rest,
            profile_name_rest, profile_dir_rest, use_cache=True,
            past_key_values=past_key_values_from_first
        )
        time.sleep(1)

    def run_all_profiles(self):
        print(f"Using device: {self.device}")
        if self.device == "xpu":
            print(f"XPU Name: {torch.xpu.get_device_name(0)}")
        elif self.device == "cuda":
            print(f"CUDA Name: {torch.cuda.get_device_name(0)}")

        # --- Eager Mode Profiling ---
        if self.model is None:
            print("Eager model not loaded. Skipping eager profile.")
        else:
            self._run_profile_for_model_variant(self.model, "Eager")

        self._cleanup_memory()

        # --- Torch.compile Mode Profiling ---
        if self.device == "cpu" and not hasattr(torch, '_dynamo'): # Dynamo needed for compile on CPU
            print("torch.compile on CPU requires a newer PyTorch version with Dynamo. Skipping compile profile.")
        else:
            print(f"\n{'='*20} Compiling model with torch.compile {'='*20}")
            print("Compilation can take a significant amount of time...")
            
            # Ensure model is on the target device before compiling
            # Model should already be on device from _setup_model_and_tokenizer, but to be safe:
            model_to_compile = self.model.to(self.device) 
            
            compile_backend = None
            if self.device == "xpu":
                print("Using default backend (likely Inductor) for torch.compile on XPU.")
            elif self.device == "cuda":
                # Default backend for CUDA is usually good (inductor)
                print("Using default backend (likely Inductor) for torch.compile on CUDA.")
            else: # CPU
                print("Using default backend (likely Inductor) for torch.compile on CPU.")


            try:
                # mode="max-autotune" can provide better performance but takes longer to compile
                # mode="reduce-overhead" for faster compilation
                # default mode is often a good balance
                compile_options = {}
                
                self.compiled_model = torch.compile(model_to_compile, **compile_options)
                print("Model compilation finished.")
                self._run_profile_for_model_variant(self.compiled_model, "TorchCompile")
            except Exception as e:
                print(f"Error during torch.compile or its profiling: {e}")
                print("torch.compile might have specific requirements or limitations.")

        print(f"\n{'='*20} Profiling Complete {'='*20}")
        print(f"All profile data saved in: {self.profile_base_dir}")
        print("You can view the .json traces using Perfetto UI (chrome://tracing) or import into TensorBoard.")

def main():
    profiler = LLMProfiler(
        model_name=MODEL_NAME,
        dtype=DTYPE,
        device=DEVICE,
        prompt=PROMPT,
        max_tokens_first=MAX_TOKENS_FOR_FIRST,
        max_tokens_rest=MAX_TOKENS_FOR_REST,
        num_warmup=NUM_WARMUP_RUNS,
        profile_base_dir=DEFAULT_PROFILE_OUTPUT_DIR
    )
    try:
        profiler.run_all_profiles()
    except Exception as e:
        print(f"An critical error occurred during profiling: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure you have enough VRAM/RAM, network access to Hugging Face, and correct model name.")
        print("For LLaMA models, you might need to log in via `huggingface-cli login` and accept license terms.")

if __name__ == "__main__":
    main()