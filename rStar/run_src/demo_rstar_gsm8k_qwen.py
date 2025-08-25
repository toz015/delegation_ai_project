#!/usr/bin/env python3
"""
Fixed version of the GSM8K Llama 3 processing script
This resolves the import issues that were preventing the notebook from running
"""

from argparse import Namespace
import sys
import os
import math

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

from common.arguments import post_process_args
from run_src import do_generate

# Configure arguments for GSM8K with alternative model
args = Namespace()
args.dataset_name = "GSM8K"
args.model_ckpt = "Qwen/Qwen2-7B-Instruct"  # Set this first

# ====== MODEL CONFIGURATION ======
# Choose your model here:
#args.model_ckpt = "microsoft/Phi-3-mini-4k-instruct"  # Change this to your desired model
# Other popular options:
# args.model_ckpt = "microsoft/Phi-3-mini-4k-instruct"  # Previous model
# args.model_ckpt = "mistralai/Mistral-7B-Instruct-v0.2"
# args.model_ckpt = "meta-llama/Llama-2-7b-chat-hf"

args.hf_token = "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"
args.note = "rStar_GSM8K_Qwen2_7B_Instruct"
args.test_json_filename = "test_all"

# API & Generation settings
args.api = "vllm"
args.temperature = 0.8
args.top_k = 40
args.top_p = 0.95
args.max_tokens = 256
args.seed = 42

# MCTS settings
args.num_rollouts = 16
args.max_depth_allowed = 5
args.num_a1_steps = None
args.num_subquestions = 2
args.num_votes = 1
args.disable_a1 = False
# Mistral args.disable_a5 = False

args.disable_a5 = True
args.mcts_discount_factor = 1.0
args.mcts_exploration_weight = 2.0
args.mcts_weight_scheduler = "const"
args.mcts_num_last_votes = 16

# Node limits
args.max_nodes_a1_a3 = 5
args.max_nodes_others = 1

# ====== Paths ======
args.data_root = "data"
args.prompts_root = "prompts"
args.answer_sheets_dir = "outputs/answer_sheets_qwen"
args.run_outputs_dir = "outputs/run_outputs_qwen"
args.run_outputs_root = args.run_outputs_dir
args.eval_outputs_root = args.run_outputs_dir
args.start_idx = 100
#args.end_idx = 1318
args.end_idx = 300

# Prompt templates
args.decompose_template_path = os.path.join(args.prompts_root, "GSM8K", "decompose", "decompose_template.json")
args.decompose_prompt_path = os.path.join(args.prompts_root, "GSM8K", "decompose", "decompose_prompt.txt")
args.decompose_prompt_rephrased_path = os.path.join(args.prompts_root, "GSM8K", "decompose", "decompose_prompt_rephrased.txt")
args.fewshot_cot_prompt_path = os.path.join(args.prompts_root, "GSM8K", "fewshot_cot", "fewshot_cot_prompt.txt")
args.fewshot_cot_config_path = os.path.join(args.prompts_root, "GSM8K", "fewshot_cot", "fewshot_cot_config.json")
args.fewshot_ost_prompt_path = os.path.join(args.prompts_root, "GSM8K", "fewshot_ost", "fewshot_ost_prompt.txt")
args.fewshot_ost_config_path = os.path.join(args.prompts_root, "GSM8K", "fewshot_ost", "fewshot_ost_config.json")
args.rephrasing_prompt_template_path = os.path.join(args.prompts_root, "GSM8K", "rephrasing_prompt_template.txt")
args.fewshot_cot_prompt_rephrased_path = args.fewshot_cot_prompt_path
args.fewshot_ost_prompt_rephrased_path = args.fewshot_ost_prompt_path

# Precision / Parallelism
args.model_parallel = False
args.tensor_parallel_size = 1
args.half_precision = True
args.verbose = True

# Flags
args.modify_prompts_for_rephrasing = False
args.enable_potential_score = False
args.save_tree = False

# Finalize and run
print("🔧 Processing arguments...")
args = post_process_args(args)
print("✅ Configuration complete!")
print(f"📊 Processing {args.end_idx - args.start_idx} GSM8K problems")
print(f"🎯 Model: {args.model_ckpt}")
print(f"🔧 API: {args.api}")
print(f"🔄 Reasoning path: A5 (rephrasing) DISABLED")
print(f"📁 Output directory: {args.answer_sheets_dir}")

# Run the generation with proper multiprocessing guard
if __name__ == '__main__':
    do_generate.main(args)
    print("✅ GSM8K processing completed!") 