#!/usr/bin/env python3
"""
Mistral model configuration for STG processing
- Uses Mistral-7B-Instruct-v0.2 model
- Enables A5 (rephrasing) for enhanced reasoning
- Processes STG questions 0-100 (100 questions)
"""

from argparse import Namespace
import sys
import os
import math

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

from common.arguments import post_process_args
from run_src import do_generate

# Configure arguments for STG with Mistral model
args = Namespace()
args.dataset_name = "STG"  # Changed from GSM8K to STG

# ====== MODEL CONFIGURATION ======
args.model_ckpt = "mistralai/Mistral-7B-Instruct-v0.2"  # Mistral model
args.hf_token = "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"
args.note = "rStar_STG_Mistral7B_with_rephrasing_0_100"  # Updated note
args.test_json_filename = "test_all"  # STG uses test_all.json

# ====== REASONING PATH CONFIGURATION ======
# A5 (Rephrasing) is ENABLED for enhanced reasoning path
# This means the model will work with both original and rephrased questions
args.disable_a5 = False  # Rephrasing ENABLED - uses both original and rephrased questions

# ====== API & Generation settings ======
args.api = "vllm"
args.temperature = 0.8
args.top_k = 40
args.top_p = 0.95
args.max_tokens = 256
args.seed = 42

# ====== vLLM Memory Settings ======
args.max_model_len = 2048  # Reduced from 4096 to save memory
args.gpu_memory_utilization = 0.35  # Reduced from 0.6 to fit available memory (~7.7 GiB)

# ====== MCTS settings ======
args.num_rollouts = 16  # Balanced configuration for speed and quality
args.max_depth_allowed = 5
args.num_a1_steps = None
args.num_subquestions = 2
args.num_votes = 1
args.disable_a1 = False  # Keep A1 (one-step thoughts) enabled
args.mcts_discount_factor = 1.0
args.mcts_exploration_weight = 2.0
args.mcts_weight_scheduler = "const"
args.mcts_num_last_votes = 16

# ====== Node limits ======
args.max_nodes_a1_a3 = 5
args.max_nodes_others = 1

# ====== Paths ======
args.data_root = "data"
args.prompts_root = "prompts"
args.answer_sheets_dir = "outputs/answer_sheets_mistral_stg"  # Updated for STG
args.run_outputs_dir = "outputs/run_outputs_mistral_stg"  # Updated for STG
args.run_outputs_root = args.run_outputs_dir
args.eval_outputs_root = args.run_outputs_dir
args.start_idx = 41 # Start from question 0 (first 100 problems)
args.end_idx = 100  # Process questions 0-99 (100 questions total)

# ====== Prompt templates ======
# Updated all paths to use STG instead of GSM8K
args.decompose_template_path = os.path.join(args.prompts_root, "STG", "decompose", "decompose_template.json")
args.decompose_prompt_path = os.path.join(args.prompts_root, "STG", "decompose", "decompose_prompt.txt")
args.decompose_prompt_rephrased_path = os.path.join(args.prompts_root, "STG", "decompose", "decompose_prompt_rephrased.txt")
args.fewshot_cot_prompt_path = os.path.join(args.prompts_root, "STG", "fewshot_cot", "fewshot_cot_prompt.txt")
args.fewshot_cot_config_path = os.path.join(args.prompts_root, "STG", "fewshot_cot", "fewshot_cot_config.json")
args.fewshot_ost_prompt_path = os.path.join(args.prompts_root, "STG", "fewshot_ost", "fewshot_ost_prompt.txt")
args.fewshot_ost_config_path = os.path.join(args.prompts_root, "STG", "fewshot_ost", "fewshot_ost_config.json")
args.rephrasing_prompt_template_path = os.path.join(args.prompts_root, "STG", "rephrasing_prompt_template.txt")
args.fewshot_cot_prompt_rephrased_path = args.fewshot_cot_prompt_path
args.fewshot_ost_prompt_rephrased_path = args.fewshot_ost_prompt_path

# ====== Precision / Parallelism ======
args.model_parallel = False
args.tensor_parallel_size = 1
args.half_precision = True
args.verbose = False

# ====== Flags ======
args.modify_prompts_for_rephrasing = False
args.enable_potential_score = False
args.save_tree = False

# ====== Finalize and run ======
print("🔧 Processing arguments...")
args = post_process_args(args)
print("✅ Configuration complete!")
print(f"📊 Processing {args.end_idx - args.start_idx} STG problems (questions {args.start_idx}-{args.end_idx-1})")
print(f"🎯 Model: {args.model_ckpt}")
print(f"🔧 API: {args.api}")
print(f"🔄 Reasoning path: A5 (rephrasing) ENABLED")
print(f"📁 Output directory: {args.answer_sheets_dir}")
print(f"📚 Dataset: STG (True/False questions)")

# Run the generation with proper multiprocessing guard
if __name__ == '__main__':
    do_generate.main(args)
    print("✅ STG processing completed!")
