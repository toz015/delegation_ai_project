#!/usr/bin/env python3
"""
Extract LLM answers using discriminator logic for MATH dataset
This script uses the same logic as do_discriminate.py to extract answers and confidence scores
from multiple LLM models on the MATH dataset
"""

import os
import json
import sys
from itertools import combinations
from collections import Counter, defaultdict
from typing import Dict, List, Any

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.utils import read_json
from eval_src.Evaluator import MATHEvaluator
from run_src.rstar_utils import concat_solution_trace


class Candidate:
    def __init__(
        self,
        solution_trace,
        final_step,
        final_answer,
        id,
        freq=1,
        trace_reward=1.0,
    ):
        self.solution_trace = solution_trace
        self.final_step = final_step
        self.final_answer = final_answer
        self.id = id
        self.freq = freq
        self.trace_reward = trace_reward


def group_candidates_by_answer(candidates: list[Candidate], evaluator, criteria="freq"):
    """Return answer2candidates, answer2confidence, answer2cnt."""
    answer2candidates = {}
    answer2confidence = defaultdict(float)
    answer2cnt = defaultdict(int)

    for c in candidates:
        has_existed = False
        for existing_answer in answer2candidates.keys():
            if evaluator.check_answers_equiv(c.final_answer, existing_answer):
                has_existed = True
                answer2candidates[str(existing_answer)].extend([c] * c.freq)
                answer2confidence[str(existing_answer)] += c.trace_reward if criteria == "reward" else c.freq
                answer2cnt[str(existing_answer)] += c.freq
                break

        if not has_existed:
            if str(c.final_answer) in answer2candidates:
                answer2candidates[str(c.final_answer)].extend([c] * c.freq)
            else:
                answer2candidates[str(c.final_answer)] = [c] * c.freq
            answer2confidence[str(c.final_answer)] += c.trace_reward if criteria == "reward" else c.freq
            answer2cnt[str(c.final_answer)] += c.freq

    # Normalize confidence scores
    candidates_count = sum([candidate.freq for candidate in candidates])
    for ans in answer2confidence.keys():
        answer2confidence[ans] /= candidates_count

    return answer2candidates, answer2confidence, answer2cnt


def extract_answers_from_generator_discriminator(output_dir: str, question_range: tuple) -> Dict[int, Dict[str, Any]]:
    """
    Extract answers from a single generator's output directory using discriminator logic
    
    Args:
        output_dir: Path to the generator's output directory
        question_range: Tuple of (start_idx, end_idx)
    
    Returns:
        Dictionary mapping question index to answer data
    """
    start_idx, end_idx = question_range
    results = {}
    
    # Find the answer sheets directory
    answer_sheets_dir = None
    for root, dirs, files in os.walk(output_dir):
        if "answer_sheets" in dirs:
            answer_sheets_dir = os.path.join(root, "answer_sheets")
            break
    
    if not answer_sheets_dir or not os.path.exists(answer_sheets_dir):
        print(f"⚠️  No answer_sheets directory found in {output_dir}")
        return results
    
    print(f"📁 Processing {answer_sheets_dir}")
    
    # Initialize evaluator
    evaluator = MATHEvaluator()
    
    # Process each question in the range
    for question_idx in range(start_idx, end_idx):
        question_file = os.path.join(answer_sheets_dir, f"Question {question_idx:04d} - Answer.json")
        
        if not os.path.exists(question_file):
            print(f"⚠️  Question {question_idx} file not found: {question_file}")
            continue
        
        try:
            # Read the answer file
            answer_data = read_json(question_file)
            
            # Find the corresponding trace files
            final_solutions_file = os.path.join(answer_sheets_dir, f"Question {question_idx:04d} - Final Solutions.json")
            rollout_solutions_file = os.path.join(answer_sheets_dir, f"Question {question_idx:04d} - Rollout Solutions.json")
            
            trace_data = []
            if os.path.exists(final_solutions_file):
                trace_data.extend(read_json(final_solutions_file))
            if os.path.exists(rollout_solutions_file):
                trace_data.extend(read_json(rollout_solutions_file))
            
            if not trace_data:
                print(f"⚠️  No trace data found for question {question_idx}")
                continue
            
            # Process trace data using discriminator logic
            all_candidates = []
            solution_trace_dic = {}
            
            for id, s in enumerate(trace_data):
                trace = s["trace"] if "trace" in s else s
                solution_trace, final_step, _, reward = concat_solution_trace(trace)
                
                if solution_trace in solution_trace_dic:
                    solution_trace_dic[solution_trace]["freq"] = solution_trace_dic[solution_trace]["freq"] + 1
                    solution_trace_dic[solution_trace]["reward"] = (
                        solution_trace_dic[solution_trace]["reward"] + reward
                    )
                    if len(solution_trace_dic[solution_trace]["final_step"]) < len(final_step):
                        solution_trace_dic[solution_trace]["final_step"] = final_step
                else:
                    solution_trace_dic[solution_trace] = {"freq": 1, "reward": reward, "final_step": final_step}
            
            # Create candidates
            for solution_trace in solution_trace_dic.keys():
                final_step = solution_trace_dic[solution_trace]["final_step"]
                trace_freq = solution_trace_dic[solution_trace]["freq"]
                trace_reward = solution_trace_dic[solution_trace]["reward"]
                
                final_answer = evaluator.extract_answer_from_model_completion(final_step)
                
                candidate = Candidate(
                    solution_trace,
                    final_step,
                    final_answer,
                    id,
                    trace_freq,
                    trace_reward,
                )
                all_candidates.append(candidate)
            
            # Group candidates by answer and calculate confidence
            answer2candidates, answer2confidence, answer2cnt = group_candidates_by_answer(
                all_candidates, evaluator, "freq"
            )
            
            # Get most confident answer (filter out None/None answers)
            if answer2confidence:
                # Filter out None answers and get the most confident valid answer
                valid_answers = {k: v for k, v in answer2confidence.items() if k != "None" and k is not None}
                if valid_answers:
                    most_confident_answer = max(valid_answers.keys(), key=lambda x: valid_answers[x])
                    highest_confidence = valid_answers[most_confident_answer]
                else:
                    most_confident_answer = "NO_VALID_ANSWER"
                    highest_confidence = 0.0
            else:
                most_confident_answer = "NO_ANSWER"
                highest_confidence = 0.0
            
            # Store results
            # Filter out None answers from all answer-related fields
            filtered_answers = [ans for ans in answer2candidates.keys() if ans != "None" and ans is not None]
            filtered_answer_counts = {k: v for k, v in answer2cnt.items() if k != "None" and k is not None}
            filtered_answer_confidence = {k: v for k, v in answer2confidence.items() if k != "None" and k is not None}
            
            # Create answer_to_solution_path mapping
            answer_to_solution_path = {}
            for answer in filtered_answers:
                # Find the first candidate that produced this answer
                for candidate in all_candidates:
                    if str(candidate.final_answer) == answer:
                        answer_to_solution_path[answer] = candidate.solution_trace
                        break
            
            results[question_idx] = {
                "answers": filtered_answers,
                "answer_counts": filtered_answer_counts,
                "answer_confidence": filtered_answer_confidence,
                "answer_to_solution_path": answer_to_solution_path,  # New field: maps answers to solution paths
                "most_frequent_answer": most_confident_answer,
                "highest_confidence": highest_confidence,
                "total_rollouts": len(all_candidates),
                "unique_answers": len(filtered_answers),
                "golden_answer": answer_data.get("gold_answer")
            }
            
            print(f"✅ Question {question_idx}: {len(all_candidates)} rollouts, "
                  f"most confident: {most_confident_answer[:50]}... (confidence: {highest_confidence:.3f})")
            
        except Exception as e:
            print(f"❌ Error processing question {question_idx}: {e}")
            continue
    
    return results


def process_model_combination(selected_models, all_llm_generators, question_range):
    """Process a specific combination of 3 models."""
    print(f"\n🎯 Processing combination: {', '.join(selected_models)}")
    
    llm_generators = {model: all_llm_generators[model] for model in selected_models}
    
    # Extract answers from each generator
    all_results = {}
    
    for llm_name, output_dir in llm_generators.items():
        print(f"\n{'='*60}")
        print(f"Processing {llm_name}")
        print(f"{'='*60}")
        
        if not os.path.exists(output_dir):
            print(f"⚠️  Output directory not found: {output_dir}")
            continue
        
        results = extract_answers_from_generator_discriminator(output_dir, question_range)
        all_results[llm_name] = results
        
        print(f"✅ Extracted {len(results)} questions from {llm_name}")
    
    # Create output data structure
    output_data = {
        "metadata": {
            "llm_generators": selected_models,
            "available_models": list(all_llm_generators.keys()),
            "selected_count": len(selected_models),
            "question_range": question_range,
            "total_questions": len(all_results[list(all_results.keys())[0]]) if all_results else 0,
            "extraction_method": "discriminator_logic",
            "dataset": "MATH",
            "timestamp": "2025-01-09"
        },
        "results": all_results
    }
    
    # Save results
    model_names = "_".join([name.replace("-", "_") for name in selected_models])
    output_filename = f"llm_answers_discriminator_MATH_{question_range[0]}_{question_range[1]-1}_{model_names}.json"
    with open(output_filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n🎉 Extraction complete! Results saved to {output_filename}")
    print(f"📊 Total questions processed: {output_data['metadata']['total_questions']}")
    
    # Show summary
    for llm_name, results in all_results.items():
        total_questions = len(results)
        correct_answers = sum(1 for q_data in results.values() 
                            if q_data.get('golden_answer') and 
                            q_data.get('most_frequent_answer') and
                            q_data.get('most_frequent_answer') != "NO_VALID_ANSWER" and
                            q_data.get('most_frequent_answer') != "NO_ANSWER" and
                            str(q_data.get('most_frequent_answer')).strip() == str(q_data.get('golden_answer')).strip())
        accuracy = correct_answers / total_questions if total_questions > 0 else 0
        
        print(f"{llm_name}: {correct_answers}/{total_questions} correct ({accuracy:.1%})")
    
    return output_filename


def main():
    """Main function to extract answers from all possible combinations of 3 models."""
    print("🚀 Starting LLM Answer Extraction using Discriminator Logic...")
    print("📋 Generating all possible combinations of 3 models from 5 available models")
    
    # Define all available LLM generators and their output directories
    all_llm_generators = {
        "Falcon-7B": "/home/wanghd/restore/delegation_ai_project/rStar/outputs/organized/MATH/Falcon-7B",
        "Zephyr-7B-Beta": "/home/wanghd/restore/delegation_ai_project/rStar/outputs/organized/MATH/Zephyr-7B-Beta",
        "Mistral-7B-Instruct-v0.2": "/home/wanghd/restore/delegation_ai_project/rStar/outputs/organized/MATH/Mistral-7B-Instruct-v0.2",
        "Qwen2-7B-Instruct": "/home/wanghd/restore/delegation_ai_project/rStar/outputs/organized/MATH/Qwen2-7B-Instruct"
    }
    
    question_range = (0, 299)
    
    # Generate all possible combinations of 3 models from 5 available models
    available_models = list(all_llm_generators.keys())
    model_combinations = list(combinations(available_models, 3))
    
    print(f"📊 Found {len(model_combinations)} possible combinations of 3 models:")
    for i, combo in enumerate(model_combinations, 1):
        print(f"  {i}. {', '.join(combo)}")
    
    # Process each combination
    generated_files = []
    for i, selected_models in enumerate(model_combinations, 1):
        print(f"\n{'#'*80}")
        print(f"PROCESSING COMBINATION {i}/{len(model_combinations)}")
        print(f"{'#'*80}")
        
        output_file = process_model_combination(selected_models, all_llm_generators, question_range)
        generated_files.append(output_file)
    
    print(f"\n🎉 All combinations processed!")
    print(f"📁 Generated {len(generated_files)} JSON files:")
    for file in generated_files:
        print(f"  - {file}")


if __name__ == "__main__":
    main() 