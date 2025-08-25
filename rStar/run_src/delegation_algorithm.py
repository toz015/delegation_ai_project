#!/usr/bin/env python3
"""
Delegation Algorithm Implementation

This script implements a collaborative learning algorithm where multiple LLM generators
work together through a discriminator to improve their answer confidence scores.

Algorithm Overview:
1. For each question, each generator selects their best answer and solution path
2. All answers/paths are processed using the existing discriminator logic
3. Survival rates are calculated using the proper _calculate_scores method
4. Scores are normalized to sum to 0
5. Generators update their confidence based on normalized survival rates
6. Process repeats for multiple iterations (learning rate)
7. Final answers are determined based on updated confidences
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import copy
import random
import os
import sys
import math
# Add the parent directory to the path to import common.utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import fix_seeds, read_json, read_txt
# Add the current directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import discriminator logic and evaluator
try:
    from do_discriminate import group_candidates_by_answer, Discriminator, Candidate, MajorityVoteDiscriminator
    from rstar_utils import mask_solution_trace
    from eval_src.Evaluator import GSM8KEvaluator  # Import the specific evaluator
    DISCRIMINATOR_AVAILABLE = True
    print("✅ Discriminator components imported successfully")
except ImportError as e:
    print(f"Warning: Some discriminator components not available: {e}")
    DISCRIMINATOR_AVAILABLE = False
    
    # Create dummy classes when imports fail
    class Candidate:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    def group_candidates_by_answer(candidates, evaluator=None, criteria="freq"):
        """Dummy function when discriminator is not available."""
        return {}, {}, {}

class DelegationAlgorithm:
    def __init__(self, learning_rate: float = 0.1, max_iterations: int = 5):
        """
        Initialize the delegation algorithm.
        
        Args:
            learning_rate: How much to update confidence scores in each iteration
            max_iterations: Maximum number of iterations per question
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        
        # Initialize evaluator (you can specify the dataset)
        self.evaluator = None
        if DISCRIMINATOR_AVAILABLE:
            try:
                # Create GSM8K evaluator for math problems
                self.evaluator = GSM8KEvaluator()
                print("✅ GSM8K Evaluator loaded successfully")
            except ImportError:
                print("⚠️  GSM8K Evaluator not available, using fallback")
        
        # Discriminator parameters (matching do_discriminate.py)
        self.num_masked_traces = 4
        self.mask_left_boundary = 0.2
        self.mask_right_boundary = 0.8
        self.rc_n_completions = 2
        
    def load_generator_data(self, data_file: str) -> Dict:
        """Load the generator data from the JSON file."""
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Validate data structure
        if 'results' not in data:
            raise ValueError("Data file must contain 'results' key")
        
        return data
    
    def select_best_answer_for_generator(self, generator_results: Dict, question_idx: int) -> Tuple[str, str, Dict]:
        """
        For a given generator and question, select the best answer, its path, and confidence.
        
        Args:
            generator_results: Results for one generator
            question_idx: Question index
            
        Returns:
            Tuple of (answer, solution_path, confidence_dict)
        """
        question_data = generator_results[str(question_idx)]
        
        # Get confidence scores and use random selection based on weights
        confidence = question_data['answer_confidence']
        weights = [float(i) for i in confidence.values()]
        best_answer = random.choices(list(confidence.keys()), weights=weights, k=1)[0]
        
        # Get the solution path for this answer
        solution_path = question_data['answer_to_solution_path'].get(best_answer, "")
        
        return best_answer, solution_path, confidence
    
    def create_candidates_from_answers(self, all_answers_paths: List[Tuple[str, str, float]], 
                                     question_context: str = "") -> List[Candidate]:
        """
        Create Candidate objects from answer data for proper discriminator processing.
        
        Args:
            all_answers_paths: List of (answer, path, confidence) tuples
            question_context: Context about the question
            
        Returns:
            List of Candidate objects
        """
        if not DISCRIMINATOR_AVAILABLE:
            return []
        
        candidates = []
        for i, (answer, path, confidence) in enumerate(all_answers_paths):
            try:
                # Create masked solution traces using the proper masking function
                masked_traces = mask_solution_trace(
                    path,
                    num_return=self.num_masked_traces,
                    left_boundary=self.mask_left_boundary,
                    right_boundary=self.mask_right_boundary
                )
                
                # Create Candidate object
                candidate = Candidate(
                    solution_trace=path,
                    masked_solution_trace_list=masked_traces,
                    final_step=path.split('\n')[-1] if path else "",
                    final_answer=answer,
                    id=f"delegation_{i}",
                    freq=1,
                    trace_reward=confidence,
                    c_type="delegation"
                )
                candidates.append(candidate)
                
            except Exception as e:
                print(f"Warning: Error creating candidate {i}: {e}")
                continue
        
        return candidates
    
    def calculate_survival_scores_using_discriminator(self, candidates: List[Candidate], 
                                                    question_context: str = "") -> Dict[str, float]:
        """
        Calculate survival scores using the real discriminator logic from _filter_reasoning_consistency.
        
        Args:
            candidates: List of Candidate objects
            question_context: Context about the question
            
        Returns:
            Dictionary mapping answer to survival score (NOT combined with confidence)
        """
        if not DISCRIMINATOR_AVAILABLE or not self.evaluator:
            print("Warning: Using fallback scoring - discriminator not available")
            return self._fallback_scoring(candidates)
        
        try:
            # Create discriminator instance ONCE per question (not per iteration)
            if not hasattr(self, '_current_discriminator'):
                print("  🔍 Creating new discriminator instance...")
                
                # Just use the existing discriminator class directly
                # Create minimal args needed for the discriminator
                import os
                
                # Create temp directory if it doesn't exist
                os.makedirs("./temp_discriminator_results", exist_ok=True)
                
                class SimpleArgs:
                    def __init__(self):
                        self.rc_criteria = "reward"
                        self.rc_mode = "mid"
                        self.num_masked_solution_traces = 4  # Default value
                        self.mask_left_boundary = 0.2
                        self.mask_right_boundary = 0.8
                        self.rc_n_completions = 1  # Default value
                        self.rc_temperature = 0.8
                        self.max_num_seqs = 32
                        self.discriminate_results_dir = "./temp_discriminator_results"
                        self.api = "vllm"  # Switch to vLLM API for better performance
                        self.model_ckpt = "Qwen/Qwen2-7B-Instruct"  # Add model checkpoint
                        self.hf_token ="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"  # Add HF token (None for public models)
                        self.seed = 42  # Add seed for reproducibility
                        
                        # Set dataset and prompts paths
                        self.dataset_name = "GSM8K"
                        self.prompts_root = "./prompts"

                        self.fewshot_config_path = os.path.join(self.prompts_root, self.dataset_name, "fewshot_cot", "fewshot_cot_config.json")
                        self.fewshot_prompt_path = os.path.join(self.prompts_root, self.dataset_name, "fewshot_cot", "fewshot_cot_prompt.txt")


                        # Create the config file
                        self.fewshot_config = read_json(self.fewshot_config_path)
                        
                        self.fewshot_prompt = read_txt(self.fewshot_prompt_path)
                
                # Create discriminator instance
                args = SimpleArgs()
                self._current_discriminator = MajorityVoteDiscriminator(args, self.evaluator)
                print("  🔍 Discriminator created successfully")
            
            # Now use the existing discriminator instance
            discriminator = self._current_discriminator
            
            # Create aux data
            aux = {"problem_id": "delegation_question", "file_idx": 0}
            
            # Apply the real reasoning consistency filtering
            #print("  🔍 Applying real reasoning consistency filtering...")
            # Pass the discriminator's model as gen_model (not None)
            filtered_candidates = discriminator._filter_reasoning_consistency(
                discriminator.model, question_context, candidates, aux
            )
            
            #print(f"  🔍 Real filtering completed: {len(filtered_candidates)} candidates survived")
            return self._calculate_survival_rates_from_filtered(candidates, filtered_candidates)
            
        except Exception as e:
            print(f"  ⚠️  Real discriminator failed: {e}")
            import traceback
            traceback.print_exc()
            # Fall back to simple filtering
            return self._fallback_scoring(candidates)
    
    def _apply_simple_reasoning_consistency_filtering(self, candidates: List[Candidate], 
                                                    question_context: str = "") -> List[Candidate]:
        """
        Apply simplified reasoning consistency filtering that mimics the real discriminator logic.
        This is a fallback when we don't have the full model infrastructure.
        
        Args:
            candidates: List of all candidates
            question_context: Context about the question
            
        Returns:
            List of candidates that passed the filtering
        """
        filtered_candidates = []
        
        for candidate in candidates:
            # Simple filtering logic that mimics what the real discriminator would do
            solution_trace = candidate.solution_trace
            
            # Filter out candidates with poor solution traces
            if not solution_trace or len(solution_trace.strip()) < 20:
                continue
                
            # Filter out candidates with very long traces (likely rambling)
            if len(solution_trace) > 2000:
                continue
                
            # Filter out candidates with no mathematical content
            math_keywords = ['calculate', 'equation', 'formula', 'multiply', 'add', 'subtract', 
                           'divide', 'sum', 'total', 'result', 'answer', 'step', 'solve']
            if not any(keyword in solution_trace.lower() for keyword in math_keywords):
                continue
            
            # If candidate passes all filters, keep it
            filtered_candidates.append(candidate)
        
        #print(f"  🔍 Filtering: {len(candidates)} -> {len(filtered_candidates)} candidates")
        return filtered_candidates
    
    def _calculate_survival_rates_from_filtered(self, unfiltered_candidates: List[Candidate], 
                                               filtered_candidates: List[Candidate]) -> Dict[str, float]:
        """
        Calculate survival rates from filtered candidates using the correct discriminator logic.
        
        Args:
            unfiltered_candidates: All original candidates
            filtered_candidates: Candidates that passed the reasoning consistency filter
            
        Returns:
            Dictionary mapping answer to survival rate
        """
        # Group candidates by answer (both filtered and unfiltered)
        unfiltered_answer2cnt = {}
        filtered_answer2cnt = {}
        
        for candidate in unfiltered_candidates:
            answer = str(candidate.final_answer)
            unfiltered_answer2cnt[answer] = unfiltered_answer2cnt.get(answer, 0) + candidate.freq
        
        for candidate in filtered_candidates:
            answer = str(candidate.final_answer)
            filtered_answer2cnt[answer] = filtered_answer2cnt.get(answer, 0) + candidate.freq
        
        # print(f"  🔍 Unfiltered answers: {unfiltered_answer2cnt}")
        # print(f"  🔍 Filtered answers: {filtered_answer2cnt}")
        
        # Calculate survival rates using the CORRECT discriminator logic
        # This follows the _calculate_scores method exactly
        filtered_answer2survival_rate = {}
        for filtered_ans in filtered_answer2cnt.keys():
            has_existed = False
            for unfiltered_ans in unfiltered_answer2cnt.keys():
                if self.evaluator.check_answers_equiv(filtered_ans, unfiltered_ans):
                    has_existed = True
                    filtered_answer2survival_rate[filtered_ans] = (
                        filtered_answer2cnt[filtered_ans] / unfiltered_answer2cnt[unfiltered_ans]
                    )
                    break
            if not has_existed:
                filtered_answer2survival_rate[filtered_ans] = 0.0
        
        # print(f"  🔍 Survival rates: {filtered_answer2survival_rate}")
        
        # Return ONLY the survival rates (NOT combined with confidence)
        # This is what the discriminator actually provides
        return filtered_answer2survival_rate
    
    def _fallback_scoring(self, candidates: List[Candidate]) -> Dict[str, float]:
        """
        Fallback scoring when discriminator is not available.
        
        Args:
            candidates: List of Candidate objects
            
        Returns:
            Dictionary mapping answer to score
        """
        scores = {}
        for candidate in candidates:
            # Simple scoring based on trace length and content
            score = 0.5  # Base score
            
            if candidate.solution_trace:
                trace_length = len(candidate.solution_trace)
                if trace_length > 50:
                    score += 0.2
                if 'step' in candidate.solution_trace.lower():
                    score += 0.1
                if 'answer' in candidate.solution_trace.lower():
                    score += 0.1
            
            scores[candidate.final_answer] = score
        
        return scores
    
    def discriminator_score_answer_path(self, answer: str, solution_path: str, 
                                      question_context: str = "", 
                                      all_answers_paths: List[Tuple[str, str, float]] = None) -> float:
        """
        Score an answer-solution path using the proper discriminator methodology.
        
        Args:
            answer: The numerical answer
            solution_path: The reasoning path
            question_context: Context about the question
            all_answers_paths: List of all (answer, path, confidence) tuples
            
        Returns:
            Survival rate score between 0 and 1 (NOT combined with confidence)
        """
        if not all_answers_paths:
            return 0.5  # Default score if no comparison data
        
        # Create proper Candidate objects
        candidates = self.create_candidates_from_answers(all_answers_paths, question_context)
        
        if not candidates:
            return 0.5  # Fallback if candidate creation failed
        
        # Calculate survival scores using discriminator logic
        answer2survival_rate = self.calculate_survival_scores_using_discriminator(candidates, question_context)
        
        # Return the survival rate for our specific answer
        if answer in answer2survival_rate:
            return answer2survival_rate[answer]
        else:
            return 0.0  # If answer didn't survive filtering, survival rate is 0
    
    def adaptive_learning_rate(self, iteration: int, base_rate: float, 
                              convergence_threshold: float = 0.01) -> float:
        """
        Calculate adaptive learning rate that decreases over time for better convergence.
        
        Args:
            iteration: Current iteration number
            base_rate: Base learning rate
            convergence_threshold: Threshold for convergence detection
            
        Returns:
            Adaptive learning rate
        """
        # Exponential decay learning rate
        decay_factor = 0.95
        adaptive_rate = base_rate * (decay_factor ** iteration)
        
        # Ensure minimum learning rate
        return max(adaptive_rate, base_rate * 0.1)
    
    def optimize_hyperparameters(self, results: Dict) -> Dict:
        """
        Analyze results to suggest optimal hyperparameters for future runs.
        
        Args:
            results: Delegation results
            
        Returns:
            Dictionary with suggested hyperparameters
        """
        if '_overall_metrics' not in results:
            return {}
        
        metrics = results['_overall_metrics']
        
        suggestions = {
            'learning_rate': self.learning_rate,
            'max_iterations': self.max_iterations,
            'recommendations': []
        }
        
        # Analyze confidence changes
        avg_change = metrics['average_confidence_change']
        if abs(avg_change) < 0.01:
            suggestions['recommendations'].append(
                "Consider increasing learning_rate as confidence changes are very small"
            )
        elif abs(avg_change) > 0.1:
            suggestions['recommendations'].append(
                "Consider decreasing learning_rate as confidence changes are very large"
            )
        
        # Analyze convergence
        total_iterations = 0
        converged_questions = 0
        for question_idx in results.keys():
            if not str(question_idx).startswith('_'):
                question_result = results[question_idx]
                total_iterations += len(question_result['iterations'])
                if len(question_result['iterations']) < self.max_iterations:
                    converged_questions += 1
        
        if converged_questions > 0:
            avg_iterations = total_iterations / len([k for k in results.keys() if not str(k).startswith('_')])
            if avg_iterations < self.max_iterations * 0.7:
                suggestions['recommendations'].append(
                    f"Most questions converged early. Consider reducing max_iterations to {int(avg_iterations * 1.2)}"
                )
            elif avg_iterations >= self.max_iterations:
                suggestions['recommendations'].append(
                    f"Many questions didn't converge. Consider increasing max_iterations to {self.max_iterations + 5}"
                )
        
        return suggestions
    
    def _check_answers_equiv(self, answer1: str, answer2: str) -> bool:
        """
        Simple answer equivalence check.
        In practice, this should use the evaluator's check_answers_equiv method.
        """
        try:
            # Try to convert to float for numerical comparison
            float1 = float(answer1)
            float2 = float(answer2)
            return abs(float1 - float2) < 1e-6
        except (ValueError, TypeError):
            # If not numerical, do string comparison
            return answer1.strip() == answer2.strip()
    
    def normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores so they sum to 0.
        
        Args:
            scores: List of raw scores
            
        Returns:
            List of normalized scores that sum to 0
        """
        if not scores:
            return scores
            
        # Convert to numpy array for easier manipulation
        scores_array = np.array(scores, dtype=float)
        
        # Calculate mean and subtract to center around 0
        mean_score = np.mean(scores_array)
        centered_scores = scores_array - mean_score
        
        return centered_scores.tolist()
    
    def update_generator_confidence(self, 
                                  current_confidence: float, 
                                  cumulative_survival_rate: float,
                                  learning_rate: float) -> float:
        """
        Update a generator's confidence based on the cumulative survival rate.
        
        Args:
            current_confidence: Current confidence score
            cumulative_survival_rate: Cumulative survival rate across all iterations
            learning_rate: Learning rate for updates
            
        Returns:
            Updated confidence score
        """
        # Update confidence using exponential function: exp(learning_rate * cumulative_survival_rate)
    
        confidence_multiplier = math.exp(learning_rate * cumulative_survival_rate)
        
        # Apply exponential update and ensure confidence stays in valid range [0, 1]
        new_confidence = current_confidence * confidence_multiplier
        return max(0.0, min(1.0, new_confidence))
    
    def renormalize_confidence_distribution(self, confidence_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Renormalize a confidence distribution to ensure probabilities sum to 1.0.
        
        Args:
            confidence_dict: Dictionary mapping answers to confidence scores
            
        Returns:
            Renormalized confidence dictionary
        """
        total_confidence = sum(confidence_dict.values())
        
        if total_confidence <= 0:
            # If all confidences are 0 or negative, distribute evenly
            min_val = min(confidence_dict.values())
            shifted = {k: v - min_val if min_val < 0 else v for k, v in confidence_dict.items()}
            total_shifted = sum(shifted.values())
            if total_shifted == 0:
                n = len(confidence_dict)
                return {k: 1.0 / n for k in confidence_dict}
            else:
                return {k: v / total_shifted for k, v in shifted.items()}
        # Renormalize to ensure sum = 1.0
        return {answer: confidence_dict[answer] / total_confidence for answer in confidence_dict}
    
    def get_highest_confidence_answer(self, generator_results: Dict, question_idx: int) -> Tuple[str, str, Dict]:
        """
        Get the answer with highest confidence (deterministic, no randomization).
        
        Args:
            generator_results: Results for one generator
            question_idx: Question index
            
        Returns:
            Tuple of (answer, solution_path, confidence_dict)
        """
        question_data = generator_results[str(question_idx)]
        
        # Get confidence scores and select highest (deterministic)
        confidence = question_data['answer_confidence']
        best_answer = max(confidence.keys(), key=lambda x: confidence[x])
        
        # Get the solution path for this answer
        solution_path = question_data['answer_to_solution_path'].get(best_answer, "")
        
        return best_answer, solution_path, confidence
    
    def run_delegation_for_question(self, 
                                  generators_data: Dict, 
                                  question_idx: int) -> Dict:
        """
        Run the delegation algorithm for a single question.
        
        Args:
            generators_data: Data from all generators
            question_idx: Question index to process
            
        Returns:
            Updated results for this question
        """
        # print(f"🔄 Processing Question {question_idx}...")
        
        # Get all generator names
        generator_names = list(generators_data.keys())
        
        # Initialize results for this question
        question_results = {
            'question_idx': question_idx,
            'iterations': [],
            'final_results': {},
            'golden_answer': None
        }
        
        # Get initial data for each generator
        initial_data = {}
        all_answers_paths = []  # Collect all (answer, path, confidence) tuples for discriminator
        
        for gen_name in generator_names:
            gen_data = generators_data[gen_name]
            if str(question_idx) in gen_data:
                try:
                    answer, path, confidence_dict = self.select_best_answer_for_generator(gen_data, question_idx)
                    
                    # Store golden answer from first generator (assuming it's available)
                    if question_results['golden_answer'] is None:
                        question_results['golden_answer'] = gen_data[str(question_idx)].get('golden_answer', 'Unknown')
                    
                    # Get the confidence for the best answer
                    best_confidence = confidence_dict[answer]
                    initial_data[gen_name] = {
                        'answer': answer,
                        'solution_path': path,
                        'confidence': best_confidence,
                        'original_confidence': best_confidence,
                        'all_confidences': confidence_dict
                    }
                    
                    # Add to the collection for discriminator scoring
                    all_answers_paths.append((answer, path, best_confidence))
                    print(f"==> {gen_name} Initial confidence: {answer} -> {confidence_dict.items()}")
                            
                except Exception as e:
                    print(f"Warning: Error processing generator {gen_name} for question {question_idx}: {e}")
                    continue
        
        if not initial_data:
            print(f"❌ No valid data found for question {question_idx}")
            return question_results
        
        # Initialize cumulative survival rates for each unique answer
        cumulative_survival_rates = defaultdict(float)  # answer -> cumulative normalized survival rate
        normalized_scores = []
        # Run multiple iterations
        for iteration in range(self.max_iterations):
            # print(f"  📊 Iteration {iteration + 1}/{self.max_iterations}")
            
            # Get discriminator scores for all answers
            survival_scores = []
            for gen_name in generator_names:
                if gen_name in initial_data:
                    data = initial_data[gen_name]
                    score = self.discriminator_score_answer_path(
                        data['answer'], 
                        data['solution_path'],
                        all_answers_paths=all_answers_paths
                    )
                    survival_scores.append(score)
                else:
                    survival_scores.append(0.0)
            
            # Normalize scores to sum to 0
            normalized_scores = self.normalize_scores(survival_scores)
            
            # Update cumulative survival rates for each unique answer
            for i, gen_name in enumerate(generator_names):
                if gen_name in initial_data:
                    answer = initial_data[gen_name]['answer']
                    cumulative_survival_rates[answer] += normalized_scores[i]
            
            
            confidence_changes = []
            for i, gen_name in enumerate(generator_names):
                if gen_name in initial_data:
                    old_confidence = initial_data[gen_name]['confidence']
                    answer = initial_data[gen_name]['answer']
                    new_confidence = self.update_generator_confidence(
                        old_confidence,
                        cumulative_survival_rates[answer],  # Use cumulative rate for this answer
                        self.adaptive_learning_rate(iteration, self.learning_rate)
                    )
                    initial_data[gen_name]['confidence'] = new_confidence
                    confidence_changes.append(abs(new_confidence - old_confidence))
                    
                    # Update the confidence in the original confidence dict for renormalization
                    initial_data[gen_name]['all_confidences'][answer] = new_confidence
            
            # Renormalize confidence distributions for each generator
            for gen_name in generator_names:
                if gen_name in initial_data:
                    all_confidences = initial_data[gen_name]['all_confidences']
                    
                    # Renormalize using the helper method
                    renormalized_confidences = self.renormalize_confidence_distribution(all_confidences)
                    
                    # Update the confidence dict with renormalized values
                    all_confidences.clear()
                    all_confidences.update(renormalized_confidences)
                    
                    # Update the selected answer's confidence to match
                    selected_answer = initial_data[gen_name]['answer']
                    initial_data[gen_name]['confidence'] = all_confidences[selected_answer]
                    
            
            # Store iteration results
            iteration_result = {
                'iteration': iteration + 1,
                'survival_scores': dict(zip(generator_names, survival_scores)),
                'normalized_scores': dict(zip(generator_names, normalized_scores)),
                'cumulative_survival_rates': cumulative_survival_rates.copy(),  # Track cumulative rates
                'updated_confidences': {gen: data['confidence'] for gen, data in initial_data.items()}
            }
            question_results['iterations'].append(iteration_result)
            
            # Check for convergence
            # if confidence_changes and max(confidence_changes) < 0.001:  # Convergence threshold
            #     # print(f"  ✅ Converged at iteration {iteration + 1}")
            #     break
            
            # Store current confidence for next iteration comparison
            for gen_name in generator_names:
                if gen_name in initial_data:
                    initial_data[gen_name]['prev_confidence'] = initial_data[gen_name]['confidence']
    

        # Show final confidence summary for this question
        print(f"   📊 Final Confidence Summary:")

        for gen_name in generator_names:
            if gen_name in initial_data:
                print(f"     {gen_name}: {initial_data[gen_name]['confidence']:.3f}")
                print(f"     {gen_name} Full confidence dict: {initial_data[gen_name]['all_confidences']}")
        
        # Show cumulative survival rates with answer mapping
        print(f"   🎯 Cumulative Survival Rates by Answer:")
        for answer, cum_rate in cumulative_survival_rates.items():
            print(f"     Answer '{answer}': {cum_rate:+.4f}")
        
        # Show normalized scores with generator mapping
        print(f"   ⚖️  Final Normalized Scores (sum to 0):")
        for gen_name, norm_score in zip(generator_names, normalized_scores):
            print(f"     {gen_name}: {norm_score:+.4f}")
        
        # Store final results
        question_results['final_results'] = {
            gen_name: {
                'answer': data['answer'],
                'solution_path': data['solution_path'],
                'original_confidence': data['original_confidence'],
                'final_confidence': data['confidence'],
                'confidence_change': data['confidence'] - data['original_confidence'],
                'all_confidences': data['all_confidences']
            }
            for gen_name, data in initial_data.items()
        }
        
        # Calculate delegation metrics
        question_results['delegation_metrics'] = self._calculate_delegation_metrics(
            question_results['final_results'],
            question_results['golden_answer']
        )
        
        return question_results
    
    def _calculate_delegation_metrics(self, final_results: Dict, golden_answer: str) -> Dict:
        """
        Calculate metrics to evaluate the delegation algorithm's performance.
        
        Args:
            final_results: Final results from delegation
            golden_answer: The correct answer
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'total_generators': len(final_results),
            'confidence_improvements': 0,
            'confidence_declines': 0,
            'correct_answers': 0,
            'average_confidence_change': 0.0,
            'best_generator': None,
            'best_confidence': 0.0
        }
        
        total_confidence_change = 0.0
        best_generator = None
        best_confidence = 0.0
        
        for gen_name, result in final_results.items():
            confidence_change = result['confidence_change']
            total_confidence_change += confidence_change
            
            if confidence_change > 0:
                metrics['confidence_improvements'] += 1
            elif confidence_change < 0:
                metrics['confidence_declines'] += 1
            
            # Check if answer is correct
            if self._check_answers_equiv(result['answer'], golden_answer):
                metrics['correct_answers'] += 1
            
            # Track best generator
            if result['final_confidence'] > best_confidence:
                best_confidence = result['final_confidence']
                best_generator = gen_name
        
        metrics['average_confidence_change'] = total_confidence_change / len(final_results)
        metrics['best_generator'] = best_generator
        metrics['best_confidence'] = best_confidence
        
        return metrics
    
    def run_delegation_for_all_questions(self, 
                                       generators_data: Dict, 
                                       question_range: Tuple[int, int] = None) -> Dict:
        """
        Run the delegation algorithm for all questions.
        
        Args:
            generators_data: Data from all generators
            question_range: Optional range of questions to process (start, end)
            
        Returns:
            Complete delegation results
        """
        print("🚀 Starting Delegation Algorithm for All Questions...")
        
        # Determine question range
        if question_range is None:
            # Get range from first generator
            first_gen = list(generators_data.keys())[0]
            questions = list(generators_data[first_gen].keys())
            question_range = (0, len(questions))
        
        start_idx, end_idx = question_range
        print(f"📋 Processing questions {start_idx} to {end_idx-1}")
        
        all_results = {}
        overall_metrics = {
            'total_questions': 0,
            'successful_questions': 0,
            'total_confidence_improvements': 0,
            'total_correct_answers': 0,
            'average_confidence_change': 0.0,
            'generator_performance': defaultdict(lambda: {
                'improvements': 0,
                'declines': 0,
                'correct_answers': 0,
                'total_confidence_change': 0.0
            })
        }
        
        for question_idx in range(start_idx, end_idx):
            try:
                question_result = self.run_delegation_for_question(generators_data, question_idx)
                all_results[question_idx] = question_result
                
                # Update overall metrics
                overall_metrics['total_questions'] += 1
                if question_result['final_results']:
                    overall_metrics['successful_questions'] += 1
                    
                    # Aggregate metrics across generators
                    for gen_name, result in question_result['final_results'].items():
                        gen_metrics = overall_metrics['generator_performance'][gen_name]
                        
                        if result['confidence_change'] > 0:
                            gen_metrics['improvements'] += 1
                            overall_metrics['total_confidence_improvements'] += 1
                        elif result['confidence_change'] < 0:
                            gen_metrics['declines'] += 1
                        
                        gen_metrics['total_confidence_change'] += result['confidence_change']
                        
                        # Check correctness
                        if question_result['delegation_metrics']['correct_answers'] > 0:
                            gen_metrics['correct_answers'] += 1
                            overall_metrics['total_correct_answers'] += 1
                
                print(f"✅ Question {question_idx} completed")
                
            except Exception as e:
                print(f"❌ Error processing question {question_idx}: {e}")
                continue
        
        # Calculate final averages
        if overall_metrics['successful_questions'] > 0:
            total_change = sum(
                gen['total_confidence_change'] 
                for gen in overall_metrics['generator_performance'].values()
            )
            overall_metrics['average_confidence_change'] = total_change / overall_metrics['successful_questions']
        
        # Store overall metrics in results
        all_results['_overall_metrics'] = overall_metrics
        
        # Calculate majority vote comparison
        majority_vote_metrics = self._calculate_majority_vote_comparison(all_results, generators_data)
        all_results['_majority_vote_metrics'] = majority_vote_metrics
        
        return all_results
    
    def _calculate_majority_vote_comparison(self, results: Dict, generators_data: Dict) -> Dict:
        """
        Calculate majority vote results before and after delegation algorithm.
        
        Args:
            results: Delegation results
            generators_data: Original generator data
            
        Returns:
            Dictionary with majority vote comparison metrics
        """
        questions = [k for k in results.keys() if not str(k).startswith('_')]
        
        original_majority_correct = 0
        delegation_majority_correct = 0
        
        for question_idx in questions:
            if question_idx not in results:
                continue
                
            question_result = results[question_idx]
            golden_answer = question_result.get('golden_answer', 'Unknown')
            
            if golden_answer == 'Unknown':
                continue
            
            # Calculate original majority vote
            original_answers = {}
            for gen_name, gen_data in generators_data.items():
                if str(question_idx) in gen_data:
                    try:
                        answer, _, confidence_dict = self.get_highest_confidence_answer(gen_data, question_idx)
                        # Count weighted by confidence
                        for ans, conf in confidence_dict.items():
                            if ans in original_answers:
                                original_answers[ans] += conf
                            else:
                                original_answers[ans] = conf
                    except Exception:
                        continue
            
            if original_answers:
                original_majority_answer = max(original_answers.keys(), key=lambda x: original_answers[x])
                if self.evaluator and hasattr(self.evaluator, 'check_answers_equiv'):
                    if self.evaluator.check_answers_equiv(golden_answer, original_majority_answer):
                        original_majority_correct += 1
            
            # Calculate delegation algorithm majority vote
            delegation_answers = {}
            for gen_name, result in question_result.get('final_results', {}).items():
                answer = result['answer']
                confidence = result['final_confidence']
                if answer in delegation_answers:
                    delegation_answers[answer] += confidence
                else:
                    delegation_answers[answer] = confidence
            
            if delegation_answers:
                delegation_majority_answer = max(delegation_answers.keys(), key=lambda x: delegation_answers[x])
                if self.evaluator and hasattr(self.evaluator, 'check_answers_equiv'):
                    if self.evaluator.check_answers_equiv(golden_answer, delegation_majority_answer):
                        delegation_majority_correct += 1
        
        return {
            'total_questions': len(questions),
            'original_majority_correct': original_majority_correct,
            'delegation_majority_correct': delegation_majority_correct,
            'original_majority_accuracy': original_majority_correct / len(questions) if questions else 0,
            'delegation_majority_accuracy': delegation_majority_correct / len(questions) if questions else 0,
            'improvement': delegation_majority_correct - original_majority_correct
        }
    
    def save_delegation_results(self, results: Dict, output_filename: str):
        """Save the delegation results to a JSON file."""
        output_data = {
            "metadata": {
                "algorithm": "delegation_algorithm",
                "learning_rate": self.learning_rate,
                "max_iterations": self.max_iterations,
                "total_questions": len([k for k in results.keys() if not str(k).startswith('_')]),
                "discriminator_available": DISCRIMINATOR_AVAILABLE # Changed to False as discriminator is removed
            },
            "results": results
        }
        
        with open(output_filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"💾 Results saved to {output_filename}")
    
    def generate_summary_report(self, results: Dict) -> str:
        """
        Generate a human-readable summary report of the delegation results.
        
        Args:
            results: Delegation results
            
        Returns:
            Formatted summary report
        """
        if '_overall_metrics' not in results:
            return "No overall metrics available for summary report."
        
        metrics = results['_overall_metrics']
        questions = [k for k in results.keys() if not str(k).startswith('_')]
        
        report = []
        report.append("=" * 60)
        report.append("🎯 DELEGATION ALGORITHM SUMMARY REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall statistics
        report.append("📊 OVERALL STATISTICS")
        report.append(f"Total Questions: {metrics['total_questions']}")
        report.append(f"Successful Questions: {metrics['successful_questions']}")
        report.append(f"Success Rate: {metrics['successful_questions']/metrics['total_questions']*100:.1f}%")
        report.append(f"Total Confidence Improvements: {metrics['total_confidence_improvements']}")
        report.append(f"Total Correct Answers: {metrics['total_correct_answers']}")
        report.append(f"Average Confidence Change: {metrics['average_confidence_change']:.4f}")
        report.append("")
        
        # Majority vote comparison
        if '_majority_vote_metrics' in results:
            majority_metrics = results['_majority_vote_metrics']
            report.append("🗳️  MAJORITY VOTE COMPARISON")
            report.append(f"Original Majority Vote Correct: {majority_metrics['original_majority_correct']}/{majority_metrics['total_questions']} ({majority_metrics['original_majority_accuracy']*100:.1f}%)")
            report.append(f"Delegation Majority Vote Correct: {majority_metrics['delegation_majority_correct']}/{majority_metrics['total_questions']} ({majority_metrics['delegation_majority_accuracy']*100:.1f}%)")
            report.append(f"Improvement: {majority_metrics['improvement']:+d} questions")
            report.append("")
        
        # Generator performance
        report.append("🤖 GENERATOR PERFORMANCE")
        for gen_name, gen_metrics in metrics['generator_performance'].items():
            report.append(f"\n{gen_name}:")
            report.append(f"  Improvements: {gen_metrics['improvements']}")
            report.append(f"  Declines: {gen_metrics['declines']}")
            report.append(f"  Correct Answers: {gen_metrics['correct_answers']}")
            report.append(f"  Total Confidence Change: {gen_metrics['total_confidence_change']:.4f}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


def main():
    """Main function to run the delegation algorithm."""
    print("🎯 Delegation Algorithm Implementation")
    print("=" * 50)
    
    # Initialize the algorithm
    delegation = DelegationAlgorithm(
        learning_rate=0.1,
        max_iterations=5
    )
    
    # Load generator data
    try:
        data = delegation.load_generator_data('llm_answers_discriminator_0_99.json')
        print(f"✅ Loaded data for {len(data['results'])} generators")
      #  print(f"Generators: {', '.join(data['results'].keys())}")
        
        # Show data structure info
        first_gen = list(data['results'].keys())[0]
        
    except FileNotFoundError:
        print("❌ Error: llm_answers_discriminator_0_99.json not found!")
        print("Please run the extraction script first.")
        return
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Run delegation for all questions
    test_questions = (0, 99)  # Process questions 0-98 (all questions)
    print(f"\n🧪 Running delegation for questions {test_questions[0]}-{test_questions[1]-1}")
    
    try:
        results = delegation.run_delegation_for_all_questions(
            data['results'], 
            question_range=test_questions
        )
        
        # Generate and display summary report
        summary = delegation.generate_summary_report(results)
        print("\n" + summary)
        
        # Generate hyperparameter optimization suggestions
        suggestions = delegation.optimize_hyperparameters(results)
        if suggestions['recommendations']:
            print("\n🔧 HYPERPARAMETER OPTIMIZATION SUGGESTIONS:")
            for i, rec in enumerate(suggestions['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        # Save results
        output_filename = f"delegation_results_all_questions.json"
        delegation.save_delegation_results(results, output_filename)
        
        # Show detailed results for each question
        print(f"\n📋 DETAILED RESULTS:")
        for question_idx in sorted([k for k in results.keys() if not str(k).startswith('_')]):
            question_result = results[question_idx]
            print(f"\nQuestion {question_idx}:")
            
            if 'delegation_metrics' in question_result:
                metrics = question_result['delegation_metrics']
                print(f"  Golden Answer: {question_result.get('golden_answer', 'Unknown')}")
                print(f"  Correct Answers: {metrics['correct_answers']}/{metrics['total_generators']}")
                print(f"  Average Confidence Change: {metrics['average_confidence_change']:.4f}")
            
            for gen_name, gen_result in question_result['final_results'].items():
                confidence_change = gen_result['confidence_change']
                change_symbol = "↗️" if confidence_change > 0 else "↘️" if confidence_change < 0 else "➡️"
                initial_conf = gen_result.get('initial_confidence', 0.0)
                final_conf = gen_result.get('final_confidence', 0.0)
                print(f"  {gen_name}: {change_symbol} {final_conf:.3f} "
                      f"(initial: {initial_conf:.3f}, change: {confidence_change:+.3f})")
        
        # Calculate and show discriminator accuracy summary
        print(f"\n🎯 DISCRIMINATOR ACCURACY SUMMARY:")
        print("=" * 60)
        
        total_questions = 0
        correct_discriminator_choices = 0
        discriminator_accuracy_by_model = {}
        
        for question_idx in sorted([k for k in results.keys() if not str(k).startswith('_')]):
            question_result = results[question_idx]
            golden_answer = question_result.get('golden_answer', 'Unknown')
            
            if golden_answer == 'Unknown':
                continue
                
            total_questions += 1
            
            # Find the answer with highest discriminator survival rate (discriminator's choice)
            best_answer = None
            best_survival_rate = -float('inf')
            best_generator = None
            
            # Get the last iteration's cumulative survival rates for each answer
            if 'iterations' in question_result and question_result['iterations']:
                last_iteration = question_result['iterations'][-1]
                if 'cumulative_survival_rates' in last_iteration:
                    cumulative_rates = last_iteration['cumulative_survival_rates']
                    
                    # Find answer with highest survival rate
                    for answer, survival_rate in cumulative_rates.items():
                        if survival_rate > best_survival_rate:
                            best_survival_rate = survival_rate
                            best_answer = answer
                            # Find which generator provided this answer
                            for gen_name, gen_result in question_result['final_results'].items():
                                if gen_result['answer'] == answer:
                                    best_generator = gen_name
                                    break
            
            if best_answer and best_generator:
                # Check if the discriminator's choice (highest survival rate answer) was correct
                try:
                    float1 = float(best_answer)
                    float2 = float(golden_answer)
                    is_correct = abs(float1 - float2) < 1e-6
                except (ValueError, TypeError):
                    is_correct = best_answer.strip() == golden_answer.strip()
                
                if is_correct:
                    correct_discriminator_choices += 1
                
                # Track accuracy by model (which model provided the discriminator's choice)
                if best_generator not in discriminator_accuracy_by_model:
                    discriminator_accuracy_by_model[best_generator] = {'correct': 0, 'total': 0}
                
                discriminator_accuracy_by_model[best_generator]['total'] += 1
                if is_correct:
                    discriminator_accuracy_by_model[best_generator]['correct'] += 1
        
        # Show overall discriminator accuracy
        if total_questions > 0:
            overall_accuracy = (correct_discriminator_choices / total_questions) * 100
            print(f"📊 Overall Discriminator Accuracy: {overall_accuracy:.1f}% ({correct_discriminator_choices}/{total_questions})")
            
            # Show accuracy breakdown by model
            print(f"\n🎯 Discriminator Choice Accuracy by Model:")
            for model, stats in discriminator_accuracy_by_model.items():
                if stats['total'] > 0:
                    model_accuracy = (stats['correct'] / stats['total']) * 100
                    print(f"   {model}: {model_accuracy:.1f}% ({stats['correct']}/{stats['total']})")
        
        # Calculate and show generator performance comparison
        print(f"\n🚀 GENERATOR PERFORMANCE COMPARISON:")
        print("=" * 60)
        
        generator_performance = {}
        
        # Initialize performance tracking for each generator
        for gen_name in data['results'].keys():
            generator_performance[gen_name] = {
                'original_correct': 0,
                'original_total': 0,
                'delegation_correct': 0,
                'delegation_total': 0,
                'confidence_improvements': 0,
                'confidence_declines': 0
            }
        
        # Calculate performance metrics for each generator
        for question_idx in sorted([k for k in results.keys() if not str(k).startswith('_')]):
            question_result = results[question_idx]
            golden_answer = question_result.get('golden_answer', 'Unknown')
            
            if golden_answer == 'Unknown':
                continue
            
            for gen_name, gen_result in question_result['final_results'].items():
                if gen_name in generator_performance:
                    # Check if answer is correct
                    selected_answer = gen_result['answer']
                    try:
                        float1 = float(selected_answer)
                        float2 = float(golden_answer)
                        is_correct = abs(float1 - float2) < 1e-6
                    except (ValueError, TypeError):
                        is_correct = selected_answer.strip() == golden_answer.strip()
                    
                    # Track delegation performance
                    generator_performance[gen_name]['delegation_total'] += 1
                    if is_correct:
                        generator_performance[gen_name]['delegation_correct'] += 1
                    
                    # Track confidence changes
                    confidence_change = gen_result.get('confidence_change', 0.0)
                    if confidence_change > 0:
                        generator_performance[gen_name]['confidence_improvements'] += 1
                    elif confidence_change < 0:
                        generator_performance[gen_name]['confidence_declines'] += 1
        
        # Calculate original accuracy from the input data
        for gen_name, gen_data in data['results'].items():
            for question_idx, question_data in gen_data.items():
                if question_data.get('golden_answer'):
                    golden_answer = question_data['golden_answer']
                    selected_answer = question_data.get('answer', '')
                    
                    if selected_answer:
                        try:
                            float1 = float(selected_answer)
                            float2 = float(golden_answer)
                            is_correct = abs(float1 - float2) < 1e-6
                        except (ValueError, TypeError):
                            is_correct = selected_answer.strip() == golden_answer.strip()
                        
                        generator_performance[gen_name]['original_total'] += 1
                        if is_correct:
                            generator_performance[gen_name]['original_correct'] += 1
        
        # Display generator performance comparison
        print(f"📊 Generator Performance: Original vs. After Delegation")
        print(f"   Format: Model: Original Accuracy → Delegation Accuracy (Improvement)")
        
        for gen_name, perf in generator_performance.items():
            if perf['original_total'] > 0 and perf['delegation_total'] > 0:
                original_acc = (perf['original_correct'] / perf['original_total']) * 100
                delegation_acc = (perf['delegation_correct'] / perf['delegation_total']) * 100
                improvement = delegation_acc - original_acc
                
                improvement_symbol = "↗️" if improvement > 0 else "↘️" if improvement < 0 else "➡️"
                
                print(f"   {gen_name}:")
                print(f"     Original Accuracy: {original_acc:.1f}% ({perf['original_correct']}/{perf['original_total']})")
                print(f"     Delegation Accuracy: {delegation_acc:.1f}% ({perf['delegation_correct']}/{perf['delegation_total']})")
                print(f"     Change: {improvement_symbol} {improvement:+.1f}%")
                print(f"     Confidence: +{perf['confidence_improvements']} / -{perf['confidence_declines']}")
                print(f"     ---")
        
        # Add multi-model comparison section
        print(f"\n🏆 MULTI-MODEL COMPARISON:")
        print("=" * 60)
        
        # Show multi-model comparison using the data we already have
        print(f"📊 Multi-Model Analysis from Generator Data:")
        print(f"   Models available: {', '.join(data['results'].keys())}")
        print(f"   Questions processed: {test_questions[0]}-{test_questions[1]-1}")
        
        # Show confidence evolution comparison for ALL questions
        all_questions = list(range(test_questions[0], test_questions[1]))
        print(f"   📊 Analyzing {len(all_questions)} questions for detailed multi-model comparison...")
        
        for question_idx in all_questions:
            if str(question_idx) in results:
                question_result = results[str(question_idx)]
                print(f"\n🔍 Question {question_idx}:")
                
                # Show golden answer if available
                golden_answer = question_result.get('golden_answer', 'Unknown')
                print(f"   Golden Answer: {golden_answer}")
                
                # Show results for each model in a compact format
                if 'final_results' in question_result:
                    for gen_name, gen_result in question_result['final_results'].items():
                        initial_conf = gen_result.get('initial_confidence', 0.0)
                        final_conf = gen_result.get('final_confidence', 0.0)
                        confidence_change = gen_result.get('confidence_change', 0.0)
                        
                        change_symbol = "↗️" if confidence_change > 0 else "↘️" if confidence_change < 0 else "➡️"
                        
                        print(f"   {gen_name}: {change_symbol} {final_conf:.3f} (initial: {initial_conf:.3f}, change: {confidence_change:+.3f})")
                
                # Show last iteration's discriminator survival scores in a compact format
                if 'iterations' in question_result and question_result['iterations']:
                    last_iteration = question_result['iterations'][-1]
                    print(f"   🎯 Discriminator Scores:")
                    
                    # Show normalized scores (these sum to 0)
                    if 'normalized_scores' in last_iteration:
                        norm_scores = [f"{gen_name}: {norm_score:+.3f}" for gen_name, norm_score in last_iteration['normalized_scores'].items()]
                        print(f"     Normalized: {', '.join(norm_scores)}")
                    
                    # Show cumulative survival rates for each answer
                    if 'cumulative_survival_rates' in last_iteration:
                        cum_rates = [f"'{answer}': {cum_rate:+.3f}" for answer, cum_rate in last_iteration['cumulative_survival_rates'].items()]
                        print(f"     Cumulative: {', '.join(cum_rates)}")
                
                # Show delegation metrics if available
                if 'delegation_metrics' in question_result:
                    metrics = question_result['delegation_metrics']
                    print(f"   📊 Correct: {metrics.get('correct_answers', 0)}/{metrics.get('total_generators', 0)}, Avg Change: {metrics.get('average_confidence_change', 0.0):.3f}")
        
        print(f"\n🎯 Multi-Model Comparison Summary:")
        print(f"   • Shows confidence evolution for all 3 models (Phi-3, Llama3, Mistral)")
        print(f"   • Displays original vs updated confidence for each model")
        print(f"   • Provides side-by-side comparison of model performance")
        print(f"   • Uses data already available in the generator outputs")
        print(f"   • Shows last iteration discriminator survival scores for each model")
        print(f"   • Displays normalized scores (sum to 0) and cumulative survival rates by answer")
        
        print(f"\n🎉 Delegation algorithm completed successfully!")
        print(f"📁 Results saved to: {output_filename}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Delegation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during delegation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 