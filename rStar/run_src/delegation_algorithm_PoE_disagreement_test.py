
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
import hashlib
import re
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
        
        # Initialize discriminator cache
        self.discriminator_cache = {}
        self.question_contexts = {}
        
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
                        self.rc_mode = "loose"
                        self.num_masked_solution_traces = 4  # Default value
                        self.mask_left_boundary = 0.2
                        self.mask_right_boundary = 0.8
                        self.rc_n_completions = 1  # Default value
                        self.rc_temperature = 0.8
                        self.max_num_seqs = 32
                        self.discriminate_results_dir = "./temp_discriminator_results"
                        self.api = "vllm"  # Switch to vLLM API for better performance
                        self.model_ckpt = "Qwen/Qwen2-7B"  # Use Qwen model for discriminator
                        self.hf_token ="your_hf_token"  # Add HF token (None for public models)
                        self.seed = 42  # Add seed for reproducibility
                        # Add memory optimization parameters
                        self.max_model_len = 2048  # Reduced from 4096 to fix KV cache memory
                        self.gpu_memory_utilization = 0.9  # Use 90% of GPU memory
                        
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
            
            gen_model = getattr(discriminator, "model", None)
            if gen_model is None and hasattr(discriminator, "llm"):
                gen_model = discriminator.llm
            if gen_model is None and hasattr(discriminator, "engine"):
                gen_model = discriminator.engine

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
            candidates: List of Candidate objects or strings
            
        Returns:
            Dictionary mapping answer to score
        """
        scores = {}
        for candidate in candidates:
            # Handle both candidate objects and simple strings
            if hasattr(candidate, 'final_answer'):
                # Candidate object
                answer = candidate.final_answer
                score = 0.5  # Base score
                
                if hasattr(candidate, 'solution_trace') and candidate.solution_trace:
                    trace_length = len(candidate.solution_trace)
                    if trace_length > 50:
                        score += 0.2
                    if 'step' in candidate.solution_trace.lower():
                        score += 0.1
                    if 'answer' in candidate.solution_trace.lower():
                        score += 0.1
            else:
                # Simple string answer
                answer = str(candidate)
                score = 0.5  # Base score for all string answers
            
            scores[answer] = score
        
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
    
    def _poe_update_distribution(
        self,
        p_t: dict[str, float],          # 生成器当前分布
        s_t: dict[str, float],          # 判别器的答案级 signal (零均值)
        eta: float,                     # 学习率
        alpha: float = 0.4,             # 控制先验/证据信任度
        mix_uniform: float = 0.0,       # 可选：和均匀分布混合，防坍缩
    ) -> dict[str, float]:
        """
        Product-of-Experts 更新:
        p_{t+1}(a) ∝ p_t(a)^(1-α) * exp(η * s_t(a))^α
        
        Args:
            p_t: 生成器当前答案分布
            s_t: 判别器的零均值信号
            eta: 学习率
            alpha: 信任度参数 (0=完全信任生成器，1=完全信任判别器)
            mix_uniform: 与均匀分布的混合比例
            
        Returns:
            更新后的答案分布
        """
        # 1. 计算 log-odds 形式的更新
        mixed = {}
        for a in p_t:
            # PoE 更新: log p_{t+1}(a) = (1-α) * log p_t(a) + α * (η * s_t(a))
            log_p_t = math.log(p_t[a] + 1e-10)  # 避免 log(0)
            signal_term = eta * s_t.get(a, 0.0)
            mixed[a] = (1 - alpha) * log_p_t + alpha * signal_term
        
        # 2. 转换回概率分布 (softmax)
        max_log = max(mixed.values())
        mixed = {a: math.exp(mixed[a] - max_log) for a in mixed}  # 数值稳定
        sum_exp = sum(mixed.values())
        mixed = {a: mixed[a] / sum_exp for a in mixed}
        
        # 3. 可选：与均匀分布混合
        if mix_uniform > 0:
            n = len(mixed)
            u = 1.0 / n
            mixed = {a: (1 - mix_uniform) * mixed[a] + mix_uniform * u for a in mixed}
        return mixed

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
    
    def _normalize_answer_text(self, text: str) -> str:
        """Normalize answer text for consistent comparison."""
        if not text:
            return ""
        # Remove extra whitespace and normalize
        normalized = re.sub(r'\s+', ' ', text.strip())
        return normalized.lower()
    
    def _candidate_set_key(self, question_text: str, answers: List[str]) -> str:
        """
        生成稳定的候选集 key：只依赖题目文本 + 答案文本集合（排序、标准化），
        不包含置信度、路径、顺序等易变因素。
        """
        norm_q = self._normalize_answer_text(question_text or "")
        norm_answers = sorted([self._normalize_answer_text(a) for a in answers])
        
        # 创建稳定的字符串表示
        content = f"q:{norm_q}|answers:{'|'.join(norm_answers)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _candidates_changed(self, question_idx: int, current_candidates: List[Tuple[str, str, float]]) -> bool:
        """
        Check if candidates have changed for a question since last iteration.
        
        Args:
            question_idx: Question index
            current_candidates: Current list of (answer, path, confidence) tuples
            
        Returns:
            True if candidates have changed, False otherwise
        """
        if question_idx not in self.discriminator_cache:
            return True
        
        # Get cached candidates for this question
        cached_bucket = self.discriminator_cache[question_idx]
        
        # Check if any cached result exists for current candidate set
        question_context = self.question_contexts.get(question_idx, "")
        unique_answers = list(set([answer for answer, _, _ in current_candidates]))
        cache_key = self._candidate_set_key(question_context, unique_answers)
        
        return cache_key not in cached_bucket
    
    def clear_cache(self, question_idx: int = None):
        """
        Clear the discriminator cache.
        
        Args:
            question_idx: Specific question to clear, or None to clear all
        """
        if question_idx is None:
            self.discriminator_cache.clear()
            print("🗑️  Cleared all discriminator cache")
        elif question_idx in self.discriminator_cache:
            del self.discriminator_cache[question_idx]
            print(f"🗑️  Cleared cache for question {question_idx}")
        else:
            print(f"ℹ️  No cache found for question {question_idx}")
    
    def has_disagreement(self, initial_data: Dict) -> bool:
        """
        Check if generators disagree on their best answers.
        
        Args:
            initial_data: Dictionary with generator data containing their best answers
            
        Returns:
            True if generators disagree (no majority of 2+ generators agree), False if they agree
        """
        if len(initial_data) < 2:
            return False  # Need at least 2 generators to have disagreement
        
        # Get the best answer from each generator
        best_answers = [data['answer'] for data in initial_data.values()]
        
        # Count occurrences of each answer
        answer_counts = {}
        for answer in best_answers:
            answer_counts[answer] = answer_counts.get(answer, 0) + 1
        
        # Check if any answer has 2 or more votes (majority agreement)
        max_votes = max(answer_counts.values())
        
        # If any answer has 2+ votes, consider it agreement (no disagreement)
        if max_votes > 2:
            return False  # At least 3 generators agree
        
        return True  # No answer has 2+ votes, so there's disagreement

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
        generator_names = list(generators_data['results'].keys())
        
        # Initialize results for this question
        question_results = {
            'question_idx': question_idx,
            'iterations': [],
            'final_results': {},
            'golden_answer': None,
            'had_disagreement': False,  # Track whether generators disagreed
            'chosen_answers': {}
        }
        
        # Get initial data for each generator
        initial_data = {}
        all_answers_paths = []  # Collect all (answer, path, confidence) tuples for discriminator
        
        for gen_name in generator_names:
            gen_data = generators_data['results'][gen_name]
            if str(question_idx) in gen_data:
                try:
                    answer, path, confidence_dict = self.get_highest_confidence_answer(gen_data, question_idx)
                    question_results['chosen_answers'][gen_name] = {"answer": answer}

                    # Store golden answer from first generator (assuming it's available)
                    if question_results['golden_answer'] is None:
                        question_results['golden_answer'] = gen_data[str(question_idx)].get('golden_answer', 'Unknown')
 
                    # Get the confidence for the best answer
                    best_confidence = confidence_dict[answer]
                    initial_data[gen_name] = {
                        'answer': answer,
                        'original_answer': answer,  # Store original answer for before/after comparison
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
        
        # Check if generators disagree on their best answers
        if not self.has_disagreement(initial_data):
            print(f"🤝 Question {question_idx}: Generators have majority agreement (>2 agree), skipping delegation")
            # Return results without delegation - just use the agreed answer
            question_results['final_results'] = {
                gen_name: {
                    'answer': data['answer'],
                    'confidence': data['confidence'],
                    'solution_path': data['solution_path'],
                    'original_answer': data['answer'],
                    'original_confidence': data['confidence'],
                    'final_confidence': data['confidence'],
                    'confidence_change': 0.0  # No change since no delegation occurred
                } for gen_name, data in initial_data.items()
            }
            question_results['had_disagreement'] = False
            
            # Calculate delegation metrics for agreement case
            question_results['delegation_metrics'] = self._calculate_delegation_metrics(
                question_results['final_results'],
                question_results['golden_answer']
            )
            
            return question_results
        
        print(f"⚔️ Question {question_idx}: Generators disagree (any different answer), running delegation")
        question_results['had_disagreement'] = True
        
        # Run multiple iterations
        for iteration in range(self.max_iterations):
            # print(f"  📊 Iteration {iteration + 1}/{self.max_iterations}")
            
            # Get discriminator scores for all answers with cache support
            survival_scores = []
            
            # Check cache first
            question_context = self.question_contexts.get(question_idx, "")
            unique_answers = list(set([answer for answer, _, _ in all_answers_paths]))
            cache_key = self._candidate_set_key(question_context, unique_answers)
            
            if question_idx in self.discriminator_cache and cache_key in self.discriminator_cache[question_idx]:
                # Cache hit
                print(f"  🚀 Cache hit! Using cached survival rates for question {question_idx}")
                cached_scores = self.discriminator_cache[question_idx][cache_key]
                survival_scores = [cached_scores.get(gen_name, 0.0) for gen_name in generator_names]
            else:
                # Cache miss - compute scores
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
                
                # Cache the results
                if question_idx not in self.discriminator_cache:
                    self.discriminator_cache[question_idx] = {}
                self.discriminator_cache[question_idx][cache_key] = dict(zip(generator_names, survival_scores))
            
            # Normalize scores to sum to 0
            normalized_scores = self.normalize_scores(survival_scores)
            
            # === 用 PoE 更新 ===
            eta = self.learning_rate
            alpha = 0.4  # 信任度参数
            mix_uniform = 0.0  # 可选：和均匀分布混合
            
            for i, gen_name in enumerate(generator_names):
                if gen_name not in initial_data:
                    continue
                    
                data = initial_data[gen_name]
                p_t = data['all_confidences'].copy()
                
                # 构建答案级信号 s_t (零均值)
                s_t = {}
                answer = data['answer']
                s_t[answer] = normalized_scores[i]  # 当前答案的信号
                
                # 对其他答案，给一个小的负信号
                for other_answer in p_t:
                    if other_answer != answer:
                        s_t[other_answer] = -normalized_scores[i] * 0.1  # 小负信号
                
                # PoE 更新
                p_next = self._poe_update_distribution(
                    p_t=p_t,
                    s_t=s_t,
                    eta=eta,
                    alpha=alpha,
                    mix_uniform=mix_uniform
                )
                # 写回
                initial_data[gen_name]['all_confidences'] = p_next
                sel = max(p_next, key=p_next.get)
                initial_data[gen_name]['answer'] = sel
                initial_data[gen_name]['confidence'] = p_next[sel]
            # === 用 PoE 更新结束 ===

            # 刷新候选集合：用最新 top1 + path + conf
            all_answers_paths = []
            for gen_name in generator_names:
                if gen_name not in initial_data:
                    continue
                sel = initial_data[gen_name]['answer']
                path = initial_data[gen_name]['solution_path']  # 若有 per-answer path，这里按 sel 取
                conf = initial_data[gen_name]['confidence']
                all_answers_paths.append((sel, path, conf))
                    

            # Store iteration results
            iteration_result = {
                'iteration': iteration + 1,
                'survival_scores': dict(zip(generator_names, survival_scores)),
                'normalized_scores': dict(zip(generator_names, normalized_scores)),
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
        
        # Show final survival rates with generator mapping
        print(f"   🎯 Final Survival Rates by Generator:")
        for gen_name, score in zip(generator_names, survival_scores):
            print(f"     {gen_name}: {score:+.4f}")
        
        # Show normalized scores with generator mapping
        print(f"   ⚖️  Final Normalized Scores (sum to 0):")
        for gen_name, norm_score in zip(generator_names, normalized_scores):
            print(f"     {gen_name}: {norm_score:+.4f}")

        # Store final results
        question_results['final_results'] = {
            gen_name: {
                'answer': data['answer'],
                'original_answer': data['original_answer'],  # Preserve original answer for before/after comparison
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
            first_gen = list(generators_data['results'].keys())[0]
            questions = list(generators_data['results'][first_gen].keys())
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
                'correct_answers_before': 0,  # correct answers before delegation
                'correct_answers_after': 0,   # correct answers after delegation
                'total_confidence_change': 0.0,
                'accuracy_before': 0.0,      # accuracy before delegation
                'accuracy_after': 0.0        # accuracy after delegation
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
                        
                        # Check correctness before and after delegation
                        golden_answer = question_result.get('golden_answer', 'Unknown')
                        if golden_answer != 'Unknown':
                            # Use proper evaluator method for answer equivalence checking
                            if self.evaluator and hasattr(self.evaluator, 'check_answers_equiv'):
                                # Check if original answer was correct BEFORE delegation
                                original_answer = result.get('original_answer', '')
                                if original_answer and self.evaluator.check_answers_equiv(original_answer, golden_answer):
                                    gen_metrics['correct_answers_before'] += 1
                                
                                # Check if answer was correct AFTER delegation
                                if self.evaluator.check_answers_equiv(result['answer'], golden_answer):
                                    gen_metrics['correct_answers_after'] += 1
                                    overall_metrics['total_correct_answers'] += 1
                            else:
                                # Fallback to simple method
                                original_answer = result.get('original_answer', '')
                                if original_answer and self._check_answers_equiv(original_answer, golden_answer):
                                    gen_metrics['correct_answers_before'] += 1
                                
                                if self._check_answers_equiv(result['answer'], golden_answer):
                                    gen_metrics['correct_answers_after'] += 1
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
            
            # Calculate accuracy percentages for each generator
            for gen_name, gen_metrics in overall_metrics['generator_performance'].items():
                if overall_metrics['total_questions'] > 0:
                    gen_metrics['accuracy_before'] = gen_metrics['correct_answers_before'] / overall_metrics['total_questions']
                    gen_metrics['accuracy_after'] = gen_metrics['correct_answers_after'] / overall_metrics['total_questions']
        
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
        discriminator_correct = 0
        
        for question_idx in questions:
            if question_idx not in results:
                continue
                
            question_result = results[question_idx]
            golden_answer = question_result.get('golden_answer', 'Unknown')
            
            if golden_answer == 'Unknown':
                continue
            
            # Calculate original majority vote
            original_answers = {}
            for gen_name, result in question_result.get('final_results', {}).items():
                original_answer = result.get('original_answer', '')
                if original_answer:
                    if original_answer in original_answers:
                        original_answers[original_answer] += 1
                    else:
                        original_answers[original_answer] = 1
            
            if original_answers:
                # Handle ties randomly
                max_votes = max(original_answers.values())

                tied_answers = [ans for ans, votes in original_answers.items() if votes == max_votes]
                # original_majority_answer = random.choice(tied_answers) if len(tied_answers) > 1 else tied_answers[0]
                original_majority_answer = min(tied_answers)

                if self.evaluator and hasattr(self.evaluator, 'check_answers_equiv'):
                    if self.evaluator.check_answers_equiv(golden_answer, original_majority_answer):
                        original_majority_correct += 1
            
            # Calculate delegation algorithm majority vote
            delegation_answers = {}
            for gen_name, result in question_result.get('final_results', {}).items():
                answer = result['answer']
                # Count votes (not confidence scores) - each generator gets 1 vote for their best answer
                if answer in delegation_answers:
                    delegation_answers[answer] += 1
                else:
                    delegation_answers[answer] = 1
            
            if delegation_answers:
                # Handle ties randomly
                max_votes = max(delegation_answers.values())
                tied_answers = [ans for ans, votes in delegation_answers.items() if votes == max_votes]
                # delegation_majority_answer = random.choice(tied_answers) if len(tied_answers) > 1 else tied_answers[0]
                delegation_majority_answer = min(tied_answers)

                if self.evaluator and hasattr(self.evaluator, 'check_answers_equiv'):
                    if self.evaluator.check_answers_equiv(golden_answer, delegation_majority_answer):
                        delegation_majority_correct += 1
            
            # Calculate discriminator accuracy (how often discriminator's best choice matches golden answer)
            if question_idx in results and 'iterations' in results[question_idx]:
                # Get the last iteration's survival scores to find discriminator's best choice
                last_iteration = results[question_idx]['iterations'][-1] if results[question_idx]['iterations'] else None
                if last_iteration and 'survival_scores' in last_iteration:
                    survival_scores = last_iteration['survival_scores']
                    if survival_scores:
                        # Find the generator with highest survival score (discriminator's choice)
                        discriminator_best_generator = max(survival_scores.keys(), key=lambda x: survival_scores[x])
                        # Get the answer from the best generator
                        if question_idx in results and 'final_results' in results[question_idx]:
                            final_results = results[question_idx]['final_results']
                            if discriminator_best_generator in final_results:
                                discriminator_best_answer = final_results[discriminator_best_generator]['answer']
                                if self.evaluator and hasattr(self.evaluator, 'check_answers_equiv'):
                                    if self.evaluator.check_answers_equiv(golden_answer, discriminator_best_answer):
                                        discriminator_correct += 1
        
        return {
            'total_questions': len(questions),
            'original_majority_correct': original_majority_correct,
            'delegation_majority_correct': delegation_majority_correct,
            'original_majority_accuracy': original_majority_correct / len(questions) if questions else 0,
            'delegation_majority_accuracy': delegation_majority_correct / len(questions) if questions else 0,
            'improvement': delegation_majority_correct - original_majority_correct,
            'discriminator_correct': discriminator_correct,
            'discriminator_accuracy': discriminator_correct / len(questions) if questions else 0
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
        
        # Calculate overall generator improvement
        total_improvements = 0
        total_generators = len(metrics['generator_performance'])
        for gen_metrics in metrics['generator_performance'].values():
            if gen_metrics['accuracy_after'] > gen_metrics['accuracy_before']:
                total_improvements += 1
        
        report.append(f"Generators Improved: {total_improvements}/{total_generators}")
        report.append("")
        
        # Majority vote comparison
        if '_majority_vote_metrics' in results:
            majority_metrics = results['_majority_vote_metrics']
            report.append("🗳️  MAJORITY VOTE COMPARISON")
            report.append("=" * 40)
            report.append(f"Original Majority Vote:     {majority_metrics['original_majority_correct']:>3}/{majority_metrics['total_questions']:<3} ({majority_metrics['original_majority_accuracy']*100:>5.1f}%)")
            report.append(f"Delegation Majority Vote:   {majority_metrics['delegation_majority_correct']:>3}/{majority_metrics['total_questions']:<3} ({majority_metrics['delegation_majority_accuracy']*100:>5.1f}%)")
            report.append(f"Improvement:                {majority_metrics['improvement']:+3d} questions")
            
            if majority_metrics['improvement'] > 0:
                report.append(f"🎉 Delegation algorithm improved majority vote accuracy!")
            elif majority_metrics['improvement'] < 0:
                report.append(f"⚠️  Delegation algorithm decreased majority vote accuracy")
            else:
                report.append(f"➡️  Delegation algorithm maintained majority vote accuracy")
            
            # Add golden answer guidance accuracy
            report.append("")
            report.append("🎯 GOLDEN ANSWER GUIDANCE ACCURACY")
            report.append("-" * 40)
            report.append(f"Guidance Best Choice:       {majority_metrics['discriminator_correct']:>3}/{majority_metrics['total_questions']:<3} ({majority_metrics['discriminator_accuracy']*100:>5.1f}%)")
            
            # Compare guidance vs majority vote
            if majority_metrics['discriminator_accuracy'] > majority_metrics['original_majority_accuracy']:
                report.append(f"🏆 Golden answer guidance outperforms original majority vote!")
            elif majority_metrics['discriminator_accuracy'] < majority_metrics['original_majority_accuracy']:
                report.append(f"⚠️  Golden answer guidance underperforms original majority vote")
            else:
                report.append(f"➡️  Golden answer guidance matches original majority vote")
            report.append("")
        
        # Generator performance
        report.append("🤖 GENERATOR PERFORMANCE COMPARISON")
        report.append("=" * 80)
        report.append(f"{'Generator':<25} {'Before':<20} {'After':<20} {'Change':<15}")
        report.append(f"{'':<25} {'Accuracy':<10} {'Count':<10} {'Accuracy':<10} {'Count':<10} {'(+/-)':<15}")
        report.append("-" * 80)
        
        for gen_name, gen_metrics in metrics['generator_performance'].items():
            accuracy_before = gen_metrics['accuracy_before'] * 100
            accuracy_after = gen_metrics['accuracy_after'] * 100
            accuracy_change = accuracy_after - accuracy_before
            count_before = gen_metrics['correct_answers_before']
            count_after = gen_metrics['correct_answers_after']
            count_change = count_after - count_before
            change_symbol = "↗️" if accuracy_change > 0 else "↘️" if accuracy_change < 0 else "➡️"
            
            report.append(f"{gen_name:<25} {accuracy_before:>6.1f}% {count_before:>8} {accuracy_after:>6.1f}% {count_after:>8} {change_symbol} {accuracy_change:+5.1f}%")
        
        report.append("-" * 80)
        report.append("")
        
        # Detailed generator metrics
        report.append("📊 DETAILED GENERATOR METRICS")
        for gen_name, gen_metrics in metrics['generator_performance'].items():
            report.append(f"\n{gen_name}:")
            report.append(f"  Confidence Improvements: {gen_metrics['improvements']}")
            report.append(f"  Confidence Declines: {gen_metrics['declines']}")
            report.append(f"  Correct Answers Before: {gen_metrics['correct_answers_before']}")
            report.append(f"  Correct Answers After: {gen_metrics['correct_answers_after']}")
            report.append(f"  Total Confidence Change: {gen_metrics['total_confidence_change']:.4f}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)

    def submit_all_answers_for_generator(self, generator_results: Dict, question_idx: int) -> Dict[str, Dict]:
        """
        For a given generator and question, submit ALL answers (not just the best one).
        This is useful for recording/analysis.

        Args:
            generator_results: Results for one generator
            question_idx: Question index

        Returns:
            Dictionary mapping each answer -> {
                "solution_path": str,
                "confidence": float
            }
        """
        question_data = generator_results[str(question_idx)]

        # Get full confidence distribution
        confidence_dict = question_data['answer_confidence']

        # Record every answer with its path + confidence
        all_submissions = {}
        for answer, conf in confidence_dict.items():
            solution_path = question_data['answer_to_solution_path'].get(answer, "")
            all_submissions[answer] = {
                "solution_path": solution_path,
                "confidence": conf
            }

        return all_submissions
        
    def get_principal_prob_distribution(
        self,
        model,
        question_text: str,
        agent_answers: List[str],
        *,
        return_dist: bool = False  # 保留但始终返回 (per_answer, principal)
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
        """
        vLLM 版本：返回
          1) per_answer: {answer: {"yes": p_yes, "no": p_no}}  (同一答案内归一化)
          2) principal_dist: {answer: prob}  (用 p_yes 在候选间归一化)
        兼容封装器（如 MajorityVoteDiscriminator）：自动解包出内部 vLLM LLM。
        """
        from typing import Any, Dict, List, Tuple
        from math import exp
        from vllm import SamplingParams

        def _uniform_pair() -> Dict[str, Dict[str, float]]:
            return {ans: {"yes": 0.5, "no": 0.5} for ans in agent_answers}

        def _answers_dist_from_yes(per_answer: Dict[str, Dict[str, float]]) -> Dict[str, float]:
            scores = {ans: d["yes"] for ans, d in per_answer.items()}
            s = sum(scores.values())
            if s <= 0:
                n = len(scores)
                return {ans: 1.0 / n for ans in scores} if n else {}
            return {ans: v / s for ans, v in scores.items()}

        def _unwrap_vllm_llm(obj):
            if obj is not None and hasattr(obj, "generate") and not hasattr(obj, "parameters"):
                return obj
            for attr in ("model", "llm", "engine", "client"):
                inner = getattr(obj, attr, None)
                if inner is not None and hasattr(inner, "generate"):
                    return inner
            return None

        def _get_tokenizer(llm) -> Any:
            tok = getattr(llm, "tokenizer", None)
            if tok is None and hasattr(llm, "get_tokenizer"):
                try:
                    tok = llm.get_tokenizer()
                except Exception:
                    tok = None
            return tok

        def _decode_token(tok_obj: Any, tok) -> str:
            try:
                if hasattr(tok_obj, "token"):
                    return str(tok_obj.token)
                if isinstance(tok_obj, str):
                    return tok_obj
                if isinstance(tok_obj, int):
                    if tok is not None:
                        try:
                            return tok.decode([tok_obj])
                        except Exception:
                            try:
                                return tok.convert_ids_to_tokens([tok_obj])[0]
                            except Exception:
                                return str(tok_obj)
                    return str(tok_obj)
                return str(tok_obj)
            except Exception:
                return str(tok_obj)

        def _lp_value(lp: Any) -> float | None:
            try:
                if isinstance(lp, (float, int)):
                    return float(lp)
                if hasattr(lp, "logprob"):
                    return float(lp.logprob)
                if isinstance(lp, dict) and "logprob" in lp:
                    return float(lp["logprob"])
            except Exception:
                return None
            return None

        def _norm_token_key(s: str) -> str:
            s = str(s)
            s = s.strip().lstrip(" ▁").strip("'\"")
            return s.lower()

        if not agent_answers:
            return {}, {}

        llm = _unwrap_vllm_llm(model)
        if llm is None:
            per_answer = _uniform_pair()
            return per_answer, _answers_dist_from_yes(per_answer)

        tok = _get_tokenizer(llm)

        prompts = [
            (
                f"Question: {question_text}\n"
                f"Answer: {ans}\n"
                f"Is this answer correct? Yes or No?\n"
                f"Answer:"
            )
            for ans in agent_answers
        ]
        sp = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            stop=["\n"],
            logprobs=20
        )

        try:
            outputs = llm.generate(prompts, sp)
        except Exception as e:
            print(f"❌ vLLM generation failed: {e}")
            per_answer = _uniform_pair()
            return per_answer, _answers_dist_from_yes(per_answer)

        per_answer: Dict[str, Dict[str, float]] = {}

        for ans, out in zip(agent_answers, outputs):
            p_yes = None
            p_no = None
            try:
                out0 = out.outputs[0]
                top_lp_seq = getattr(out0, "top_logprobs", None)
                if top_lp_seq is None:
                    top_lp_seq = getattr(out0, "logprobs", None)

                first_step = None
                if isinstance(top_lp_seq, list) and len(top_lp_seq) > 0:
                    first_step = top_lp_seq[0]
                else:
                    first_step = top_lp_seq

                if isinstance(first_step, dict):
                    for k, lp in first_step.items():
                        token_text = _decode_token(k, tok)
                        key = _norm_token_key(token_text)
                        val = _lp_value(lp)
                        if val is None:
                            continue
                        if key in ("yes", "yes.", "yes,"):
                            p_yes = exp(val)
                        elif key in ("no", "no.", "no,"):
                            p_no = exp(val)

                elif isinstance(first_step, list) and len(first_step) > 0:
                    for obj in first_step:
                        token_text = _decode_token(obj, tok)
                        key = _norm_token_key(token_text)
                        val = _lp_value(obj)
                        if val is None:
                            continue
                        if key in ("yes", "yes.", "yes,"):
                            p_yes = exp(val)
                        elif key in ("no", "no.", "no,"):
                            p_no = exp(val)

                if p_yes is None and p_no is None:
                    text = (out0.text or "").strip().lower()
                    if text.startswith("yes"):
                        p_yes, p_no = 1.0, 0.0
                    elif text.startswith("no"):
                        p_yes, p_no = 0.0, 1.0
                    else:
                        p_yes, p_no = 0.5, 0.5

            except Exception as e:
                print(f"⚠️ Error parsing vLLM logprobs for '{ans}': {e}")
                p_yes, p_no = 0.5, 0.5

            s = (p_yes or 0.0) + (p_no or 0.0)
            if s <= 0:
                per_answer[ans] = {"yes": 0.5, "no": 0.5}
            else:
                per_answer[ans] = {"yes": float((p_yes or 0.0) / s), "no": float((p_no or 0.0) / s)}

        principal_dist = _answers_dist_from_yes(per_answer)
        return per_answer, principal_dist


    def get_agent_principal_probs(self, generator_results: Dict, question_idx: int) -> Dict[str, float]:
        """
        For a given generator and question, submit all answers (with solution paths)
        to the principal, and return the probability distribution.
        """
        # Get all submissions for this generator (dict of answers -> {path, confidence})
        all_submissions = self.submit_all_answers_for_generator(generator_results, question_idx)

        #Extract candidate answers (just the raw answers) 
        agent_answers = list(all_submissions.keys())
        
        # Get the actual question text for context
        question_text = self.question_contexts.get(question_idx, "")

        # Ask the principal to assign probabilities
        _, probs = self.get_principal_prob_distribution(
            #self.discriminator_model,
            self._current_discriminator,
            question_text,
            agent_answers
        )

        return probs


    def run_principal_for_all_questions(self, generators_data: Dict, question_range: Tuple[int, int]) -> Dict:
        """
        Run the principal probability assignment for all questions and all generators.
        
        Args:
            generators_data: Dict of {generator_name: {question_idx: {...}}}
            question_range: Tuple (start_idx, end_idx) for questions
            
        Returns:
            Dictionary of {question_idx: {generator_name: {AnswerLetter: Probability}}}
        """
        start_idx, end_idx = question_range
        all_probs = {}

        for question_idx in range(start_idx, end_idx):
            all_probs[question_idx] = {}
            for gen_name, gen_data in generators_data.items():
                if str(question_idx) not in gen_data:
                    continue
                try:
                    probs = self.get_agent_principal_probs(gen_data, question_idx)
                    all_probs[question_idx][gen_name] = probs
                except Exception as e:
                    print("=" * 80)
                    print(f"⚠️ Error processing gen={gen_name}, q={question_idx}")
                    print(f"Exception: {repr(e)}")
                    traceback.print_exc() 
                    print("=" * 80)
                    all_probs[question_idx][gen_name] = {}
        return all_probs

    def compute_generator_principal_correlation(
        self,
        generators_data: Dict[str, Dict],
        question_range: Tuple[int, int],
        method: str = "pearson"
    ) -> Dict[str, Dict]:
        """
        Compute correlation between generator confidence distributions and
        principal probability distributions across a range of questions.
        Also print example distributions for the first question of each generator,
        and compute % of correlations that fall within [0, 1].
        Additionally, record positively correlated question indices and show
        up to 1 negative correlation example per generator.
        """
        probs = self.run_principal_for_all_questions(generators_data, question_range)
        import numpy as np

        def _pearson(x, y):
            if len(x) < 2:
                return 0.0
            if np.std(x) == 0 or np.std(y) == 0:
                return 0.0
            return float(np.corrcoef(x, y)[0, 1])

        def _spearman(x, y):
            rx = np.argsort(np.argsort(x))
            ry = np.argsort(np.argsort(y))
            if np.std(rx) == 0 or np.std(ry) == 0:
                return 0.0
            return float(np.corrcoef(rx, ry)[0, 1])

        start_q, end_q = question_range
        results = {}

        for gen_name, gen_data in generators_data.items():
            vals = []
            in_range_count = 0
            negative_examples = []
            positive_qs = []  # ✅ record positively correlated questions

            for q in range(start_q, end_q):
                q_key = str(q)

                if q_key not in gen_data or q not in probs or gen_name not in probs[q]:
                    continue

                gen_conf = gen_data[q_key].get("answer_confidence", {})
                if not gen_conf:
                    continue
                gen_vals = list(gen_conf.values())

                prin_probs_raw = probs[q][gen_name]
                prin_vals = [p for _, p in prin_probs_raw.items()]

                m = min(len(gen_vals), len(prin_vals))
                if m < 2:
                    continue
                x, y = gen_vals[:m], prin_vals[:m]

                corr = _spearman(x, y) if method == "spearman" else _pearson(x, y)
                vals.append(corr)

                if 0.0 <= corr <= 1.0:
                    in_range_count += 1
                if corr > 0:
                    positive_qs.append(q)  # ✅ add to positives
                elif corr < 0 and len(negative_examples) < 1:
                    negative_examples.append((q, x, y, corr))

                if q == start_q:
                    print("=" * 60)
                    print(f"Example Q{q}, Generator={gen_name}")
                    print(f"Generator confidences: {x}")
                    print(f"Principal probabilities: {y}")
                    print(f"Correlation ({method}): {corr:.4f}")

            total = len(vals)
            percent_in_range = (in_range_count / total * 100) if total > 0 else 0.0

            results[gen_name] = {
                "mean_corr": float(np.mean(vals)) if vals else float("nan"),
                "percent_in_range": percent_in_range,
                "positive_qs": positive_qs  # ✅ now available downstream
            }

            if negative_examples:
                print(f"\n⚠️  Negative correlation examples for {gen_name}:")
                for (q, x, y, corr) in negative_examples:
                    print(f"  Q{q}: corr={corr:.4f}")
                    print(f"    Generator confidences: {x}")
                    print(f"    Principal probabilities: {y}")

        print("\n" + "=" * 60)
        print("Correlation Validity Summary:")
        for gen_name, stats in results.items():
            print(f"{gen_name}: mean_corr={stats['mean_corr']:.4f}, "
                  f"% corr in [0,1] = {stats['percent_in_range']:.1f}%, ")
                  #f"positively correlated={len(stats['positive_qs'])}")
        print("=" * 60 + "\n")

        return results

    def verify_chosen_answers(
        self,
        all_results: Dict[int, Dict],
        probs: Dict[int, Dict[str, Dict[str, float]]]
    ) -> Dict[str, Dict]:
        """
        Verify that each generator's chosen answer is consistent:
        it should NOT be strictly dominated by another answer on BOTH axes:
        (agent confidence and principal probability).

        Prints only a brief summary and up to the first 5 invalid examples per generator.
        """
        results = {}

        for q_idx, q_res in all_results.items():
            if "chosen_answers" not in q_res:
                continue

            for gen_name, chosen_info in q_res["chosen_answers"].items():
                chosen = chosen_info["answer"]

                # Skip if probs missing
                if q_idx not in probs or gen_name not in probs[q_idx]:
                    continue

                # Agent confidence distribution
                gen_conf = q_res["final_results"].get(gen_name, {}).get("all_confidences", {})
                if not gen_conf:
                    continue

                # Principal probability distribution (keys should be answer texts)
                prin_probs = probs[q_idx][gen_name]
                if not prin_probs:
                    continue

                # Init stats container
                if gen_name not in results:
                    results[gen_name] = {
                        "total_checked": 0,
                        "valid": 0,
                        "invalid": 0,
                        "invalid_examples": []  # (q_idx, chosen, dominating_ans, dom_conf, dom_prob, chosen_conf, chosen_prob)
                    }

                results[gen_name]["total_checked"] += 1

                chosen_conf = gen_conf.get(chosen, float("-inf"))
                chosen_prob = prin_probs.get(chosen, float("-inf"))

                # Check if any other answer dominates the chosen one
                invalid = False
                for ans in set(gen_conf.keys()).intersection(prin_probs.keys()):
                    if ans == chosen:
                        continue
                    if gen_conf[ans] > chosen_conf and prin_probs[ans] > chosen_prob:
                        invalid = True
                        if len(results[gen_name]["invalid_examples"]) < 5:
                            results[gen_name]["invalid_examples"].append(
                                (q_idx, chosen, ans, gen_conf[ans], prin_probs[ans], chosen_conf, chosen_prob)
                            )
                        break

                if invalid:
                    results[gen_name]["invalid"] += 1
                else:
                    results[gen_name]["valid"] += 1

        # Print compact summary (and only first five invalid cases)
        print("\n" + "=" * 60)
        print("Verification Summary:")
        for gen_name, stats in results.items():
            total = stats["total_checked"]
            valid = stats["valid"]
            percent_valid = (valid / total * 100) if total else 0.0
            print(f"Generator {gen_name}: Checked={total}, Valid={valid}, "
                  f"Invalid={stats['invalid']}, Valid%={percent_valid:.1f}%")
            if stats["invalid_examples"]:
                print("  First few invalid cases:")
                for ex in stats["invalid_examples"]:
                    q, chosen, dom_ans, dom_conf, dom_prob, ch_conf, ch_prob = ex
                    print(f"    Q{q}: chosen='{chosen}' vs dom='{dom_ans}' "
                          f"(dom_conf={dom_conf:.3f}, dom_prob={dom_prob:.3f}; "
                          f"chosen_conf={ch_conf:.3f}, chosen_prob={ch_prob:.3f})")
        print("=" * 60 + "\n")

        return results

    
    def calculate_r_star_accuracy(self, results: Dict) -> Dict:
        """
        R* method: Select the answer with the highest discriminator score.
        
        Args:
            results: Delegation results dictionary (same format as majority vote comparison)

        Returns:
            Dictionary with R* selection metrics
        """
        questions = [k for k in results.keys() if not str(k).startswith('_')]

        r_star_correct = 0
        total_questions = 0

        for question_idx in questions:
            question_result = results[question_idx]
            golden_answer = question_result.get("golden_answer", "Unknown")

            if golden_answer == "Unknown":
                continue

            final_results = question_result.get("final_results", {})
            if not final_results:
                continue

            # Step 1: compute discriminator scores for each generator
            scores = {}
            all_answers_paths = [
                (res["answer"], res.get("solution_path", ""), gen_name)
                for gen_name, res in final_results.items()
            ]

            for gen_name, res in final_results.items():
                score = self.discriminator_score_answer_path(
                    res["answer"],
                    res.get("solution_path", ""),
                    all_answers_paths=all_answers_paths
                )
                scores[gen_name] = score

            if not scores:
                continue

            # Step 2: pick generator(s) with max score (tie → random)
            max_score = max(scores.values())
            best_generators = [g for g, s in scores.items() if s == max_score]
            chosen_generator = random.choice(best_generators)
            chosen_answer = final_results[chosen_generator]["answer"]

            # Step 3: evaluate correctness
            if self.evaluator and hasattr(self.evaluator, "check_answers_equiv"):
                if self.evaluator.check_answers_equiv(golden_answer, chosen_answer):
                    r_star_correct += 1

            total_questions += 1

        return {
            "total_questions": total_questions,
            "r_star_correct": r_star_correct,
            "r_star_accuracy": r_star_correct / total_questions if total_questions else 0,
        }

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
        data = delegation.load_generator_data('./llm_answers_discriminator_GSM8K_0_298_qwen.json')
        print(f"✅ Loaded data for {len(data['results'])} generators")
      #  print(f"Generators: {', '.join(data['results'].keys())}")
        
        # Show data structure info
        first_gen = list(data['results'].keys())[0]
        
    except FileNotFoundError:
        print("❌ Error: ../llm_answers_discriminator_GSM8K_0_451_qwen.json not found!")
        print("Please run the extraction script first.")
        return
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Run delegation for all questions - automatically determine range from data
    # Get the actual number of questions from the data
    first_gen = list(data['results'].keys())[0]
    total_questions = len(data['results'][first_gen])
    test_questions = (0, total_questions)  # Process all questions in the data
    print(f"\n🧪 Running delegation for questions {test_questions[0]}-{test_questions[1]-1} (total: {total_questions} questions)")
    
    try:
        results = delegation.run_delegation_for_all_questions(
            data, 
            question_range=test_questions
        )
        
        probs = delegation.run_principal_for_all_questions(data['results'], question_range=test_questions)
        #print(f"Principal assigned probabilities: {probs}")
        correlations = delegation.compute_generator_principal_correlation(data['results'], question_range=test_questions, method = "pearson")
        print(correlations)
        pareto_optimal = delegation.verify_chosen_answers(results, probs) 

        r_star_accuracy = delegation.calculate_r_star_accuracy(results)
        print(f"🎯 R* Accuracy: {r_star_accuracy['r_star_accuracy']*100:.1f}%") 

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
        # print(f"\n📋 DETAILED RESULTS:")
        # for question_idx in sorted([k for k in results.keys() if not str(k).startswith('_')]):
        #     question_result = results[question_idx]
        #     print(f"\nQuestion {question_idx}:")
            
        #     if 'delegation_metrics' in question_result:
        #         metrics = question_result['delegation_metrics']
        #         print(f"  Golden Answer: {question_result.get('golden_answer', 'Unknown')}")
        #         print(f"  Correct Answers: {metrics['correct_answers']}/{metrics['total_generators']}")
        #         print(f"  Average Confidence Change: {metrics['average_confidence_change']:.4f}")
            
        #     for gen_name, gen_result in question_result['final_results'].items():
        #         confidence_change = gen_result['confidence_change']
        #         change_symbol = "↗️" if confidence_change > 0 else "↘️" if confidence_change < 0 else "➡️"
        #         initial_conf = gen_result.get('initial_confidence', 0.0)
        #         final_conf = gen_result.get('final_confidence', 0.0)
        #         print(f"  {gen_name}: {change_symbol} {final_conf:.3f} "
        #               f"(initial: {initial_conf:.3f}, change: {confidence_change:+.3f})")
        
        # Calculate and show discriminator accuracy summary
        print(f"\n🎯 DISCRIMINATOR ACCURACY SUMMARY:")
        print("=" * 60)
        
        questions_with_disagreement = 0
        questions_with_agreement = 0
        correct_discriminator_choices = 0
        discriminator_accuracy_by_model = {}
        
        # First, count agreement vs disagreement questions
        for question_idx in sorted([k for k in results.keys() if not str(k).startswith('_')]):
            question_result = results[question_idx]
            if question_result.get('had_disagreement', False):
                questions_with_disagreement += 1
            else:
                questions_with_agreement += 1
        
        print(f"📊 Questions with Agreement (skipped delegation): {questions_with_agreement}")
        print(f"📊 Questions with Disagreement (ran delegation): {questions_with_disagreement}")
        print(f"📊 Total Questions: {questions_with_agreement + questions_with_disagreement}")
        
        # Now calculate discriminator accuracy only for disagreement questions
        for question_idx in sorted([k for k in results.keys() if not str(k).startswith('_')]):
            question_result = results[question_idx]
            golden_answer = question_result.get('golden_answer', 'Unknown')
            
            if golden_answer == 'Unknown':
                continue
            
            # Only count questions where generators disagreed (where discriminator was used)
            if not question_result.get('had_disagreement', False):
                continue
            
            # Find the answer with highest discriminator survival rate (discriminator's choice)
            best_answer = None
            best_survival_rate = -float('inf')
            best_generator = None
            
            # Get the last iteration's survival rates for each generator
            if 'iterations' in question_result and question_result['iterations']:
                last_iteration = question_result['iterations'][-1]
                if 'survival_scores' in last_iteration:
                    survival_rates = last_iteration['survival_scores']
                    
                    # Find generator with highest survival rate
                    for gen_name, survival_rate in survival_rates.items():
                        if survival_rate > best_survival_rate:
                            best_survival_rate = survival_rate
                            best_generator = gen_name
                            # Get the answer from this generator
                            if gen_name in question_result['final_results']:
                                best_answer = question_result['final_results'][gen_name]['answer']
            
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
        if questions_with_disagreement > 0:
            overall_accuracy = (correct_discriminator_choices / questions_with_disagreement) * 100
            print(f"📊 Overall Discriminator Accuracy: {overall_accuracy:.1f}% ({correct_discriminator_choices}/{questions_with_disagreement})")
        else:
            print(f"📊 Overall Discriminator Accuracy: No disagreement questions found")
        
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
        
        # for question_idx in all_questions:
        #     if str(question_idx) in results:
        #         question_result = results[str(question_idx)]
        #         print(f"\n🔍 Question {question_idx}:")
                
        #         # Show golden answer if available
        #         golden_answer = question_result.get('golden_answer', 'Unknown')
        #         print(f"   Golden Answer: {golden_answer}")
                
        #         # Show results for each model in a compact format
        #         if 'final_results' in question_result:
        #             for gen_name, gen_result in question_result['final_results'].items():
        #                 initial_conf = gen_result.get('initial_confidence', 0.0)
        #                 final_conf = gen_result.get('final_confidence', 0.0)
        #                 confidence_change = gen_result.get('confidence_change', 0.0)
                        
        #                 change_symbol = "↗️" if confidence_change > 0 else "↘️" if confidence_change < 0 else "➡️"
                        
        #                 print(f"   {gen_name}: {change_symbol} {final_conf:.3f} (initial: {initial_conf:.3f}, change: {confidence_change:+.3f})")
                
        #         # Show last iteration's discriminator survival scores in a compact format
        #         if 'iterations' in question_result and question_result['iterations']:
        #             last_iteration = question_result['iterations'][-1]
        #             print(f"   🎯 Discriminator Scores:")
                    
        #             # Show normalized scores (these sum to 0)
        #             if 'normalized_scores' in last_iteration:
        #                 norm_scores = [f"{gen_name}: {norm_score:+.3f}" for gen_name, norm_score in last_iteration['normalized_scores'].items()]
        #                 print(f"     Normalized: {', '.join(norm_scores)}")
                    
        #             # Show survival rates for each generator
        #             if 'survival_scores' in last_iteration:
        #                 surv_rates = [f"'{gen}': {rate:+.3f}" for gen, rate in last_iteration['survival_scores'].items()]
        #                 print(f"     Survival: {', '.join(surv_rates)}")
                
        #         # Show delegation metrics if available
        #         if 'delegation_metrics' in question_result:
        #             metrics = question_result['delegation_metrics']
        #             print(f"   📊 Correct: {metrics.get('correct_answers', 0)}/{metrics.get('total_generators', 0)}, Avg Change: {metrics.get('average_confidence_change', 0.0):.3f}")
        
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