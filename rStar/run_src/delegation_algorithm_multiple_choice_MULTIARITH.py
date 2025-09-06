#!/usr/bin/env python3
"""
Delegation Algorithm Implementation with Multiple Choice Discriminator

This script implements a collaborative learning algorithm where multiple LLM generators
work together through a multiple choice discriminator to improve their answer confidence scores.

Algorithm Overview:
1. For each question, each generator selects their best answer and solution path
2. All answers are formatted as multiple choice options for the discriminator
3. The discriminator selects the most probable correct answer from the choices
4. Survival rates are calculated and normalized to sum to 0
5. Generators update their confidence based on normalized survival rates
6. Process repeats for multiple iterations (learning rate)
7. Final answers are determined based on updated confidences
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import copy
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
    from eval_src.Evaluator import MULTIARITHEvaluator  # Import the specific evaluator
    DISCRIMINATOR_AVAILABLE = True
    print("✅ Evaluator components imported successfully")
except ImportError as e:
    print(f"Warning: Evaluator components not available: {e}")
    DISCRIMINATOR_AVAILABLE = False
    
class DelegationAlgorithm:
    def __init__(self, learning_rate: float = 0.1, max_iterations: int = 10):
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
                # Create MULTIARITH evaluator for arithmetic problems
                self.evaluator = MULTIARITHEvaluator()
                print("✅ MULTIARITH Evaluator loaded successfully")
            except ImportError:
                print("⚠️  MULTIARITH Evaluator not available, using fallback")
        
        # Load MULTIARITH questions for context
        self.question_contexts = self.load_multiarith_questions()
        
        # Initialize discriminator model for multiple choice approach
        self.discriminator_model = None
        self.discriminator_tokenizer = None
        if DISCRIMINATOR_AVAILABLE:
            try:
                # Load a HuggingFace discriminator model using your existing API
                from models.HuggingFace_API import load_HF_model
                
                # Use a smaller, efficient HuggingFace model for discriminator
                model_name = "Llama3.1-8B-Instruct"  # ~345M parameters, fast and efficient
                print(f"🔧 Loading {model_name} as discriminator using HuggingFace API...")
                
                self.discriminator_tokenizer, self.discriminator_model = load_HF_model(
                    model_name, 
                    hf_token= "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" # ""  # Public model
                )
                
                # Fix padding token issue for GPT-2 based models
                if self.discriminator_tokenizer.pad_token is None:
                    self.discriminator_tokenizer.pad_token = self.discriminator_tokenizer.eos_token
                
                print("✅ HuggingFace discriminator model loaded successfully")
            except Exception as e:
                print(f"⚠️  Failed to load HuggingFace discriminator model: {e}")
                print("   Will use fallback scoring method")
        
        # Initialize cache for discriminator results
        self.discriminator_cache = {}  # question_idx -> {candidate_hash -> survival_rates}
    
    def load_generator_data(self, data_file: str) -> Dict:
        """Load the generator data from the JSON file."""
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Validate data structure
        if 'results' not in data:
            raise ValueError("Data file must contain 'results' key")
        
        return data
    
    def load_multiarith_questions(self, questions_file: str = "./data/MULTIARITH/test_all.json") -> Dict[int, str]:
        """
        Load MULTIARITH questions to get context for the discriminator.
        
        Args:
            questions_file: Path to MULTIARITH test questions file
            
        Returns:
            Dictionary mapping question index to question text
        """
        try:
            with open(questions_file, 'r') as f:
                questions_data = json.load(f)
            
            # Create mapping from question index to question text
            question_contexts = {}
            
            # Method 1: Array index mapping (primary method)
            for i, item in enumerate(questions_data):
                if 'problem' in item:
                    question_contexts[i] = item['problem']
            
            # Method 2: ID field mapping (backup method for any mismatches)
            for item in questions_data:
                if 'id' in item and 'problem' in item:
                    q_id = item['id']
                    if q_id not in question_contexts:  # Don't overwrite array index
                        question_contexts[q_id] = item['problem']
            
            print(f"✅ Loaded {len(question_contexts)} MULTIARITH questions for context")
            print(f"📊 Available question indices: {sorted(list(question_contexts.keys()))[:10]}...")
            
            # Verify we have the questions we need (0-179 for MULTIARITH)
            missing_questions = []
            for i in range(180):
                if i not in question_contexts:
                    missing_questions.append(i)
            
            if missing_questions:
                print(f"⚠️  Warning: Missing questions: {missing_questions[:10]}...")
            else:
                print(f"✅ All questions 0-179 are available")
            
            return question_contexts
            
        except FileNotFoundError:
            print(f"⚠️  Warning: MULTIARITH questions file not found at {questions_file}")
            print("   Will use empty question context")
            return {}
        except Exception as e:
            print(f"⚠️  Warning: Error loading MULTIARITH questions: {e}")
            print("   Will use empty question context")
            return {}
    
    def _create_candidate_hash(self, candidates: List[Tuple[str, str, float]]) -> str:
        """
        Create a hash of candidates for caching discriminator results.
        
        Args:
            candidates: List of (answer, path, confidence) tuples
            
        Returns:
            Hash string representing the candidate set
        """
        # Sort candidates by answer to ensure consistent hashing
        sorted_candidates = sorted(candidates, key=lambda x: x[0])
        
        # Create hash from answers AND confidence scores to detect changes in confidence distributions
        candidate_string = "|".join([f"{c[0]}:{c[2]:.6f}" for c in sorted_candidates])
        
        # Use simple hash for efficiency
        return str(hash(candidate_string))
    
    def _candidates_changed(self, question_idx: int, current_candidates: List[Tuple[str, str, float]]) -> bool:
        """
        Check if candidates have changed for a question since last iteration.
        
        Args:
            question_idx: Question index
            current_candidates: Current list of candidates
            
        Returns:
            True if candidates changed, False if they're the same
        """
        if question_idx not in self.discriminator_cache:
            return True  # First time seeing this question
        
        current_hash = self._create_candidate_hash(current_candidates)
        
        # Check if we have this exact candidate set cached
        return current_hash not in self.discriminator_cache[question_idx]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the discriminator cache.
        
        Returns:
            Dictionary with cache statistics
        """
        total_questions = len(self.discriminator_cache)
        total_entries = sum(len(cache) for cache in self.discriminator_cache.values())
        
        return {
            'total_questions_cached': total_questions,
            'total_entries': total_entries,
            'cache_size_mb': total_entries * 0.001,  # Rough estimate
            'questions': list(self.discriminator_cache.keys())
        }
    
    def clear_cache(self, question_idx: int = None):
        """
        Clear the discriminator cache.
        
        Args:
            question_idx: Specific question to clear, or None to clear all
        """
        if question_idx is None:
            self.discriminator_cache.clear()
            print("🗑️  Cleared entire discriminator cache")
        elif question_idx in self.discriminator_cache:
            del self.discriminator_cache[question_idx]
            print(f"🗑️  Cleared cache for question {question_idx}")
        else:
            print(f"ℹ️  No cache found for question {question_idx}")
    
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
        
        # Get confidence scores and use deterministic selection based on weights
        confidence = question_data['answer_confidence']
        weights = [float(i) for i in confidence.values()]
        # best_answer = random.choices(list(confidence.keys()), weights=weights, k=1)[0]
        best_answer = max(confidence.keys(), key=lambda x: confidence[x])
        # Get the solution path for this answer
        solution_path = question_data['answer_to_solution_path'].get(best_answer, "")
        
        return best_answer, solution_path, confidence
    
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
    
    
    def normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to range [0, 1] using min-max normalization.
        
        Args:
            scores: List of raw scores
            
        Returns:
            List of normalized scores in range [0, 1]
        """
        if not scores:
            return scores
            
        # Convert to numpy array for easier manipulation
        scores_array = np.array(scores, dtype=float)
        min_score = np.min(scores_array)
        max_score = np.max(scores_array)
        
        # Handle case where all scores are the same
        if max_score == min_score:
            return [0.5] * len(scores)  # All equal scores get 0.5
        
        # Apply min-max normalization: (x - min) / (max - min)
        normalized = (scores_array - min_score) / (max_score - min_score)
        
        return normalized.tolist()
    
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
        confidence_multiplier = math.exp(learning_rate * cumulative_survival_rate * current_confidence)
        
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
                    # print(f" Model: {gen_name}, 🔍 Answer: {answer}, Confidence: {confidence_dict}")


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
                            
                except Exception as e:
                    print(f"Warning: Error processing generator {gen_name} for question {question_idx}: {e}")
                    continue
        
        if not initial_data:
            print(f"❌ No valid data found for question {question_idx}")
            return question_results
        
        # Initialize cumulative survival rates for each unique answer
        cumulative_survival_rates = defaultdict(float)  # answer -> cumulative normalized survival rate
        
        # Run multiple iterations
        for iteration in range(self.max_iterations):
            # print(f"  📊 Iteration {iteration + 1}/{self.max_iterations}")
            
            # Check if candidates have changed since last iteration
            candidates_changed = self._candidates_changed(question_idx, all_answers_paths)
            if not candidates_changed and iteration > 0:
            # print(f"  🔄 Iteration {iteration + 1}: Candidates unchanged, using cached results")
                pass
            else:
                print(f"  🔄 Iteration {iteration + 1}: Candidates changed, calling discriminator")
                # pass
                
            
            # Get discriminator scores using multiple choice approach
            if self.discriminator_model and self.discriminator_tokenizer:
                # Use the new multiple choice discriminator with actual question context
                question_context = self.question_contexts.get(question_idx, "")
                if question_context.strip():
                    pass
                    # print(f"  📝 Using question context: {question_context[:100]}{'...' if len(question_context) > 100 else ''}")
                else:
                    print(f"  ⚠️  No question context available for question {question_idx}")
                    print(f"  🔍 Available question indices: {sorted(list(self.question_contexts.keys()))[:10]}...")
                    print(f"  🔍 Looking for question {question_idx}")
                
                survival_rates = self.calculate_survival_scores_multiple_choice(
                    question_context=question_context,  # Use actual question context
                    candidates=all_answers_paths,
                    discriminator_model=self.discriminator_model,
                    discriminator_tokenizer=self.discriminator_tokenizer,
                    question_idx=question_idx # Pass question_idx for caching
                )
                
                # Convert survival rates to scores for each generator
                survival_scores = []
                for gen_name in generator_names:
                    if gen_name in initial_data:
                        answer = initial_data[gen_name]['answer']
                        score = survival_rates.get(answer, 0.0)
                        survival_scores.append(score)
                    else:
                        survival_scores.append(0.0)
            else:
                # Fallback: use simple scoring
                survival_scores = []
                for gen_name in generator_names:
                    if gen_name in initial_data:
                        score = self.discriminator_score_answer_path(
                            initial_data[gen_name]['answer'], 
                            initial_data[gen_name]['solution_path']
                        )
                        survival_scores.append(score)
                    else:
                        survival_scores.append(0.0)
            
            # Normalize scores to sum to 0
            # normalized_scores = self.normalize_scores(survival_scores)
            normalized_scores = survival_scores
            # Update cumulative survival rates for each unique answer
            for i, gen_name in enumerate(generator_names):
                if gen_name in initial_data:
                    answer = initial_data[gen_name]['answer']
                    cumulative_survival_rates[answer] += normalized_scores[i]
            
            
            # Update generator confidences using cumulative survival rates of their answers
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
                    
                    # if iteration < 3:  # Debug output for first few iterations
                    #     print(f"    🔄 {gen_name} renormalized confidences: {dict(all_confidences)}")
            
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
        
        # Store final results - RE-SELECT best answer based on updated confidences
        question_results['final_results'] = {}
        for gen_name, data in initial_data.items():
            # Find the answer with highest confidence AFTER delegation
            updated_confidences = data['all_confidences']
            best_answer_after = max(updated_confidences.keys(), key=lambda x: updated_confidences[x])
            
            # Get the solution path for the newly selected answer
            best_solution_path_after = data.get('solution_path', '')  # Keep original path for now
            
            question_results['final_results'][gen_name] = {
                'answer': best_answer_after,  # Use the re-selected answer based on updated confidences
                'solution_path': best_solution_path_after,
                'original_answer': data['answer'],  # Keep track of original answer
                'original_confidence': data['original_confidence'],
                'final_confidence': updated_confidences[best_answer_after],
                'confidence_change': updated_confidences[best_answer_after] - data['original_confidence'],
                'all_confidences': data['all_confidences']
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
                'correct_answers_before': 0,  # New: correct answers before delegation
                'correct_answers_after': 0,   # New: correct answers after delegation
                'total_confidence_change': 0.0,
                'accuracy_before': 0.0,      # New: accuracy before delegation
                'accuracy_after': 0.0        # New: accuracy after delegation
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
                            # Check if answer was correct AFTER delegation (re-selected answer based on updated confidences)
                            if self._check_answers_equiv(result['answer'], golden_answer):
                                gen_metrics['correct_answers_after'] += 1
                                overall_metrics['total_correct_answers'] += 1
                            
                            # Check if answer was correct BEFORE delegation (original answer)
                            original_answer = result.get('original_answer', '')
                            if original_answer and self._check_answers_equiv(original_answer, golden_answer):
                                gen_metrics['correct_answers_before'] += 1
                
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
            
            # Calculate original majority vote (using the original answers from delegation results)
            original_answers = {}
            for gen_name, result in question_result.get('final_results', {}).items():
                original_answer = result.get('original_answer', '')
                if original_answer:
                    if original_answer in original_answers:
                        original_answers[original_answer] += 1
                    else:
                        original_answers[original_answer] = 1
            
            if original_answers:
                original_majority_answer = max(original_answers.keys(), key=lambda x: original_answers[x])
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
                delegation_majority_answer = max(delegation_answers.keys(), key=lambda x: delegation_answers[x])
                if self.evaluator and hasattr(self.evaluator, 'check_answers_equiv'):
                    if self.evaluator.check_answers_equiv(golden_answer, delegation_majority_answer):
                        delegation_majority_correct += 1
            
            # Calculate discriminator accuracy (how often discriminator's best choice matches golden answer)
            if question_idx in results and 'iterations' in results[question_idx]:
                # Get the last iteration's cumulative survival rates to find discriminator's best choice
                last_iteration = results[question_idx]['iterations'][-1] if results[question_idx]['iterations'] else None
                if last_iteration and 'cumulative_survival_rates' in last_iteration:
                    cumulative_survival_rates = last_iteration['cumulative_survival_rates']
                    if cumulative_survival_rates:
                        # Find the answer with highest cumulative survival rate (discriminator's choice)
                        discriminator_best_answer = max(cumulative_survival_rates.keys(), key=lambda x: cumulative_survival_rates[x])
                        if self.evaluator and hasattr(self.evaluator, 'check_answers_equiv'):
                            if self.evaluator.check_answers_equiv(golden_answer, discriminator_best_answer):
                                discriminator_correct += 1
                            # Debug output for first few questions
                            if discriminator_correct <= 3:
                                print(f"    🎯 Q{question_idx}: Discriminator chose '{discriminator_best_answer}' vs Golden '{golden_answer}' -> {'✅' if self.evaluator.check_answers_equiv(golden_answer, discriminator_best_answer) else '❌'}")
        
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
            
            # Add discriminator accuracy
            report.append("")
            report.append("🎯 DISCRIMINATOR ACCURACY")
            report.append("-" * 30)
            report.append(f"Discriminator Best Choice:  {majority_metrics['discriminator_correct']:>3}/{majority_metrics['total_questions']:<3} ({majority_metrics['discriminator_accuracy']*100:>5.1f}%)")
            
            # Compare discriminator vs majority vote
            if majority_metrics['discriminator_accuracy'] > majority_metrics['original_majority_accuracy']:
                report.append(f"🏆 Discriminator outperforms original majority vote!")
            elif majority_metrics['discriminator_accuracy'] < majority_metrics['original_majority_accuracy']:
                report.append(f"⚠️  Discriminator underperforms original majority vote")
            else:
                report.append(f"➡️  Discriminator matches original majority vote")
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

    def build_generator_prompt(self, subject, target_question, target_choices, get_correct):
        """Build prompt for multiple choice question."""
        prompt = "The following are multiple choice questions (answers) about {}.\n\n".format(
            subject)

        prompt += f"{target_question}"
        for i, c in enumerate(target_choices):
            prompt += "\n{}".format(c)
            
        if get_correct:
            prompt += "\nAnswer:"
        else:
            prompt += "\nIncorrect Answer:"
        return prompt

    def get_generator_answer_probs(self, model, tokenizer, prompt_text, choices_list):
        """Get probability scores for each choice using the HuggingFace discriminator model."""
        import torch
        
        try:
            # Ensure model is in evaluation mode
            model.eval()
            
            # Tokenize the prompt
            input_ids = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True)
            
            # Move to the same device as the model
            if hasattr(model, 'device'):
                device = model.device
            else:
                # Try to get device from model parameters
                device = next(model.parameters()).device
            
            input_ids = {k: v.to(device) for k, v in input_ids.items()}
            
            # Get logits from the model
            with torch.no_grad():
                outputs = model(**input_ids)
                logits = outputs.logits[0, -1]  # Last token logits
            
            # Get probabilities for each choice (A, B, C, etc.)
            choices = [f"{chr(65+i)}" for i in range(len(choices_list))]
            choice_probs = {}
            
            for i, choice in enumerate(choices):
                try:
                    # Get token ID for the choice letter
                    choice_tokens = tokenizer(choice, return_tensors="pt", add_special_tokens=False)
                    token_id = choice_tokens.input_ids[0, 0].item()  # First token of the choice
                    
                    # Get probability for that token using softmax
                    prob = torch.softmax(logits, dim=0)[token_id].item()
                    choice_probs[choice] = prob
                    
                except Exception as e:
                    print(f"    ⚠️  Error processing choice {choice}: {e}")
                    # Fallback: equal probability
                    choice_probs[choice] = 1.0 / len(choices_list)
            
            # If we got valid probabilities, normalize them
            if choice_probs and any(prob > 0 for prob in choice_probs.values()):
                # Normalize to sum to 1
                total_prob = sum(choice_probs.values())
                choice_probs = {choice: prob/total_prob for choice, prob in choice_probs.items()}
                
                print(f"    🎯 Real discriminator probabilities: {choice_probs}")
                return choice_probs
            else:
                # Fallback: equal probabilities
                equal_prob = 1.0 / len(choices_list)
                return {f"{chr(65+i)}": equal_prob for i in range(len(choices_list))}
            
        except Exception as e:
            print(f"    ⚠️  HuggingFace discriminator scoring failed: {e}")
            # Fallback: equal probabilities
            return {f"{chr(65+i)}": 1.0/len(choices_list) for i in range(len(choices_list))}

    def calculate_survival_scores_multiple_choice(self, question_context: str, candidates: List[Tuple[str, str, float]], 
                                                discriminator_model, discriminator_tokenizer, question_idx: int = None) -> Dict[str, float]:
        """
        Calculate survival scores using multiple choice selection approach.
        
        Args:
            question_context: The question text
            candidates: List of (answer, path, confidence) tuples
            discriminator_model: The discriminator model
            discriminator_tokenizer: The discriminator tokenizer
            question_idx: Question index for caching (optional)
            
        Returns:
            Dictionary mapping answer to survival rate (normalized to sum to 0)
        """
        # Check cache first if question_idx is provided
        if question_idx is not None:
            candidate_hash = self._create_candidate_hash(candidates)
            
            if question_idx in self.discriminator_cache:
                if candidate_hash in self.discriminator_cache[question_idx]:
                    # print(f"  🚀 Cache hit! Using cached survival rates for question {question_idx}")
                    return self.discriminator_cache[question_idx][candidate_hash]
                else:
                    # Initialize cache for this question if not exists
                    if question_idx not in self.discriminator_cache:
                        self.discriminator_cache[question_idx] = {}
            else:
                # Initialize cache for this question
                self.discriminator_cache[question_idx] = {}
        
        try:
            # Extract unique answers (remove duplicates)
            unique_answers = list(dict.fromkeys([c[0] for c in candidates]))
            
            if len(unique_answers) < 2:
                # If only one answer, give it survival rate 0 (neutral)
                result = {unique_answers[0]: 0.0}
            else:
                # Create multiple choice format
                choices = [f"{chr(65+i)}. {answer}" for i, answer in enumerate(unique_answers)]
                
                # Build prompt asking which answer is most correct
                if question_context.strip():
                    prompt = f"The following are multiple choice questions (with answers) about Math.\n"
                    prompt += f"Question: {question_context}\n\nWhich of the following answers is most correct?\n"
                else:
                    prompt = "Which of the following answers is most correct?\n"
                
                for choice in choices:
                    prompt += f"{choice}\n"
                prompt += "Answer:"
            
                choice_probs = self.get_generator_answer_probs(
                    discriminator_model, 
                    discriminator_tokenizer, 
                    prompt, 
                    unique_answers
                )
                
                # Convert probabilities to survival rates and normalize to sum to 0
                if len(unique_answers) > 0:
                    # Map choice probabilities back to answers
                    answer_probs = {}
                    for i, answer in enumerate(unique_answers):
                        choice_letter = f"{chr(65+i)}"
                        answer_probs[answer] = choice_probs.get(choice_letter, 0.0)
                    
                    # First normalize probabilities to sum to 1
                    total_prob = sum(answer_probs.values())
                    if total_prob > 0:
                        normalized_probs = {answer: prob / total_prob for answer, prob in answer_probs.items()}
                    else:
                        # Fallback: equal probabilities
                        normalized_probs = {answer: 1.0 / len(unique_answers) for answer in unique_answers}
                    
                    # Convert to survival rates that sum to 0
                    # Method: subtract the mean from each probability
                    mean_prob = 1.0 / len(unique_answers)
                    survival_rates = {answer: prob - mean_prob for answer, prob in normalized_probs.items()}
                    
                    # Verify they sum to 0 (with small tolerance for floating point)
                    total_survival = sum(survival_rates.values())
                    if abs(total_survival) > 1e-6:
                        # Adjust to ensure exact sum to 0
                        adjustment = total_survival / len(unique_answers)
                        survival_rates = {answer: rate - adjustment for answer, rate in survival_rates.items()}
                else:
                    survival_rates = {}
                
                result = survival_rates
            
            # Cache the result if question_idx is provided
            if question_idx is not None:
                candidate_hash = self._create_candidate_hash(candidates)
                self.discriminator_cache[question_idx][candidate_hash] = result
                # print(f"  💾 Cached survival rates for question {question_idx}")
            
            return result
            
        except Exception as e:
            print(f"  ⚠️  Multiple choice discriminator failed: {e}")
            # Fallback: equal survival rates that sum to 0
            unique_answers = list(dict.fromkeys([c[0] for c in candidates]))
            if len(unique_answers) > 0:
                mean_rate = 0.0  # All rates are 0, so sum is 0
                result = {answer: mean_rate for answer in unique_answers}
            else:
                result = {}
            
            # Cache the fallback result if question_idx is provided
            if question_idx is not None:
                candidate_hash = self._create_candidate_hash(candidates)
                self.discriminator_cache[question_idx][candidate_hash] = result
            
            return result

    def discriminator_score_answer_path(self, answer: str, solution_path: str, 
                                      question_context: str = "", 
                                      discriminator_model=None, 
                                      discriminator_tokenizer=None) -> float:
        """
        Score an answer using the multiple choice discriminator approach.
        
        Args:
            answer: The answer to score
            solution_path: The solution path (not used in multiple choice approach)
            question_context: The question context
            discriminator_model: The discriminator model
            discriminator_tokenizer: The discriminator tokenizer
            
        Returns:
            Survival rate score for this answer
        """
        # Simple discriminator logic: score based on answer characteristics
        try:
            # Convert answer to float for numerical scoring
            answer_value = float(answer)
            
            # Score based on answer characteristics
            if answer_value == 0:
                return 0.3  # Zero answers often indicate errors
            elif answer_value < 0:
                return 0.4  # Negative answers often indicate errors
            elif answer_value > 1000000:
                return 0.6  # Very large numbers might be errors
            elif answer_value == int(answer_value):
                return 0.8  # Integer answers are often correct
            else:
                return 0.7  # Decimal answers
                
        except (ValueError, TypeError):
            # Non-numeric answers get lower scores
            if answer.lower() in ['true', 'false', 'yes', 'no']:
                return 0.6  # Boolean answers
            elif len(answer) < 10:
                return 0.5  # Very short answers
            else:
                return 0.4  # Other non-numeric answers


def main():
    """Main function to run the delegation algorithm."""
    print("🎯 Delegation Algorithm Implementation")
    
    # Initialize the algorithm
    delegation = DelegationAlgorithm(
        learning_rate=0.05,
        max_iterations=20
    )
    
    # Show question context status (minimal)
    if not delegation.question_contexts:
        print("⚠️  No question context available - will use empty context")
    
    # Load generator data
    try:
        data = delegation.load_generator_data('./llm_answers_discriminator_0_299_MULTIARITH.json')
        print(f"✅ Loaded data for {len(data['results'])} generators")
      #  print(f"Generators: {', '.join(data['results'].keys())}")
        
        # Show data structure info
        first_gen = list(data['results'].keys())[0]
        
    except FileNotFoundError:
        print("❌ Error: ./llm_answers_discriminator_0_299_MULTIARITH.json not found!")
        print("Please run the extraction script first.")
        return
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Run delegation for questions 0-180 (MULTIARITH range)
    test_questions = (0, 180)  # Process questions 0-179
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
        
        # Show summary of results (much less verbose)
        print(f"\n📋 RESULTS SUMMARY:")
        total_questions = len([k for k in results.keys() if not str(k).startswith('_')])
        print(f"  Total questions processed: {total_questions}")
        
        # Show a few sample questions for verification
        sample_questions = sorted([k for k in results.keys() if not str(k).startswith('_')])[:3]
        print(f"  Sample questions: {', '.join(map(str, sample_questions))}")
        
        # Show overall statistics
        if results:
            first_question = next(k for k in results.keys() if not str(k).startswith('_'))
            if 'delegation_metrics' in results[first_question]:
                print(f"  All results saved with detailed metrics")
        
        print(f"\n🎉 Delegation algorithm completed successfully!")
        print(f"📁 Results saved to: {output_filename}")
        
        # Show cache statistics (minimal)
        cache_stats = delegation.get_cache_stats()
        if cache_stats['total_entries'] > 0:
            print(f"\n📊 Cache: {cache_stats['total_entries']} entries, {cache_stats['cache_size_mb']:.3f} MB")
        
    except KeyboardInterrupt:
        print("\n⚠️  Delegation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during delegation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 