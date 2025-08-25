#!/usr/bin/env python3
"""
Model Comparison Analysis Tool

This script analyzes the differences and accuracy among the three models:
- Qwen2_7B_instruct
- Llama3_8B_instruct  
- Mistral_7B_instruct

It calculates:
1. Individual model accuracy
2. Pairwise overlap (same answers)
3. Accuracy for overlapping answers
4. Disagreement analysis
5. Consensus analysis
"""

import json
import os
import sys
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import numpy as np

def load_generator_data(data_file: str = "./llm_answers_discriminator_0_99_qwen.json") -> dict:
    """
    Load generator data from JSON file.
    
    Args:
        data_file: Path to generator data file
        
    Returns:
        Dictionary with generator data
    """
    print(f"🔍 Loading generator data from: {data_file}")
    
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        print(f"✅ Successfully loaded generator data file")
        print(f"📊 Generators: {list(data['results'].keys())}")
        
        return data
        
    except FileNotFoundError:
        print(f"❌ Error: File not found at {data_file}")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def get_best_answer_for_question(generator_data: dict, question_idx: int) -> Tuple[str, float]:
    """
    Get the best answer (highest probability) for a specific question from a generator.
    
    Args:
        generator_data: Data for one generator
        question_idx: Question index
        
    Returns:
        Tuple of (answer, confidence)
    """
    try:
        question_data = generator_data[str(question_idx)]
        
        # Get confidence scores and select highest
        confidence = question_data['answer_confidence']
        best_answer = max(confidence.keys(), key=lambda x: confidence[x])
        best_confidence = confidence[best_answer]
        
        return best_answer, best_confidence
        
    except Exception as e:
        print(f"    Warning: Error getting best answer for question {question_idx}: {e}")
        return None, 0.0

def analyze_individual_model_accuracy(generators_data: dict) -> dict:
    """
    Analyze individual model accuracy.
    
    Args:
        generators_data: Data from all generators
        
    Returns:
        Dictionary with accuracy metrics for each model
    """
    print(f"\n📊 ANALYZING INDIVIDUAL MODEL ACCURACY")
    print("=" * 50)
    
    model_accuracy = {}
    
    for gen_name, gen_data in generators_data.items():
        print(f"\n🔍 Analyzing {gen_name}:")
        
        correct_answers = 0
        total_questions = 0
        confidence_scores = []
        
        for question_idx in range(100):  # Questions 0-99
            if str(question_idx) in gen_data:
                try:
                    # Get best answer and golden answer
                    best_answer, confidence = get_best_answer_for_question(gen_data, question_idx)
                    golden_answer = gen_data[str(question_idx)].get('golden_answer', 'Unknown')
                    
                    if golden_answer != 'Unknown' and best_answer:
                        total_questions += 1
                        confidence_scores.append(confidence)
                        
                        # Check if answer is correct (simple numerical comparison)
                        if check_answer_correct(best_answer, golden_answer):
                            correct_answers += 1
                            
                except Exception as e:
                    continue
        
        if total_questions > 0:
            accuracy = correct_answers / total_questions
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            
            model_accuracy[gen_name] = {
                'correct_answers': correct_answers,
                'total_questions': total_questions,
                'accuracy': accuracy,
                'avg_confidence': avg_confidence
            }
            
            print(f"  ✅ Correct Answers: {correct_answers}/{total_questions}")
            print(f"  📈 Accuracy: {accuracy*100:.1f}%")
            print(f"  🎯 Average Confidence: {avg_confidence:.3f}")
        else:
            print(f"  ❌ No valid questions found")
    
    return model_accuracy

def analyze_pairwise_overlap(generators_data: dict) -> dict:
    """
    Analyze pairwise overlap between models.
    
    Args:
        generators_data: Data from all generators
        
    Returns:
        Dictionary with overlap analysis for each pair
    """
    print(f"\n🔗 ANALYZING PAIRWISE OVERLAP")
    print("=" * 50)
    
    generator_names = list(generators_data.keys())
    pairwise_overlap = {}
    
    # Analyze each pair of models
    for i, gen1 in enumerate(generator_names):
        for j, gen2 in enumerate(generator_names[i+1:], i+1):
            pair_name = f"{gen1} vs {gen2}"
            print(f"\n🔍 Analyzing {pair_name}:")
            
            same_answers = 0
            different_answers = 0
            overlap_correct = 0
            overlap_total = 0
            
            for question_idx in range(100):
                if str(question_idx) in generators_data[gen1] and str(question_idx) in generators_data[gen2]:
                    try:
                        # Get best answers from both models
                        answer1, conf1 = get_best_answer_for_question(generators_data[gen1], question_idx)
                        answer2, conf2 = get_best_answer_for_question(generators_data[gen2], question_idx)
                        golden_answer = generators_data[gen1][str(question_idx)].get('golden_answer', 'Unknown')
                        
                        if answer1 and answer2 and golden_answer != 'Unknown':
                            if check_answer_correct(answer1, golden_answer) and check_answer_correct(answer2, golden_answer):
                                # Both answers are correct
                                if answer1 == answer2:
                                    same_answers += 1
                                    overlap_correct += 1
                                    overlap_total += 1
                                else:
                                    different_answers += 1
                                    overlap_total += 1
                            elif answer1 == answer2:
                                # Same answer but not necessarily correct
                                same_answers += 1
                                if check_answer_correct(answer1, golden_answer):
                                    overlap_correct += 1
                                overlap_total += 1
                            else:
                                different_answers += 1
                                overlap_total += 1
                                
                    except Exception as e:
                        continue
            
            total_questions = same_answers + different_answers
            
            if total_questions > 0:
                overlap_percentage = (same_answers / total_questions) * 100
                overlap_accuracy = (overlap_correct / overlap_total) * 100 if overlap_total > 0 else 0.0
                
                pairwise_overlap[pair_name] = {
                    'same_answers': same_answers,
                    'different_answers': different_answers,
                    'total_questions': total_questions,
                    'overlap_percentage': overlap_percentage,
                    'overlap_correct': overlap_correct,
                    'overlap_total': overlap_total,
                    'overlap_accuracy': overlap_accuracy
                }
                
                print(f"  🔄 Same Answers: {same_answers}/{total_questions} ({overlap_percentage:.1f}%)")
                print(f"  ❌ Different Answers: {different_answers}/{total_questions} ({100-overlap_percentage:.1f}%)")
                print(f"  🎯 Overlap Accuracy: {overlap_correct}/{overlap_total} ({overlap_accuracy:.1f}%)")
            else:
                print(f"  ❌ No valid questions found")
    
    return pairwise_overlap

def analyze_consensus(generators_data: dict) -> dict:
    """
    Analyze consensus among all three models.
    
    Args:
        generators_data: Data from all generators
        
    Returns:
        Dictionary with consensus analysis
    """
    print(f"\n🤝 ANALYZING THREE-MODEL CONSENSUS")
    print("=" * 50)
    
    generator_names = list(generators_data.keys())
    consensus_analysis = {
        'all_agree': 0,
        'two_agree': 0,
        'all_disagree': 0,
        'consensus_correct': 0,
        'consensus_total': 0,
        'question_details': {}
    }
    
    for question_idx in range(100):
        try:
            answers = {}
            valid_models = 0
            
            # Get answers from all models
            for gen_name in generator_names:
                if str(question_idx) in generators_data[gen_name]:
                    answer, confidence = get_best_answer_for_question(generators_data[gen_name], question_idx)
                    if answer:
                        answers[gen_name] = answer
                        valid_models += 1
            
            if valid_models >= 2:  # Need at least 2 models to analyze consensus
                golden_answer = generators_data[generator_names[0]][str(question_idx)].get('golden_answer', 'Unknown')
                
                if golden_answer != 'Unknown':
                    # Count answer frequencies
                    answer_counts = Counter(answers.values())
                    most_common_answer = answer_counts.most_common(1)[0][0]
                    most_common_count = answer_counts.most_common(1)[0][1]
                    
                    # Determine consensus level
                    if most_common_count == valid_models:
                        consensus_level = "all_agree"
                        consensus_analysis['all_agree'] += 1
                    elif most_common_count >= 2:
                        consensus_level = "two_agree"
                        consensus_analysis['two_agree'] += 1
                    else:
                        consensus_level = "all_disagree"
                        consensus_analysis['all_disagree'] += 1
                    
                    # Check if consensus answer is correct
                    consensus_correct = check_answer_correct(most_common_answer, golden_answer)
                    if consensus_correct:
                        consensus_analysis['consensus_correct'] += 1
                    consensus_analysis['consensus_total'] += 1
                    
                    # Store question details
                    consensus_analysis['question_details'][question_idx] = {
                        'answers': answers,
                        'consensus_answer': most_common_answer,
                        'consensus_level': consensus_level,
                        'consensus_correct': consensus_correct,
                        'golden_answer': golden_answer
                    }
                    
        except Exception as e:
            continue
    
    # Print consensus summary
    total_questions = consensus_analysis['consensus_total']
    if total_questions > 0:
        print(f"  🤝 All Models Agree: {consensus_analysis['all_agree']}/{total_questions} ({consensus_analysis['all_agree']/total_questions*100:.1f}%)")
        print(f"  🤝 Two Models Agree: {consensus_analysis['two_agree']}/{total_questions} ({consensus_analysis['two_agree']/total_questions*100:.1f}%)")
        print(f"  ❌ All Models Disagree: {consensus_analysis['all_disagree']}/{total_questions} ({consensus_analysis['all_disagree']/total_questions*100:.1f}%)")
        print(f"  🎯 Consensus Accuracy: {consensus_analysis['consensus_correct']}/{total_questions} ({consensus_analysis['consensus_correct']/total_questions*100:.1f}%)")
    
    return consensus_analysis

def check_answer_correct(predicted: str, golden: str) -> bool:
    """
    Check if predicted answer matches golden answer.
    
    Args:
        predicted: Predicted answer string
        golden: Golden answer string
        
    Returns:
        True if answers match, False otherwise
    """
    try:
        # Try to convert to float for numerical comparison
        pred_float = float(predicted)
        golden_float = float(golden)
        return abs(pred_float - golden_float) < 1e-6
    except (ValueError, TypeError):
        # If not numerical, do string comparison
        return predicted.strip() == golden.strip()

def generate_comparison_report(model_accuracy: dict, pairwise_overlap: dict, consensus_analysis: dict) -> str:
    """
    Generate a comprehensive comparison report.
    
    Args:
        model_accuracy: Individual model accuracy data
        pairwise_overlap: Pairwise overlap analysis
        consensus_analysis: Consensus analysis data
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("🎯 COMPREHENSIVE MODEL COMPARISON REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Individual Model Performance
    report.append("📊 INDIVIDUAL MODEL PERFORMANCE")
    report.append("-" * 40)
    for gen_name, metrics in model_accuracy.items():
        report.append(f"{gen_name}:")
        report.append(f"  Accuracy: {metrics['accuracy']*100:.1f}% ({metrics['correct_answers']}/{metrics['total_questions']})")
        report.append(f"  Average Confidence: {metrics['avg_confidence']:.3f}")
    report.append("")
    
    # Pairwise Overlap Analysis
    report.append("🔗 PAIRWISE OVERLAP ANALYSIS")
    report.append("-" * 40)
    for pair_name, metrics in pairwise_overlap.items():
        report.append(f"{pair_name}:")
        report.append(f"  Overlap: {metrics['overlap_percentage']:.1f}% ({metrics['same_answers']}/{metrics['total_questions']})")
        report.append(f"  Overlap Accuracy: {metrics['overlap_accuracy']:.1f}% ({metrics['overlap_correct']}/{metrics['overlap_total']})")
        report.append(f"  Disagreement: {100-metrics['overlap_percentage']:.1f}% ({metrics['different_answers']}/{metrics['total_questions']})")
    report.append("")
    
    # Consensus Analysis
    report.append("🤝 CONSENSUS ANALYSIS")
    report.append("-" * 40)
    total_questions = consensus_analysis['consensus_total']
    if total_questions > 0:
        report.append(f"All Models Agree: {consensus_analysis['all_agree']}/{total_questions} ({consensus_analysis['all_agree']/total_questions*100:.1f}%)")
        report.append(f"Two Models Agree: {consensus_analysis['two_agree']}/{total_questions} ({consensus_analysis['two_agree']/total_questions*100:.1f}%)")
        report.append(f"All Models Disagree: {consensus_analysis['all_disagree']}/{total_questions} ({consensus_analysis['all_disagree']/total_questions*100:.1f}%)")
        report.append(f"Consensus Accuracy: {consensus_analysis['consensus_correct']}/{total_questions} ({consensus_analysis['consensus_correct']/total_questions*100:.1f}%)")
    
    # Key Insights
    report.append("")
    report.append("💡 KEY INSIGHTS")
    report.append("-" * 40)
    
    # Find best performing model
    best_model = max(model_accuracy.items(), key=lambda x: x[1]['accuracy'])
    report.append(f"🏆 Best Individual Model: {best_model[0]} ({best_model[1]['accuracy']*100:.1f}%)")
    
    # Find best pairwise overlap
    best_overlap = max(pairwise_overlap.items(), key=lambda x: x[1]['overlap_accuracy'])
    report.append(f"🤝 Best Pairwise Overlap: {best_overlap[0]} ({best_overlap[1]['overlap_accuracy']:.1f}%)")
    
    # Consensus vs Individual
    if consensus_analysis['consensus_total'] > 0:
        consensus_acc = consensus_analysis['consensus_correct'] / consensus_analysis['consensus_total']
        best_individual_acc = max(metrics['accuracy'] for metrics in model_accuracy.values())
        if consensus_acc > best_individual_acc:
            report.append(f"🎯 Consensus outperforms individual models: {consensus_acc*100:.1f}% vs {best_individual_acc*100:.1f}%")
        else:
            report.append(f"⚠️  Consensus underperforms best individual: {consensus_acc*100:.1f}% vs {best_individual_acc*100:.1f}%")
    
    report.append("")
    report.append("=" * 60)
    
    return "\n".join(report)

def save_detailed_analysis(model_accuracy: dict, pairwise_overlap: dict, consensus_analysis: dict, output_file: str = "model_comparison_analysis.json"):
    """
    Save detailed analysis to JSON file.
    
    Args:
        model_accuracy: Individual model accuracy data
        pairwise_overlap: Pairwise overlap analysis
        consensus_analysis: Consensus analysis data
        output_file: Output JSON filename
    """
    analysis_data = {
        "metadata": {
            "analysis_type": "model_comparison",
            "total_questions": 100,
            "models_analyzed": list(model_accuracy.keys())
        },
        "individual_model_accuracy": model_accuracy,
        "pairwise_overlap": pairwise_overlap,
        "consensus_analysis": consensus_analysis
    }
    
    with open(output_file, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    print(f"💾 Detailed analysis saved to: {output_file}")

def main():
    """Main function to run the analysis."""
    print("🎯 Model Comparison Analysis Tool")
    print("=" * 50)
    
    # Load generator data
    data = load_generator_data()
    if not data:
        print("❌ Failed to load data. Exiting.")
        return
    
    generators_data = data['results']
    
    # Run analyses
    print(f"\n🚀 Starting analysis for {len(generators_data)} models...")
    
    # 1. Individual model accuracy
    model_accuracy = analyze_individual_model_accuracy(generators_data)
    
    # 2. Pairwise overlap analysis
    pairwise_overlap = analyze_pairwise_overlap(generators_data)
    
    # 3. Consensus analysis
    consensus_analysis = analyze_consensus(generators_data)
    
    # Generate and display report
    print(f"\n📋 GENERATING COMPREHENSIVE REPORT")
    print("=" * 50)
    
    report = generate_comparison_report(model_accuracy, pairwise_overlap, consensus_analysis)
    print(report)
    
    # Save detailed analysis
    save_detailed_analysis(model_accuracy, pairwise_overlap, consensus_analysis)
    
    print(f"\n🎉 Analysis completed successfully!")
    print(f"💡 Check the detailed JSON file for more information.")

if __name__ == "__main__":
    main()
