#!/usr/bin/env python3
"""
Script to create a consolidated folder structure with all individual question files
from Qwen2.5-7B-Instruct STG experiments only. This creates a folder with all 
Answer.json, Final Solutions.json, and Rollout Solutions.json files from this model.
"""

import json
import os
import shutil
from pathlib import Path
from collections import defaultdict

def create_consolidated_folder():
    """Create a consolidated folder with all question files from Qwen2.5-7B-Instruct experiments only."""
    base_dir = Path(__file__).parent
    
    # Create the consolidated folder
    consolidated_dir = base_dir / "consolidated_answers"
    if consolidated_dir.exists():
        shutil.rmtree(consolidated_dir)
    consolidated_dir.mkdir(exist_ok=True)
    
    print("🔍 Creating consolidated answers folder for Qwen2.5-7B-Instruct...")
    
    # Dictionary to track unique questions and their files
    unique_questions = {}
    total_files = 0
    
    # Process each experiment directory in this model folder
    for exp_dir in base_dir.iterdir():
        if not exp_dir.is_dir() or exp_dir.name in ['consolidated_answers', '__pycache__']:
            continue
            
        print(f"📂 Processing experiment: {exp_dir.name}")
        
        answer_sheets_dir = exp_dir / "answer_sheets"
        if not answer_sheets_dir.exists():
            print(f"   ⚠️  No answer_sheets directory found in {exp_dir.name}")
            continue
            
        # Process each file in answer_sheets
        for file_path in answer_sheets_dir.iterdir():
            if not file_path.is_file():
                continue
                
            filename = file_path.name
            
            # Extract question number and file type
            if "Question" in filename and "Answer.json" in filename:
                # Extract question number
                parts = filename.split("Question")
                if len(parts) > 1:
                    question_part = parts[1].split(" -")[0].strip()
                    try:
                        question_num = int(question_part)
                        question_key = f"Question_{question_num:04d}"
                        
                        if question_key not in unique_questions:
                            unique_questions[question_key] = {}
                        
                        # Copy Answer.json
                        dest_file = consolidated_dir / f"{question_key}_Answer.json"
                        shutil.copy2(file_path, dest_file)
                        unique_questions[question_key]['Answer'] = dest_file.name
                        total_files += 1
                        print(f"   ✅ Copied {question_key} Answer.json")
                        
                    except ValueError:
                        print(f"      ⚠️  Could not parse question number from: {filename}")
                        
            elif "Question" in filename and "Final Solutions.json" in filename:
                # Extract question number
                parts = filename.split("Question")
                if len(parts) > 1:
                    question_part = parts[1].split(" -")[0].strip()
                    try:
                        question_num = int(question_part)
                        question_key = f"Question_{question_num:04d}"
                        
                        if question_key not in unique_questions:
                            unique_questions[question_key] = {}
                        
                        # Copy Final Solutions.json
                        dest_file = consolidated_dir / f"{question_key}_Final_Solutions.json"
                        shutil.copy2(file_path, dest_file)
                        unique_questions[question_key]['Final_Solutions'] = dest_file.name
                        total_files += 1
                        print(f"   ✅ Copied {question_key} Final Solutions.json")
                        
                    except ValueError:
                        print(f"      ⚠️  Could not parse question number from: {filename}")
                        
            elif "Question" in filename and "Rollout Solutions.json" in filename:
                # Extract question number
                parts = filename.split("Question")
                if len(parts) > 1:
                    question_part = parts[1].split(" -")[0].strip()
                    try:
                        question_num = int(question_part)
                        question_key = f"Question_{question_num:04d}"
                        
                        if question_key not in unique_questions:
                            unique_questions[question_key] = {}
                        
                        # Copy Rollout Solutions.json
                        dest_file = consolidated_dir / f"{question_key}_Rollout_Solutions.json"
                        shutil.copy2(file_path, dest_file)
                        unique_questions[question_key]['Rollout_Solutions'] = dest_file.name
                        total_files += 1
                        print(f"   ✅ Copied {question_key} Rollout Solutions.json")
                        
                    except ValueError:
                        print(f"      ⚠️  Could not parse question number from: {filename}")
    
    # Create summary
    summary_file = consolidated_dir / "consolidation_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Qwen2.5-7B-Instruct STG Consolidated Answers Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: Qwen2.5-7B-Instruct\n")
        f.write(f"Dataset: STG\n")
        f.write(f"Total unique questions: {len(unique_questions)}\n")
        f.write(f"Total files copied: {total_files}\n\n")
        
        if unique_questions:
            f.write("Question breakdown:\n")
            for question_key, files in sorted(unique_questions.items()):
                f.write(f"  {question_key}: {', '.join(files.keys())}\n")
    
    print(f"\n📊 Summary:")
    print(f"   Total unique questions: {len(unique_questions)}")
    print(f"   Total files copied: {total_files}")
    print(f"   Consolidated folder created: {consolidated_dir}")
    print(f"   Summary file: {summary_file}")

if __name__ == "__main__":
    create_consolidated_folder()



