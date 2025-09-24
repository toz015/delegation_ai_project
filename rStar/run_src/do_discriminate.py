# Licensed under the MIT license.

import sys
import os, json
from tqdm import tqdm

sys.path.append(".")

from common.utils import fix_seeds, read_json, read_txt
from eval_src.Evaluator import *
from run_src.rstar_utils import concat_solution_trace, mask_solution_trace
from models.vLLM_API import load_vLLM_model, generate_with_vLLM_model
from models.HuggingFace_API import load_HF_model, generate_with_HF_model

from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
from datetime import datetime


class Candidate:
    def __init__(
        self,
        solution_trace,
        masked_solution_trace_list,
        final_step,
        final_answer,
        id,
        freq=1,
        trace_reward=1.0,
        c_type="default",
    ):
        self.solution_trace = solution_trace
        self.masked_solution_trace_list = masked_solution_trace_list
        self.final_step = final_step
        self.final_answer = final_answer
        self.id = id
        self.freq = freq
        self.trace_reward = trace_reward
        self.c_type = c_type

    def __str__(self):
        return f"Candidate {self.id}: {self.final_answer}"

    def to_dict(self):
        return {
            "solution_trace": self.solution_trace,
            "masked_solution_trace_list": self.masked_solution_trace_list,
            "final_step": self.final_step,
            "final_answer": self.final_answer,
            "id": self.id,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            solution_trace=data["solution_trace"],
            masked_solution_trace_list=data["masked_solution_trace_list"],
            final_step=data["final_step"],
            final_answer=data["final_answer"],
            id=data["id"],
        )


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

    assert all(answer2cnt[ans] == len(answer2candidates[ans]) for ans in answer2cnt.keys())
    assert float(sum([candidate.trace_reward for candidate in candidates])) == float(
        sum([answer2confidence[ans] for ans in answer2confidence.keys()])
    )

    candidates_count = sum([candidate.freq for candidate in candidates])
    for ans in answer2confidence.keys():
        answer2confidence[ans] /= candidates_count

    return answer2candidates, answer2confidence, answer2cnt


class Discriminator:
    def __init__(self, args, evaluator):
        self.args = args
        self.evaluator = evaluator

        self.fewshot_config = read_json(args.fewshot_config_path)
        self.fewshot_template = self.fewshot_config["prompt_template"]
        self.stop_tokens = self.fewshot_config["stop_tokens"]

        self.fewshot_prompt = read_txt(args.fewshot_prompt_path)
        
        # In-memory cache for current question's candidate answers only
        # Key: answer_hash, Value: raw_survival_rate
        self.survival_rate_cache = {}
        
    def _get_cache_key(self, answer: str) -> str:
        """Generate a cache key for answer (question context not needed for in-memory)"""
        import hashlib
        return hashlib.md5(str(answer).encode()).hexdigest()[:6]
    
    def _get_cached_survival_rate(self, answer: str) -> float:
        """Get cached raw survival rate if available"""
        cache_key = self._get_cache_key(answer)
        return self.survival_rate_cache.get(cache_key, None)
    
    def _cache_survival_rate(self, answer: str, raw_survival_rate: float):
        """Cache raw survival rate for current question"""
        cache_key = self._get_cache_key(answer)
        self.survival_rate_cache[cache_key] = raw_survival_rate
    
    def clear_cache(self):
        """Clear cache for new question"""
        self.survival_rate_cache.clear()

    def _filter_none(self, candidates: list[Candidate]) -> list[Candidate]:
        candidates = [c for c in candidates if c.final_answer is not None]
        return candidates

    def _filter_long(self, candidates: list[Candidate]) -> list[Candidate]:
        candidates = [c for c in candidates if len(c.final_answer) <= 100]
        return candidates

    def _filter_reasoning_consistency(
        self, gen_model, problem: str, candidates: list[Candidate], aux={}
    ) -> list[Candidate]:
        problem_id = aux["problem_id"]
        file_idx = aux["file_idx"]

        prompt_template = self.fewshot_template
        fewshot_examples = self.fewshot_prompt
        stop_tokens = self.stop_tokens

        assert all(
            len(c.masked_solution_trace_list) == self.args.num_masked_solution_traces
            for c in candidates
            if c.c_type == "default"
        )
        
        # Check cache first for already evaluated answers
        cached_candidates = []
        uncached_candidates = []
        
        for c in candidates:
            cached_rate = self._get_cached_survival_rate(c.final_answer)
            if cached_rate is not None:
                # Use cached result - mark as consistent if survival rate > 0
                if cached_rate > 0:
                    cached_candidates.append(c)
            else:
                uncached_candidates.append(c)
        
        # If all candidates are cached, return early
        if not uncached_candidates:
            return cached_candidates
        
        # Process only uncached candidates
        gen_input_list = []
        ground_truth_list = []
        c_completion_num_list = []
        for c in uncached_candidates:
            for masked_solution_trace in c.masked_solution_trace_list:
                for _ in range(self.args.rc_n_completions):
                    gen_input_list.append(
                        prompt_template.format(examples=fewshot_examples, instruction=problem) + masked_solution_trace
                    )
                    ground_truth_list.append(c.final_answer)
            c_completion_num_list.append(len(c.masked_solution_trace_list) * self.args.rc_n_completions)
        """gen_input_list:
        [c1_mask1, c1_mask2, ..., c2_mask1, c2_mask2, ..., ......, ct_mask1, ct_mask2, ...]
        """
        #print(f" gen_input_list: {gen_input_list}")
        # Manually split into batches
        batch_size = self.args.max_num_seqs // self.args.rc_n_completions // 2
        gen_output_list = []
        for start_idx in range(0, len(gen_input_list), batch_size):
            end_idx = start_idx + batch_size
            sub_gen_input_list = gen_input_list[start_idx:end_idx]
            sub_gen_output_list = self._gen_func(
                gen_model=gen_model,
                gen_input=sub_gen_input_list,
                temperature=self.args.rc_temperature,
                n=1,
                max_tokens=512,
                stop_tokens=stop_tokens + ["\n"],
            )
            gen_output_list.extend(sub_gen_output_list)

        with open(os.path.join(self.args.discriminate_results_dir, f"problem-{problem_id}.json"), "w") as f:
            js = {"problem_id": problem_id, "file_idx": file_idx, "gen_output_list": gen_output_list}
            json.dump(js, f)

        """gen_output_list:
        [[c1_mask1_o1, c1_mask1_o2, ...], [c1_mask2_o1, c1_mask2_o2, ...], ..., [ct_mask1_o1, ct_mask1_o2, ...], [ct_mask2_o1, ct_mask2_o2, ...], ...]
        """

        if all(isinstance(item, list) for item in gen_output_list):
            completion_list = []
            for n_completions in gen_output_list:
                for completion in n_completions:
                    completion_list.append(completion)
            assert len(completion_list) == self.args.rc_n_completions * self.args.num_masked_solution_traces * len(
                candidates
            )
            candidate_group_size = self.args.rc_n_completions * self.args.num_masked_solution_traces
        elif all(isinstance(item, str) for item in gen_output_list):
            completion_list = gen_output_list
            candidate_group_size = self.args.num_masked_solution_traces

        answer_list = [
            self.evaluator.extract_answer_from_model_completion(completion) for completion in completion_list
        ]

        count = 0
        completion_group_list = []
        answer_group_list = []
        gt_group_list = []
        for num in c_completion_num_list:
            completion_group_list.append(completion_list[count : count + num])
            answer_group_list.append(answer_list[count : count + num])
            gt_group_list.append(ground_truth_list[count : count + num])
            count += num
        assert count == len(completion_list) == len(answer_list)

        consistent_candidates = []

        for c, completion_group, answer_group, gt_answer in zip(
            uncached_candidates, completion_group_list, answer_group_list, gt_group_list
        ):
            candidate_group_size = len(c.masked_solution_trace_list)
            num_consistent = 0
            if self.args.rc_mode == "maj":
                answer = self.evaluator.find_most_confident_answer(completion_group)[0]
                if self.evaluator.check_answers_equiv(gt_answer[-1], answer):
                    consistent_candidates.append(c)
                    # Cache the raw survival rate (1.0 for consistent)
                    self._cache_survival_rate(c.final_answer, 1.0)
                else:
                    # Cache the raw survival rate (0.0 for inconsistent)
                    self._cache_survival_rate(c.final_answer, 0.0)
            else:
                for answer, gt_a in zip(answer_group, gt_answer):
                    if self.evaluator.check_answers_equiv(gt_a, answer):
                        num_consistent += 1
                
                # Calculate raw survival rate based on consistency mode
                if self.args.rc_mode == "loose":
                    raw_survival_rate = 1.0 if num_consistent > 0 else 0.0
                elif self.args.rc_mode == "mid":
                    raw_survival_rate = 1.0 if num_consistent >= candidate_group_size // 2 else 0.0
                elif self.args.rc_mode == "strict":
                    raw_survival_rate = 1.0 if num_consistent == candidate_group_size else 0.0
                
                # Cache the raw survival rate
                self._cache_survival_rate(c.final_answer, raw_survival_rate)
                
                # Add to consistent candidates if survival rate > 0
                if raw_survival_rate > 0:
                    consistent_candidates.append(c)
        
        # Combine cached and newly evaluated consistent candidates
        return cached_candidates + consistent_candidates

    def _gen_func(self, gen_model, gen_input, temperature: float, n: int = 1, max_tokens: int = 768, stop_tokens=None):
        if temperature == 0.0:
            n = 1
        response = None
        if self.args.api.lower() == "vllm":
            response = generate_with_vLLM_model(
                model=gen_model, input=gen_input, temperature=temperature, n=n, max_tokens=max_tokens, stop=stop_tokens
            )
        elif self.args.api.lower() == "huggingface":
            response = generate_with_HF_model(
                tokenizer=self.tokenizer, model=gen_model, input=gen_input, temperature=temperature, n=n, max_tokens=max_tokens, stop=stop_tokens
            )

        if n == 1:
            if isinstance(gen_input, str):
                return response[0].outputs[0].text
            elif isinstance(gen_input, list):
                return [r.outputs[0].text for r in response]
        elif n > 1:
            if isinstance(gen_input, str):
                return [o.text for o in response[0].outputs]
            elif isinstance(gen_input, list):
                return [[o.text for o in r.outputs] for r in response]

    def _calculate_scores(self, unfiltered_candidates: list[Candidate], filtered_candidates: list[Candidate]) -> dict:
        _, filtered_answer2confidence, filtered_answer2cnt = group_candidates_by_answer(
            filtered_candidates, self.evaluator, self.args.rc_criteria
        )
       # print(f"==> Confidence: {filtered_answer2confidence}")
        _, _, unfiltered_answer2cnt = group_candidates_by_answer(
            unfiltered_candidates, self.evaluator, self.args.rc_criteria
        )

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

        #print(f"==> Survival rates: {filtered_answer2survival_rate}")

        filtered_answer2score = {}
        for filtered_ans in filtered_answer2confidence.keys():
            has_existed = False
            for unfiltered_ans in unfiltered_answer2cnt.keys():
                if self.evaluator.check_answers_equiv(filtered_ans, unfiltered_ans):
                    has_existed = True
                    filtered_answer2score[filtered_ans] = (
                        filtered_answer2confidence[filtered_ans] + filtered_answer2survival_rate[filtered_ans]
                    )
                    break
            if not has_existed:
                filtered_answer2score[filtered_ans] = 0.0

        print(f"==> Scores: {filtered_answer2score}")

        return filtered_answer2score

    def _find_winner_filtered(
        self, unfiltered_candidates: list[Candidate], filtered_candidates: list[Candidate], gt_answer: str = None
    ) -> Candidate:
        if len(filtered_candidates) == 0:
            answer2candidates, answer2confidence, _ = group_candidates_by_answer(
                unfiltered_candidates, self.evaluator, self.args.rc_criteria
            )
            most_confident_answer = max(answer2confidence.keys(), key=lambda x: answer2confidence[x])
            winner = answer2candidates[most_confident_answer][0]
            # print(f"==> Winner answer: {most_confident_answer}\n")
        elif len(filtered_candidates) == 1:
            winner = filtered_candidates[0]
            # print(f"==> Winner answer: {winner.final_answer}\n")
        elif not any(self.evaluator.check_answers_equiv(c.final_answer, gt_answer) for c in filtered_candidates):
            winner = None
            # print(f"==> Winner answer: None")
        else:
            filtered_answer2score = self._calculate_scores(unfiltered_candidates, filtered_candidates)
            winner_answer = max(filtered_answer2score.keys(), key=lambda x: filtered_answer2score[x])
            # print(f"==> Winner answer: {winner_answer}")
            winner = next(
                c for c in filtered_candidates if self.evaluator.check_answers_equiv(c.final_answer, winner_answer)
            )

        return winner


class MajorityVoteDiscriminator(Discriminator):
    def __init__(self, args, evaluator):
        super().__init__(args, evaluator)
        self.tokenizer, self.model = None, None
        if self.args.api == "vllm":
            self.tokenizer, self.model = load_vLLM_model(
                args.model_ckpt, 
                args.YOUR_HUGGINGFACE_TOKEN_HERE, 
                args.seed, 
                max_num_seqs=args.max_num_seqs,
                max_model_len=getattr(args, 'max_model_len', None),
                gpu_memory_utilization=getattr(args, 'gpu_memory_utilization', None)
            )
        elif self.args.api == "huggingface":
            self.tokenizer, self.model = load_HF_model(args.model_ckpt, args.YOUR_HUGGINGFACE_TOKEN_HERE)
        else:
            print("Unspecify Model")
    
    
    def select(self, problem: str, candidates: list[Candidate], gt_answer: str = None, aux={}) -> Candidate:
        print(f"==> Ground truth answer: {gt_answer}")

        # Clear cache for new question
        self.clear_cache()

        unfiltered_candidates = candidates
        #print(f"==> Unfiltered answers: {[c.final_answer for c in unfiltered_candidates]}")
        # candidate: [1, 2, 3, 4, 5, None, paosdifjpsod]
        prefiltered_candidates = self._filter_none(candidates)
        prefiltered_candidates = self._filter_long(prefiltered_candidates)
        # prefiltered_candidates: [1, 2, 3, 4, 5]
        #print(f"==> Pre-filtered answers: {[c.final_answer for c in prefiltered_candidates]}")
        filtered_candidates = self._filter_reasoning_consistency(self.model, problem, prefiltered_candidates, aux)
        # filtered_candidates: [1, 2, 3]
        #print(f"==> RC-filtered answers: {[c.final_answer for c in filtered_candidates]}")
        
        return self._find_winner_filtered(prefiltered_candidates, filtered_candidates, gt_answer)


def main(args):
    fix_seeds(args.seed)
    print(args)
    if not hasattr(args, 'discriminate_results_dir'):
        print("Setting up output directories from within main function...")
        exp_id = f"dis_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}---{args.note}"
        discriminate_out_dir = os.path.join(args.root_dir, exp_id)
        os.makedirs(discriminate_out_dir, exist_ok=True)
        args.discriminate_results_dir = os.path.join(discriminate_out_dir, "results")
        os.makedirs(args.discriminate_results_dir, exist_ok=True)
    
    # 定义 recording 和 recording_file 变量
    recording_file = os.path.join(args.discriminate_results_dir, "..", "recording.json")
    recording = vars(deepcopy(args)) # 复制一份args用于记录，避免后续修改影响记录
    
    print(args)

    # --- Setup ---
    evaluator = eval(f"{args.dataset_name}Evaluator()")
    discriminator = MajorityVoteDiscriminator(args, evaluator)
    answer_sheets_dir = os.path.join(args.root_dir, "answer_sheets")
    #! ------ Select winner candidate for each example ------
    answer_sheet_json_files = [
        os.path.join(answer_sheets_dir, f) for f in os.listdir(answer_sheets_dir) if f.endswith("Answer.json")
    ]
    answer_sheet_json_files.sort()
    if args.start_idx > -1 and args.end_idx > -1:
        answer_sheet_json_files = answer_sheet_json_files[args.start_idx : args.end_idx]

    num_correct, num_correct_majvote, num_correct_limit, num_tested = 0, 0, 0, 0
    with tqdm(total=len(answer_sheet_json_files), disable=True) as pbar:
        total_num_candidates = 0
        for file_idx, answer_js_file in enumerate(answer_sheet_json_files):
            problem_id = int(
                answer_js_file.split("/")[-1].split(".")[0].replace(" - Answer", "").replace("Question ", "")
            )
            if args.resume and os.path.exists(
                os.path.join(args.discriminate_results_dir, f"problem-{problem_id}.json")
            ):
                print(f"\n[Skip file {file_idx}; Total number of files: {len(answer_sheet_json_files)}]\n")
                with open(os.path.join(args.discriminate_results_dir, f"problem-{problem_id}.json"), "r") as f:
                    temp_recording = json.load(f)
                correct = temp_recording["correct"]
                correct_majvote = temp_recording["correct_majvote"]
                correct_limit = temp_recording["correct_limit"]

                num_correct += int(correct)
                num_correct_majvote += int(correct_majvote)
                num_correct_limit += int(correct_limit)
                num_tested += 1

                info = f"Acc: {num_correct / num_tested:.4f}; Majority vote acc: {num_correct_majvote / num_tested:.4f}; Limit acc: {num_correct_limit / num_tested:.4f}"
                print(info)
                pbar.set_description(info, refresh=True)
            else:
                print(f"\n[Processing file {file_idx}; Total number of files: {len(answer_sheet_json_files)}]\n")
                try:
                    answer_js = read_json(answer_js_file)
                except:
                    continue

                try:
                    problem = answer_js["problem"]
                    # assert problem_id == answer_js["id"]
                    gold_answer = answer_js["gold_answer"]
                except:
                    pass

                trace_js = read_json(answer_js_file.replace("Answer", "Final Solutions")) + read_json(
                    answer_js_file.replace("Answer", "Rollout Solutions")
                )
                if args.cutoff_rollout > -1:
                    trace_js = [s for s in trace_js if s["rollout_id"] <= args.cutoff_rollout]

                # ------ Collect all_candidates, answer2candidates answer2confidence ------
                all_candidates = []
                solution_trace_dic = {}  # TODO
                for id, s in enumerate(trace_js):
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

                for solution_trace in solution_trace_dic.keys():
                    final_step = solution_trace_dic[solution_trace]["final_step"]
                    trace_freq = solution_trace_dic[solution_trace]["freq"]
                    trace_reward = solution_trace_dic[solution_trace]["reward"]

                    masked_solution_trace_list = mask_solution_trace(
                        solution_trace,
                        num_return=args.num_masked_solution_traces,
                        left_boundary=args.mask_left_boundary,
                        right_boundary=args.mask_right_boundary,
                    )
                    final_answer = evaluator.extract_answer_from_model_completion(final_step)
                    candidate = Candidate(
                        solution_trace,
                        deepcopy(masked_solution_trace_list),
                        final_step,
                        final_answer,
                        id,
                        trace_freq,
                        trace_reward,
                    )
                    all_candidates.append(candidate)

                answer2candidates, answer2confidence, _ = group_candidates_by_answer(
                    all_candidates, evaluator, args.rc_criteria
                )
                most_confident_answer = max(answer2candidates.keys(), key=lambda x: answer2confidence[x])
                highest_confidence = answer2confidence[most_confident_answer]
                assert highest_confidence > 0
                
                # Store initial confidence scores for comparison
                initial_confidence_scores = dict(answer2confidence)
                
                # Store individual generator confidence scores (before discrimination)
                # Each rollout represents a different generator
                initial_generator_confidence = {}
                for candidate in all_candidates:
                    # Each candidate has a rollout_id that identifies the generator
                    generator_id = candidate.rollout_id if hasattr(candidate, 'rollout_id') else candidate.id
                    answer = candidate.final_answer
                    
                    if answer not in initial_generator_confidence:
                        initial_generator_confidence[answer] = {}
                    
                    # Calculate individual generator confidence based on frequency/reward
                    if args.rc_criteria == "freq":
                        generator_conf = candidate.freq
                    else:  # reward
                        generator_conf = candidate.trace_reward
                    
                    initial_generator_confidence[answer][generator_id] = generator_conf
                
                # -------------------------------------------------------------------------

                # candidates = [cands[0] for _, cands in answer2candidates.items()]   #! representative
                candidates = all_candidates  #! exhaustive
                total_num_candidates += len(candidates)

                # ------ Get winner answer ------
                if not any(evaluator.check_answers_equiv(ans, gold_answer) for ans in answer2candidates.keys()):
                    # In this case, we know that there is no correct answer in the candidates
                    print("Well, no correct answer in candidates. Skipping...")
                    winner_answer = ""
                    final_survival_rates = {}
                    final_generator_confidence = {}
                else:
                    if highest_confidence > args.threshold:
                        print("You are very confident. Skipping...")
                        winner_answer = most_confident_answer
                        final_survival_rates = {}
                        final_generator_confidence = {}
                    else:
                        winner_candidate = discriminator.select(
                            problem,
                            candidates,
                            gt_answer=gold_answer,
                            aux={"file_idx": file_idx, "problem_id": problem_id},
                        )
                        if winner_candidate is not None:
                            winner_answer = winner_candidate.final_answer
                        else:
                            winner_answer = most_confident_answer
                        
                        # Get final confidence scores for each generator after discrimination
                        if len(filtered_candidates) > 1:
                            # Calculate final confidence for each generator
                            final_generator_confidence = {}
                            
                            for candidate in filtered_candidates:
                                generator_id = candidate.rollout_id if hasattr(candidate, 'rollout_id') else candidate.id
                                answer = candidate.final_answer
                                
                                if answer not in final_generator_confidence:
                                    final_generator_confidence[answer] = {}
                                
                                # Calculate individual generator confidence after filtering
                                if args.rc_criteria == "freq":
                                    generator_conf = candidate.freq
                                else:  # reward
                                    generator_conf = candidate.trace_reward
                                
                                final_generator_confidence[answer][generator_id] = generator_conf
                            
                            # Calculate final survival rates
                            _, _, final_answer2cnt = group_candidates_by_answer(
                                filtered_candidates, evaluator, args.rc_criteria
                            )
                            _, _, unfiltered_answer2cnt = group_candidates_by_answer(
                                candidates, evaluator, args.rc_criteria
                            )
                            
                            final_survival_rates = {}
                            for filtered_ans in final_answer2cnt.keys():
                                for unfiltered_ans in unfiltered_answer2cnt.keys():
                                    if evaluator.check_answers_equiv(filtered_ans, unfiltered_ans):
                                        final_survival_rates[filtered_ans] = (
                                            final_answer2cnt[filtered_ans] / unfiltered_answer2cnt[unfiltered_ans]
                                        )
                                        break
                        else:
                            final_survival_rates = {}
                            final_generator_confidence = {}
            
            # -------------------------------
            correct = evaluator.check_answers_equiv(winner_answer, gold_answer)
            correct_majvote = evaluator.check_answers_equiv(most_confident_answer, gold_answer)
            correct_limit = (
                1 if any(evaluator.check_answers_equiv(ans, gold_answer) for ans in answer2candidates.keys()) else 0
            )

            try:
                with open(os.path.join(args.discriminate_results_dir, f"problem-{problem_id}.json"), "r") as f:
                    temp_recording = json.load(f)
            except:
                temp_recording = {}
            temp_recording.update(
                {
                    "correct": correct,
                    "correct_majvote": correct_majvote,
                    "correct_limit": correct_limit,
                    "confidence_evolution": {
                        "initial_confidence_scores": initial_confidence_scores,
                        "initial_generator_confidence": initial_generator_confidence,
                        "final_generator_confidence": final_generator_confidence,
                        "final_survival_rates": final_survival_rates,
                        "winner_answer": winner_answer,
                        "generator_top_answer": most_confident_answer,
                        "gold_answer": gold_answer,
                        "model_name": model_name
                    }
                }
            )
            with open(os.path.join(args.discriminate_results_dir, f"problem-{problem_id}.json"), "w") as f:
                json.dump(temp_recording, f, indent=4)
            num_correct += int(correct)
            num_correct_majvote += int(correct_majvote)
            num_correct_limit += int(correct_limit)
            num_tested += 1

            info = f"Acc: {num_correct / num_tested:.4f}; Majority vote acc: {num_correct_majvote / num_tested:.4f}; Limit acc: {num_correct_limit / num_tested:.4f}"
            print(info)
            pbar.set_description(info, refresh=True)

            pbar.update(1)
    #! --------------------------------------------------------


    
    # Report cache statistics
    # cache_stats = discriminator.get_cache_stats() # Removed as per edit hint
    # print(f"\n📊 Cache: {cache_stats['total_entries']} entries, {cache_stats['cache_size_mb']:.2f} MB") # Removed as per edit hint
    
    # Save final cache state
    # discriminator.save_cache() # Removed as per edit hint
    
    # Save recording
    recording.update(
        {
            "num_correct": num_correct,
            "num_correct_majvote": num_correct_majvote,
            "num_correct_limit": num_correct_limit,
            "num_tested": num_tested,
            "accuracy": num_correct / num_tested,
            "majority_vote_accuracy": num_correct_majvote / num_tested,
            "limit_accuracy": num_correct_limit / num_tested,
            "avg_num_candidates": total_num_candidates / num_tested,
        }
    )

    print(f"Recording: \n{recording}")

    with open(recording_file, "w") as f:
        json.dump(recording, f, indent=4)


if __name__ == "__main__":
    # This block now mirrors do_generator.py exactly.
    # It only runs when the script is executed directly.
    parser = ArgumentParser()

    # --- Add ALL arguments here ---
    parser.add_argument("--note", type=str, default="default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api", type=str, default="vllm")
    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    # ... (Add all other arguments from your original discriminator file)
    parser.add_argument("--rc_criteria", type=str, default="reward", choices=["freq", "reward"])

    args = parser.parse_args()

    # --- Post-process args and call main ---
    # Construct paths based on args
    args.fewshot_config_path = os.path.join("prompts", args.dataset_name, "fewshot_cot", "fewshot_cot_config.json")
    args.fewshot_prompt_path = os.path.join("prompts", args.dataset_name, "fewshot_cot", "fewshot_cot_prompt.txt")

    # Setup output directories
    exp_id = f"dis_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}---{args.note}"
    discriminate_out_dir = os.path.join(args.root_dir, exp_id)
    os.makedirs(discriminate_out_dir, exist_ok=True)
    args.discriminate_results_dir = os.path.join(discriminate_out_dir, "results")
    os.makedirs(args.discriminate_results_dir, exist_ok=True)
    # Call the core logic function
    main(args)