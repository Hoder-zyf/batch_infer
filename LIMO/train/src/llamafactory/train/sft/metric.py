# Copyright 2024 HuggingFace Inc., THUDM, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library and the THUDM's ChatGLM implementation.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
# https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Any
import json

import numpy as np
import torch
from transformers.utils import is_jieba_available, is_nltk_available

from ...extras.constants import IGNORE_INDEX
from ...extras.misc import numpify
from ...extras.packages import is_rouge_available


if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer


if is_jieba_available():
    import jieba  # type: ignore


if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


if is_rouge_available():
    from rouge_chinese import Rouge


def eval_logit_processor(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    r"""
    Computes the token with the largest likelihood to reduce memory footprint.
    """
    if isinstance(logits, (list, tuple)):
        if logits[0].dim() == 3:  # (batch_size, seq_len, vocab_size)
            logits = logits[0]
        else:  # moe models have aux loss
            logits = logits[1]

    if logits.dim() != 3:
        raise ValueError("Cannot process the logits.")

    return torch.argmax(logits, dim=-1)


@dataclass
class ComputeAccuracy:
    r"""
    Computes accuracy and supports `batch_eval_metrics`.
    """

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        for i in range(len(preds)):
            pred, label = preds[i, :-1], labels[i, 1:]
            label_mask = label != IGNORE_INDEX
            self.score_dict["accuracy"].append(np.mean(pred[label_mask] == label[label_mask]))

        if compute_result:
            return self._dump()


@dataclass
class ComputeSimilarity:
    r"""
    Computes text similarity scores and supports `batch_eval_metrics`.

    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                self.score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            self.score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        if compute_result:
            return self._dump()


@dataclass
class ComputeJudgeScore:
    r"""
    Computes accuracy scores using an external LLM judge.
    """

    tokenizer: "PreTrainedTokenizer"
    judge_args: Dict[str, Any]

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"judge_accuracy": []}
        return result

    def __post_init__(self):
        self._dump()
        from openai import OpenAI
        
        self.client = OpenAI(
            api_key=self.judge_args.get("eval_judge_api_key", "not-needed"),
            base_url=self.judge_args.get("eval_judge_api_base"),
        )
        self.model_name = self.judge_args.get("eval_judge_model")

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            # Simple prompt for math problem judging
            prompt = f"""
            Please evaluate if the student's answer is correct based on the ground truth.
            
            Student Answer:
            {pred}
            
            Ground Truth:
            {label}
            
            Please provide your evaluation in JSON format with the following keys:
            - reasoning: A brief explanation of your evaluation.
            - correct: boolean (true or false) indicating if the answer is correct.
            """
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful math teacher. You must output JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=512,
                    response_format={"type": "json_object"}
                )
                judge_output_str = response.choices[0].message.content.strip()
                judge_output = json.loads(judge_output_str)
                is_correct = judge_output.get("correct", False)
                reasoning = judge_output.get("reasoning", "No reasoning provided")
                
                self.score_dict["judge_accuracy"].append(1.0 if is_correct else 0.0)
                
                # Log the first few examples
                if len(self.score_dict["judge_accuracy"]) <= 5:
                    print(f"\n[Judge] Correct: {is_correct} | Reasoning: {reasoning}")

                # Save detailed result
                # We need a place to save. Since we don't have easy access to output_dir here without passing it,
                # we will try to use the one passed in judge_args if available, or default to current dir.
                # However, judge_args comes from finetuning_args.to_dict(), which might not have output_dir unless we added it.
                # But we can just use a fixed filename in the current working directory or try to find the output dir.
                # For now, let's append to "judge_results.jsonl" in the current directory, 
                # or better, use the `output_dir` from `judge_args` if we can pass it.
                # `finetuning_args` doesn't have `output_dir`. `training_args` does.
                # We only passed `finetuning_args` to `ComputeJudgeScore`.
                
                # Let's just save to "judge_results.jsonl" in the current directory for now, 
                # or we can try to infer the output directory. 
                # Actually, `metric.py` is usually run in the context where we might want to save to the experiment folder.
                # But without `output_dir` passed explicitly, we can't be sure.
                # Let's assume the user runs this from the script root.
                
                result_entry = {
                    "prediction": pred,
                    "ground_truth": label,
                    "judge_output": judge_output,
                    "score": 1.0 if is_correct else 0.0
                }
                
                with open("judge_results.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
                    
            except Exception as e:
                print(f"Judge error: {e}")
                self.score_dict["judge_accuracy"].append(0.0)

        if compute_result:
            return self._dump()
