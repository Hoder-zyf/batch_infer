import argparse
import json
import os
import time
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm
import concurrent.futures
import threading
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Parallel inference client script for AIME dataset using multiple LoRA adapters (incremental write version)")
    parser.add_argument('--host', type=str, default="localhost", help="vLLM service host address")
    parser.add_argument('--port', type=int, default=8000, help="vLLM service port")
    parser.add_argument('--output-dir', type=str, default="./aime_results_concurrent_8", help="Directory to save generated results")
    parser.add_argument('--max-tokens', type=int, default=38912, help="Maximum number of tokens to generate")
    parser.add_argument('--temperature', type=float, default=0.6, help="Sampling temperature (0.0 for greedy)")
    parser.add_argument('--concurrency', type=int, default=240, help="Number of concurrent parallel requests")
    parser.add_argument('--timeout', type=int, default=1800, help="API request timeout (seconds), default 1800 seconds (30 minutes)")
    parser.add_argument('--num-samples', type=int, default=8, help="Number of samples per question, default 8")
    parser.add_argument('--max-retries', type=int, default=3, help="Maximum number of retries on API call failure")
    parser.add_argument('--resume', action='store_true', help="Resume from last interruption (checkpoint resume)")
    parser.add_argument('--dataset-config', type=str, default=None, help="Path to dataset configuration JSON file")
    parser.add_argument('--dataset-name', type=str, default="math-ai/aime25", help="Dataset name (if not using config file)")
    parser.add_argument('--dataset-split', type=str, default="test", help="Dataset split (if not using config file)")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--lora-modules', type=str, nargs='+', help="List of LoRA module names to test")
    group.add_argument('--model-names', type=str, nargs='+', help="List of full model names to test")
    
    return parser.parse_args()

def load_dataset_config(config_path: str) -> Dict[str, Any]:
    """
    加载数据集配置文件
    
    配置文件格式示例:
    {
        "dataset_name": "math-ai/aime25",  # HuggingFace 数据集名或本地路径
        "split": "test",  # 可选，某些数据集可能没有split
        "is_local": false,  # 可选，true表示本地路径
        "system_prompt": "Please reason step by step...",
        "problem_field": "problem",
        "answer_field": "answer"
    }
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 设置默认值
    config.setdefault('system_prompt', "Please reason step by step, and put your final answer within \\boxed{}.")
    config.setdefault('problem_field', 'problem')
    config.setdefault('answer_field', 'answer')
    config.setdefault('split', None)  # split 可以为 None
    config.setdefault('is_local', False)  # 默认从 HuggingFace 加载
    
    return config


def create_aime_prompt(problem_text: str, system_prompt: Optional[str] = None) -> list:
    """Create a standardized chat prompt for AIME problems"""
    if system_prompt is None:
        system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem_text}
    ]
    return messages

def write_error_to_file(error_filepath, lock, problem_idx, sample_idx, error_msg, model_name):
    """Thread-safely write error information to file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    error_line = f"[{timestamp}] Problem {problem_idx}, Sample {sample_idx+1}, Model: {model_name}\n"
    error_line += f"  Error: {error_msg}\n"
    error_line += "-" * 80 + "\n"
    
    with lock:
        with open(error_filepath, 'a', encoding='utf-8') as f:
            f.write(error_line)

def load_existing_results(temp_filepath: str) -> Dict[str, Any]:
    """Load existing temporary results (for checkpoint resume)"""
    if not os.path.exists(temp_filepath):
        return {}
    
    results = {}
    try:
        with open(temp_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    key = f"{data['problem_idx']}_{data['sample_idx']}"
                    results[key] = data
    except Exception as e:
        print(f"Warning: Error loading temporary file: {e}")
        return {}
    
    return results

def write_sample_result(temp_filepath: str, result: Dict[str, Any], lock: threading.Lock):
    """Incrementally write single sample result to temporary file"""
    with lock:
        with open(temp_filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

def process_single_sample(
    problem_idx: int,
    sample_idx: int,
    messages: list,
    client: OpenAI,
    model_name: str,
    args,
    temp_filepath: str,
    temp_lock: threading.Lock,
    error_filepath: Optional[str] = None,
    error_lock: Optional[threading.Lock] = None,
    existing_results: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Function to process a single sample (true parallel unit)
    
    Returns:
        Dictionary containing sample result
    """
    # Check if already completed (checkpoint resume)
    result_key = f"{problem_idx}_{sample_idx}"
    if existing_results and result_key in existing_results:
        return existing_results[result_key]
    
    for retry in range(args.max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            generated_text = response.choices[0].message.content
            
            result = {
                'problem_idx': problem_idx,
                'sample_idx': sample_idx,
                'success': True,
                'solution': generated_text,
                'error': None,
                'timestamp': datetime.now().isoformat()
            }
            
            # 立即写入临时文件
            write_sample_result(temp_filepath, result, temp_lock)
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            
            # Only record error on last retry failure
            if retry == args.max_retries - 1:
                if error_filepath and error_lock:
                    write_error_to_file(error_filepath, error_lock, problem_idx, sample_idx, error_msg, model_name)
                
                result = {
                    'problem_idx': problem_idx,
                    'sample_idx': sample_idx,
                    'success': False,
                    'solution': None,
                    'error': error_msg,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Write failure result to temporary file as well
                write_sample_result(temp_filepath, result, temp_lock)
                
                return result
            else:
                # Wait a bit before retrying
                time.sleep(1 * (retry + 1))  # Exponential backoff

def consolidate_results(temp_filepath: str, output_filepath: str, dataset, num_samples: int, model_name: str,
                       problem_field: str = 'problem', answer_field: str = 'answer'):
    """Consolidate sample results from temporary file into final format"""
    # Read all sample results
    results_dict = {}
    with open(temp_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                problem_idx = data['problem_idx']
                sample_idx = data['sample_idx']
                
                if problem_idx not in results_dict:
                    results_dict[problem_idx] = {}
                results_dict[problem_idx][sample_idx] = data
    
    # Organize into final format
    final_results = []
    for problem_idx, problem in enumerate(dataset):
        generated_solutions = []
        errors = []
        
        # Organize results in sample_idx order
        for sample_idx in range(num_samples):
            sample_result = results_dict.get(problem_idx, {}).get(sample_idx)
            if sample_result and sample_result['success']:
                generated_solutions.append(sample_result['solution'])
            else:
                generated_solutions.append(None)
                if sample_result and sample_result['error']:
                    errors.append(f"Sample {sample_idx}: {sample_result['error']}")
        
        result_entry = {
            'problem_index': problem_idx,
            'question': problem[problem_field],
            'ground_truth_answer': problem.get(answer_field, 'N/A'),
            'model_used': model_name,
            'num_samples': num_samples,
            'generated_solutions': generated_solutions,
        }
        
        if errors:
            result_entry['errors'] = errors
        
        final_results.append(result_entry)
    
    # Write to final file
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for entry in final_results:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    return final_results

def main():
    args = parse_args()

    # --- 1. Load dataset configuration if provided ---
    dataset_config = None
    is_local = False
    
    if args.dataset_config:
        print(f"Loading dataset configuration from: {args.dataset_config}")
        # 检查是否同时传了命令行参数（会被忽略）
        if args.dataset_name != "math-ai/aime25" or args.dataset_split != "test":
            print(f"Warning: --dataset-name and --dataset-split are ignored when --dataset-config is provided")
        dataset_config = load_dataset_config(args.dataset_config)
        dataset_name = dataset_config['dataset_name']
        dataset_split = dataset_config['split']
        is_local = dataset_config['is_local']
        system_prompt = dataset_config['system_prompt']
        problem_field = dataset_config['problem_field']
        answer_field = dataset_config['answer_field']
        
        if is_local:
            print(f"Dataset: {dataset_name} (local path)")
        else:
            if dataset_split:
                print(f"Dataset: {dataset_name} (split: {dataset_split})")
            else:
                print(f"Dataset: {dataset_name} (no split specified)")
    else:
        dataset_name = args.dataset_name
        dataset_split = args.dataset_split
        system_prompt = None
        problem_field = 'problem'
        answer_field = 'answer'
        print(f"Using default dataset: {dataset_name} (split: {dataset_split})")

    # --- 2. Define models and LoRA adapters to test ---
    MODELS_TO_TEST = args.lora_modules if args.lora_modules else args.model_names
    
    # --- 3. Initialize client and load dataset ---
    api_base_url = f"http://{args.host}:{args.port}/v1"
    client = OpenAI(
        api_key="not-needed-for-local",
        base_url=api_base_url,
        timeout=(30.0, float(args.timeout)),
    )
    print(f"Client ready, will connect to: {api_base_url}")
    print(f"Parallel level set to: {args.concurrency}")
    print(f"Samples per question: {args.num_samples}")
    print(f"Max retries: {args.max_retries}")
    print(f"Incremental write mode: Enabled (results saved in real-time)")
    if args.resume:
        print(f"Checkpoint resume mode: Enabled")

    # Load dataset with flexible handling
    print(f"Loading dataset...")
    try:
        if is_local:
            # 本地数据集
            print(f"  From local path: {dataset_name}")
            if dataset_split:
                dataset = load_dataset(dataset_name, split=dataset_split)
            else:
                dataset = load_dataset(dataset_name)
                # 如果返回的是 DatasetDict，尝试获取第一个可用的split
                if hasattr(dataset, 'keys'):
                    available_splits = list(dataset.keys())
                    print(f"  Available splits: {available_splits}")
                    dataset_split = available_splits[0]
                    print(f"  Using split: {dataset_split}")
                    dataset = dataset[dataset_split]
        else:
            # HuggingFace 数据集
            print(f"  From Hugging Face Hub: {dataset_name}")
            if dataset_split:
                dataset = load_dataset(dataset_name, split=dataset_split)
            else:
                # 不指定 split，让 load_dataset 返回 DatasetDict
                dataset = load_dataset(dataset_name)
                # 如果是 DatasetDict，尝试使用常见的 split
                if hasattr(dataset, 'keys'):
                    available_splits = list(dataset.keys())
                    print(f"  Available splits: {available_splits}")
                    # 优先选择 test > validation > train
                    for preferred_split in ['test', 'validation', 'val', 'train']:
                        if preferred_split in available_splits:
                            dataset_split = preferred_split
                            break
                    else:
                        dataset_split = available_splits[0]
                    print(f"  Auto-selected split: {dataset_split}")
                    dataset = dataset[dataset_split]
        
        print(f"✓ Dataset loaded successfully, contains {len(dataset)} questions.")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        print(f"  Tips:")
        print(f"    - For HuggingFace datasets: check dataset name and split")
        print(f"    - For local datasets: ensure path exists and is valid")
        print(f"    - Try setting 'is_local': true in config for local paths")
        raise
    
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 3. Iterate through all models and LoRAs for parallel inference ---
    for model_name in MODELS_TO_TEST:
        print("\n" + "="*60)
        print(f"Testing model: {model_name}")
        print("="*60)
        
        safe_model_name = model_name.replace("/", "_")
        output_filepath = os.path.join(args.output_dir, f"aime_results_{safe_model_name}.jsonl")
        temp_filepath = os.path.join(args.output_dir, f"aime_temp_{safe_model_name}.jsonl")
        error_filepath = os.path.join(args.output_dir, f"aime_errors_{safe_model_name}.txt")

        # Check if already completed
        if os.path.exists(output_filepath) and not args.resume:
            print(f"Result file {output_filepath} already exists, skipping this model.")
            print(f"Tip: Use --resume parameter to continue from interruption")
            continue
        
        # Load existing results (for checkpoint resume)
        existing_results = {}
        if args.resume and os.path.exists(temp_filepath):
            existing_results = load_existing_results(temp_filepath)
            print(f"Loaded {len(existing_results)} completed samples from temporary file")
        else:
            # Initialize temporary file
            with open(temp_filepath, 'w', encoding='utf-8') as f:
                pass  # Create empty file
        
        # Initialize error log file
        if not os.path.exists(error_filepath):
            with open(error_filepath, 'w', encoding='utf-8') as f:
                f.write(f"Failed sample records - Model: {model_name}\n")
                f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
        
        error_lock = threading.Lock()
        temp_lock = threading.Lock()
        
        # Prepare all tasks: number of problems × number of samples
        all_tasks = []
        for problem_idx, problem in enumerate(dataset):
            messages = create_aime_prompt(problem[problem_field], system_prompt)
            for sample_idx in range(args.num_samples):
                # Skip completed tasks
                result_key = f"{problem_idx}_{sample_idx}"
                if result_key not in existing_results:
                    all_tasks.append((problem_idx, sample_idx, messages, problem))
        
        total_tasks = len(all_tasks)
        already_done = len(existing_results)
        print(f"Total tasks: {len(dataset)} questions × {args.num_samples} samples = {len(dataset) * args.num_samples}")
        if already_done > 0:
            print(f"Already completed: {already_done} samples")
            print(f"To process: {total_tasks} samples")
        
        if total_tasks == 0:
            print("All tasks completed, consolidating results...")
        else:
            # True parallel execution: each sample is an independent task
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                # Submit all sample tasks
                futures = [
                    executor.submit(
                        process_single_sample,
                        problem_idx, sample_idx, messages, client, model_name, args,
                        temp_filepath, temp_lock, error_filepath, error_lock, existing_results
                    )
                    for problem_idx, sample_idx, messages, problem in all_tasks
                ]
                
                # Use tqdm to show overall progress
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=total_tasks,
                    desc=f"Inferencing ({model_name})",
                    initial=0
                ):
                    result = future.result()
                    # Results already written in real-time in process_single_sample

        # --- 4. Consolidate temporary file into final results ---
        print(f"Consolidating results to final file...")
        final_results = consolidate_results(temp_filepath, output_filepath, dataset, args.num_samples, model_name,
                                           problem_field, answer_field)
        
        print(f"Inference results for '{model_name}' successfully saved to: {output_filepath}")
        
        # Count failures
        total_failures = sum(len(result.get('errors', [])) for result in final_results)
        if total_failures > 0:
            print(f"Warning: {total_failures} sample failures, failure records saved to: {error_filepath}")
        else:
            print(f"All samples completed successfully!")
            if os.path.exists(error_filepath):
                # Check if error file only has header
                with open(error_filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                if len(lines) <= 3:  # Only header info
                    os.remove(error_filepath)
        
        # Clean up temporary file
        print(f"Cleaning up temporary file: {temp_filepath}")
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)

    print("\nAll model/LoRA parallel inference tasks completed!")

if __name__ == "__main__":
    main()



