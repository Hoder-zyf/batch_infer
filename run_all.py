#!/usr/bin/env python3
"""
主控脚本：自动化批量LoRA推理流程
- 遍历所有实验文件夹
- 每5个checkpoint为一组部署vllm服务
- 自动调用推理脚本
"""
import os
import sys
import time
import signal
import socket
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
import argparse


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="批量LoRA推理主控脚本")
    parser.add_argument('--base-dir', type=str, 
                       default="~/yifei/FT_workspace/limo/train/b200_sweep_yamls/saves/qwen3-1.7b/",
                       help="包含所有实验文件夹的基础目录")
    parser.add_argument('--output-base', type=str, default="./aime_results",
                       help="推理结果输出的基础目录")
    parser.add_argument('--gpu_ids', type=str, default="0,1",
                       help="使用的GPU ID，逗号分隔")
    parser.add_argument('--port', type=int, default=8000,
                       help="vLLM服务端口")
    parser.add_argument('--checkpoints-per-group', type=int, default=5,
                       help="每组处理的checkpoint数量")
    parser.add_argument('--concurrency', type=int, default=120,
                       help="推理的并行请求数量")
    parser.add_argument('--num-samples', type=int, default=8,
                       help="每个问题的采样次数")
    parser.add_argument('--max-model-len', type=int, default=40960,
                       help="vLLM的最大模型长度")
    parser.add_argument('--max-lora-rank', type=int, default=32,
                       help="LoRA的最大rank")
    parser.add_argument('--skip-existing', action='store_true',
                       help="跳过已存在结果的实验")
    parser.add_argument('--vllm-conda-env', type=str, default="vllm",
                       help="vLLM服务使用的conda环境名称")
    parser.add_argument('--infer-conda-env', type=str, default="yifei",
                       help="推理脚本使用的conda环境名称")
    parser.add_argument('--tensor-parallel-size', type=int, default=None,
                       help="Tensor parallel size for vLLM (默认自动根据gpu_ids计算)")
    parser.add_argument('--base-model', type=str, default="Qwen/Qwen3-1.7B",
                       help="LoRA的基础模型路径")
    parser.add_argument('--reasoning-parser', type=str, default="qwen3",
                       help="推理解析器类型")
    parser.add_argument('--dataset-config', type=str, default=None,
                       help="数据集配置 JSON 文件路径（完整配置，支持自定义字段映射和提示词）")
    parser.add_argument('--dataset-name', type=str, default="math-ai/aime25",
                       help="数据集名称（快速指定方式，当不使用 --dataset-config 时生效）")
    parser.add_argument('--dataset-split', type=str, default="test",
                       help="数据集分割（快速指定方式，当不使用 --dataset-config 时生效）")
    parser.add_argument('--test-base-model', action='store_true',
                       help="是否也测试 base model（不加载任何 LoRA）")
    
    return parser.parse_args()


def calculate_tensor_parallel_size(gpu_ids: str) -> int:
    """
    根据gpu_ids自动计算tensor parallel size
    
    Args:
        gpu_ids: GPU ID字符串，如 "0,1,2,3"
        
    Returns:
        GPU数量
    """
    gpu_list = [x.strip() for x in gpu_ids.split(',') if x.strip()]
    return len(gpu_list)


def find_available_port(start_port: int, max_tries: int = 100) -> int:
    """Find an available port starting from start_port"""
    port = start_port
    while port < start_port + max_tries:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
        port += 1
    raise RuntimeError(f"No available ports found between {start_port} and {start_port + max_tries}")


def get_experiment_folders(base_dir: Path) -> List[Path]:
    """Get all experiment folders"""
    base_dir = base_dir.expanduser()
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")
    
    # Get all subdirectories
    folders = [f for f in base_dir.iterdir() if f.is_dir()]
    folders.sort()
    
    print(f"Found {len(folders)} experiment folder(s):")
    for i, folder in enumerate(folders, 1):
        print(f"  {i}. {folder.name}")
    
    return folders


def get_checkpoint_groups(exp_folder: Path, group_size: int, 
                         include_exp_name: bool = True) -> List[List[Tuple[str, Path]]]:
    """
    将checkpoints分组 - 自动探索模式
    
    Args:
        exp_folder: 实验文件夹路径
        group_size: 每组大小
        include_exp_name: 是否在名称中包含实验名
    
    Returns:
        List of groups, each group is a list of (name, path) tuples
    """
    groups = []
    current_group = []
    exp_name = exp_folder.name
    
    # 自动扫描 checkpoint 文件夹
    checkpoints = []
    print(f"[DEBUG] Scanning for checkpoints in {exp_folder}...")
    
    for path in exp_folder.iterdir():
        if path.is_dir() and path.name.startswith("checkpoint-"):
            try:
                # Extract step number
                step = int(path.name.split("-")[-1])
                checkpoints.append((step, path))
            except ValueError:
                print(f"Warning: skipping malformed checkpoint folder {path.name}")
                continue
    
    # 按 step 排序
    checkpoints.sort(key=lambda x: x[0])
    
    if not checkpoints:
        print(f"[DEBUG] No checkpoints found starting with 'checkpoint-'")
        return []
        
    print(f"[DEBUG] Found {len(checkpoints)} checkpoints: {[f'ckpt{c[0]}' for c in checkpoints]}")

    for step, checkpoint_path in checkpoints:
        checkpoint_name = checkpoint_path.name
        
        # Build name: include experiment name and checkpoint number
        if include_exp_name:
            name = f"{exp_name}_ckpt{step}"
        else:
            name = f"ckpt{step}"
        
        # print(f"[DEBUG] Adding to group: {name} -> {checkpoint_path}")
        current_group.append((name, checkpoint_path))
        
        if len(current_group) == group_size:
            groups.append(current_group)
            current_group = []
    
    # Add remaining checkpoints
    if current_group:
        # print(f"[DEBUG] Adding remaining group: {len(current_group)} checkpoint(s)")
        groups.append(current_group)
    
    print(f"[DEBUG] Total groups generated: {len(groups)}")
    # for idx, group in enumerate(groups, 1):
    #     print(f"[DEBUG] Group {idx}: {[name for name, _ in group]}")
    
    return groups


def start_vllm_server(checkpoint_group: List[Tuple[str, Path]], args, log_file: Path) -> subprocess.Popen:
    """Start vLLM server (using vllm conda environment)"""
    # Build LoRA module arguments - correct format: --lora-modules name1=path1 name2=path2 ...
    lora_modules = [f'{name}={path}' for name, path in checkpoint_group]
    lora_args = ['--lora-modules'] + lora_modules
    
    # Use conda run to activate specific environment
    cmd = [
        'conda', 'run', '-n', args.vllm_conda_env, '--no-capture-output',
        'vllm', 'serve', args.base_model,
        '--max-model-len', str(args.max_model_len),
        '--enable-lora',
        '--max-lora-rank', str(args.max_lora_rank),
        '--tensor-parallel-size', str(args.tensor_parallel_size),
        *lora_args,
        '--reasoning-parser', args.reasoning_parser,
        '--port', str(args.port)
    ]
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    
    print(f"\nStarting vLLM server...")
    print(f"Conda environment: {args.vllm_conda_env}")
    print(f"Command: {' '.join(cmd)}")
    print(f"GPU devices: {args.gpu_ids}")
    print(f"Log file: {log_file}")
    
    with open(log_file, 'w', encoding='utf-8') as f:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid  # 创建新的进程组
        )
    
    return process


def wait_for_server_ready(port: int, timeout: int = 300, check_interval: int = 5):
    """Wait for vLLM server to be ready - simple fixed wait time"""
    wait_time = 300  # 5 minutes
    
    print(f"Waiting {wait_time} seconds (5 minutes) for vLLM server to fully start and load LoRA models...")
    
    for i in range(wait_time):
        if i % 10 == 0:  # Print every 10 seconds
            elapsed = i
            remaining = wait_time - i
            print(f"  Elapsed: {elapsed}s | Remaining: {remaining}s")
        time.sleep(1)
    
    print("✓ Wait completed, assuming vLLM server is ready!")
    return True


def stop_vllm_server(process: subprocess.Popen):
    """Stop vLLM server and clean up all related processes"""
    if process is None:
        return
    
    print("\nStopping vLLM server...")
    try:
        # 1. First try to gracefully terminate the entire process group
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            print("  Sending SIGTERM signal...")
        except Exception as e:
            print(f"  Warning: Cannot send SIGTERM: {e}")
        
        # 2. Wait for process to end
        try:
            process.wait(timeout=30)
            print("✓ vLLM server stopped gracefully")
        except subprocess.TimeoutExpired:
            print("  Wait timeout, forcefully terminating process...")
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait(timeout=10)
            except Exception:
                pass
        
        # 3. Double insurance: use pkill to clean up all vllm processes
        print("  Cleaning up remaining processes...")
        subprocess.run(['pkill', '-9', '-f', 'vllm serve'], 
                      capture_output=True, timeout=5)
        
        # 4. Wait for GPU resources to be released
        print("  Waiting for GPU resources to be released...")
        time.sleep(3)
        
        print("✓ vLLM server completely stopped")
        
    except Exception as e:
        print(f"Warning: Error stopping server: {e}")
        # Try pkill even if error occurs
        try:
            subprocess.run(['pkill', '-9', '-f', 'vllm serve'], 
                          capture_output=True, timeout=5)
        except Exception:
            pass


def run_inference(checkpoint_group: List[Tuple[str, Path]], output_dir: Path, args) -> bool:
    """Run inference (real-time tqdm progress, using yifei conda environment)"""
    lora_names = [name for name, _ in checkpoint_group]
    
    # Use conda run to activate specific environment
    cmd = [
        'conda', 'run', '-n', args.infer_conda_env, '--no-capture-output',
        'python', '-u',  # Unbuffered mode for real-time output
        'infer.py',
        '--host', 'localhost',
        '--port', str(args.port),
        '--output-dir', str(output_dir),
        '--concurrency', str(args.concurrency),
        '--num-samples', str(args.num_samples),
        '--lora-modules', *lora_names
    ]
    
    # 添加数据集配置参数
    if args.dataset_config:
        cmd.extend(['--dataset-config', args.dataset_config])
    else:
        cmd.extend(['--dataset-name', args.dataset_name])
        cmd.extend(['--dataset-split', args.dataset_split])
    
    print(f"\n{'='*80}")
    print(f"Starting inference")
    print(f"{'='*80}")
    print(f"Conda environment: {args.infer_conda_env}")
    print(f"LoRA modules: {lora_names}")
    print(f"Output directory: {output_dir}")
    print(f"Concurrency: {args.concurrency} | Samples: {args.num_samples}")
    print(f"{'='*80}\n")
    
    try:
        # Don't capture output, let tqdm display directly in terminal
        result = subprocess.run(cmd, check=True)
        print(f"\n{'='*80}")
        print("✓ Inference completed!")
        print(f"{'='*80}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n{'='*80}")
        print(f"✗ Inference failed: {e}")
        print(f"{'='*80}\n")
        return False


def process_experiment(exp_folder: Path, args) -> Dict[str, int]:
    """Process single experiment folder"""
    exp_name = exp_folder.name
    print("\n" + "="*80)
    print(f"Processing experiment: {exp_name}")
    print("="*80)
    
    # Create output directory
    output_base = Path(args.output_base).expanduser() / exp_name
    output_base.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {output_base.absolute()}")
    print(f"   Results will be saved as: aime_results_{{checkpoint_name}}.jsonl")
    
    # Initialize stats
    stats = {'total': 0, 'success': 0, 'failed': 0, 'skipped': 0}
    
    # Test base model if requested
    if args.test_base_model:
        print(f"\n{'='*80}")
        print(f"🔍 Testing Base Model (no LoRA)")
        print(f"{'='*80}")
        
        base_model_name = "base_model"
        safe_name = base_model_name.replace("/", "_")
        result_file = output_base / f"aime_results_{safe_name}.jsonl"
        
        # Check if already exists
        if args.skip_existing and result_file.exists():
            print(f"✓ Base model results already exist, skipping")
            stats['skipped'] += 1
        else:
            stats['total'] += 1
            log_file = output_base / f"vllm_base_model.log"
            vllm_process = None
            
            try:
                # Start vLLM server with base model only (no LoRA)
                cmd = [
                    'conda', 'run', '-n', args.vllm_conda_env, '--no-capture-output',
                    'vllm', 'serve', args.base_model,
                    '--max-model-len', str(args.max_model_len),
                    '--tensor-parallel-size', str(args.tensor_parallel_size),
                    '--reasoning-parser', args.reasoning_parser,
                    '--port', str(args.port)
                ]
                
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
                
                print(f"Starting vLLM server for base model...")
                with open(log_file, 'w', encoding='utf-8') as f:
                    vllm_process = subprocess.Popen(
                        cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                        preexec_fn=os.setsid
                    )
                
                if wait_for_server_ready(args.port, timeout=300):
                    # Run inference with base model
                    base_group = [(base_model_name, Path(args.base_model))]
                    if run_inference(base_group, output_base, args):
                        stats['success'] += 1
                    else:
                        stats['failed'] += 1
                else:
                    print(f"✗ Base model: Server startup failed")
                    stats['failed'] += 1
                    
            except Exception as e:
                print(f"✗ Base model testing failed: {e}")
                stats['failed'] += 1
            finally:
                if vllm_process:
                    stop_vllm_server(vllm_process)
                    time.sleep(5)
        
        print(f"{'='*80}\n")
    
    # 获取checkpoint分组
    groups = get_checkpoint_groups(
        exp_folder, 
        args.checkpoints_per_group
    )
    
    if not groups:
        print(f"Warning: No valid checkpoints found in experiment {exp_name}")
        if stats['total'] == 0:  # No base model tested either
            return stats
    
    if groups:
        print(f"📊 Total {len(groups)} group(s) of checkpoints to process (max 5 per group)")
        for i, group in enumerate(groups, 1):
            ckpt_names = [name for name, _ in group]
            print(f"   Group {i}: {ckpt_names}")
        print()
    
    stats['total'] += len(groups)
    
    for group_idx, group in enumerate(groups, 1):
        print(f"\n{'─'*80}")
        print(f"Processing group {group_idx}/{len(groups)}")
        print(f"Checkpoints: {[name for name, _ in group]}")
        print(f"{'─'*80}")
        
        # Check if all results already exist
        if args.skip_existing:
            all_exist = True
            for name, _ in group:
                safe_name = name.replace("/", "_")
                result_file = output_base / f"aime_results_{safe_name}.jsonl"
                if not result_file.exists():
                    all_exist = False
                    break
            
            if all_exist:
                print(f"✓ Group {group_idx} results already exist, skipping")
                stats['skipped'] += 1
                continue
        
        # 启动vLLM服务
        log_file = output_base / f"vllm_group{group_idx}.log"
        vllm_process = None
        
        try:
            vllm_process = start_vllm_server(group, args, log_file)
            
            # Wait for server to be ready
            if not wait_for_server_ready(args.port, timeout=300):
                print(f"✗ Group {group_idx}: Server startup failed")
                stats['failed'] += 1
                continue
            
            # Run inference
            if run_inference(group, output_base, args):
                stats['success'] += 1
            else:
                stats['failed'] += 1
        
        except Exception as e:
            print(f"✗ Group {group_idx} processing failed: {e}")
            stats['failed'] += 1
        
        finally:
            # 停止vLLM服务
            if vllm_process:
                stop_vllm_server(vllm_process)
                time.sleep(5)  # 等待端口释放
    
    return stats


def main():
    args = parse_args()
    
    # Auto-calculate tensor-parallel-size if not specified
    if args.tensor_parallel_size is None:
        args.tensor_parallel_size = calculate_tensor_parallel_size(args.gpu_ids)
        print(f"Auto-calculated tensor-parallel-size: {args.tensor_parallel_size} (based on {args.gpu_ids})")
    
    # Auto-find available port
    try:
        available_port = find_available_port(args.port)
        if available_port != args.port:
            print(f"Notice: Port {args.port} is busy, switching to available port {available_port}")
        args.port = available_port
    except Exception as e:
        print(f"Error finding available port: {e}")
        return 1
    
    print("="*80)
    print("Batch LoRA Inference Main Control Script")
    print("="*80)
    print(f"Base directory: {args.base_dir}")
    print(f"Output directory: {args.output_base}")
    print(f"GPU devices: {args.gpu_ids}")
    print(f"Checkpoints per group: {args.checkpoints_per_group}")
    print(f"Checkpoint discovery: Auto")
    print(f"Concurrency: {args.concurrency}")
    print(f"Samples per question: {args.num_samples}")
    print("="*80)
    
    # Get all experiment folders
    try:
        exp_folders = get_experiment_folders(Path(args.base_dir))
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    if not exp_folders:
        print("Error: No experiment folders found")
        return 1
    
    # Process each experiment
    total_stats = {'total': 0, 'success': 0, 'failed': 0, 'skipped': 0}
    
    for i, exp_folder in enumerate(exp_folders, 1):
        print(f"\nProgress: [{i}/{len(exp_folders)}] Experiment folder")
        
        try:
            stats = process_experiment(exp_folder, args)
            for key in total_stats:
                total_stats[key] += stats[key]
        except KeyboardInterrupt:
            print("\n\nUser interrupted!")
            return 130
        except Exception as e:
            print(f"\nError: Failed to process experiment {exp_folder.name}: {e}")
            continue
    
    # Print summary
    print("\n" + "="*80)
    print("All experiments processing completed!")
    print("="*80)
    print(f"Total groups: {total_stats['total']}")
    print(f"Successful: {total_stats['success']}")
    print(f"Failed: {total_stats['failed']}")
    print(f"Skipped: {total_stats['skipped']}")
    print("="*80)
    
    # Display result file locations
    print("\n📁 Result file locations:")
    output_base_path = Path(args.output_base).expanduser()
    print(f"   Base directory: {output_base_path.absolute()}")
    print(f"\n   Structure:")
    print(f"   {args.output_base}/")
    for exp_folder in exp_folders:
        exp_name = exp_folder.name
        exp_output = output_base_path / exp_name
        if exp_output.exists():
            jsonl_files = list(exp_output.glob("aime_results_*.jsonl"))
            print(f"   ├── {exp_name}/ ({len(jsonl_files)} result files)")
            if jsonl_files:
                # Display first 3 as examples
                for f in sorted(jsonl_files)[:3]:
                    print(f"   │   ├── {f.name}")
                if len(jsonl_files) > 3:
                    print(f"   │   └── ... ({len(jsonl_files) - 3} more files)")
    
    print("\n💡 Tips:")
    print(f"   - View all results: ls -lh {output_base_path}/*/*.jsonl")
    print(f"   - View error logs: ls -lh {output_base_path}/*/aime_errors_*.txt")
    print(f"   - Use monitoring script: python monitor.py --output-base {args.output_base}")
    print("="*80)
    
    return 0 if total_stats['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

