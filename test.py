import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from tqdm import tqdm


plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 22 
plt.rcParams['axes.titlesize'] = 22 
plt.rcParams['axes.labelsize'] = 22 
plt.rcParams['xtick.labelsize'] = 22 
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['figure.dpi'] = 800 


def load_data(fasta_file, label_file, existing_label_to_idx=None):
    """加载序列数据和标签"""
    sequences = []
    seq_ids = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))
        main_id = record.id.split('|')[0]
        seq_ids.append(main_id)
    
    print(f"加载了 {len(sequences)} 条序列")
    
    labels_df = pd.read_csv(label_file)
    print(f"标签文件包含 {len(labels_df)} 条记录")
    id_to_label = dict(zip(labels_df['accession'], labels_df['subtype']))
    
    labels = []
    valid_seq_ids = []
    valid_sequences = []
    
    for i, seq_id in enumerate(seq_ids):
        if seq_id in id_to_label:
            labels.append(id_to_label[seq_id])
            valid_seq_ids.append(seq_id)
            valid_sequences.append(sequences[i])
    
    print(f"成功匹配了 {len(labels)} 条序列的标签")
    
    if existing_label_to_idx is None:
        unique_labels = sorted(set(labels))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    else:
        label_to_idx = existing_label_to_idx
    
    numeric_labels = [label_to_idx[label] for label in labels]
    
    return valid_sequences, numeric_labels, valid_seq_ids, label_to_idx

def analyze_feature_importance(model, sequence, tokenizer, label_idx, device, window_size=5):
    """分析序列特征重要性（使用窗口遮蔽分析）"""
    model.eval()
    
    # 基准预测
    with torch.no_grad():
        base_encoding = tokenizer(
            sequence,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        base_output = model(**base_encoding)
        base_prob = torch.softmax(base_output.logits, dim=-1)[0, label_idx].item()
    
    # 初始化窗口重要性分数数组
    window_importance = np.zeros(len(sequence) - window_size + 1)
    
    # 对每个窗口进行遮蔽分析
    for i in tqdm(range(len(sequence) - window_size + 1), desc="分析窗口重要性"):
        # 创建遮蔽序列
        masked_seq = list(sequence)
        for j in range(i, i + window_size):
            masked_seq[j] = 'N'
        masked_seq = ''.join(masked_seq)
        
        # 计算遮蔽后的预测概率
        with torch.no_grad():
            masked_encoding = tokenizer(
                masked_seq,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            masked_output = model(**masked_encoding)
            masked_prob = torch.softmax(masked_output.logits, dim=-1)[0, label_idx].item()
        
        # 计算窗口重要性
        window_importance[i] = abs(base_prob - masked_prob)
    
    return window_importance

def visualize_importance(sequence, window_importance, seq_id, subtype, output_dir, window_size=5):
    """可视化窗口重要性分布"""
    plt.figure(figsize=(20, 6))
    # 创建热图
    importance_matrix = window_importance.reshape(1, -1)
    sns.heatmap(
        importance_matrix,
        cmap='YlOrRd',
        xticklabels=False,
        yticklabels=False,
        cbar_kws={'label': 'Window Importance', 'shrink': 0.8},
    )
    
    plt.title(f"Window Importance Distribution - {seq_id} ({subtype})")
    plt.xlabel(f"Window Position (size={window_size})", fontsize=22)
    
    # 添加位置标记
    seq_length = len(sequence) - window_size + 1
    step = 50
    positions = range(0, seq_length, step)
    plt.xticks(positions, positions, fontsize=22, rotation=45)  # Rotate X-axis labels by 45 degrees
    
    # 添加重要区域标注
    threshold = np.percentile(window_importance, 90)
    important_regions = np.where(window_importance > threshold)[0]
    
    if len(important_regions) > 0:
        current_start = important_regions[0]
        current_end = current_start
        
        for i in range(1, len(important_regions)):
            if important_regions[i] == current_end + 1:
                current_end = important_regions[i]
            else:
                plt.axvspan(current_start, current_end, color='blue', alpha=0.3)
                current_start = important_regions[i]
                current_end = current_start
        
        plt.axvspan(current_start, current_end, color='blue', alpha=0.3)
    
    output_file = os.path.join(output_dir, f"{subtype}_{seq_id}_importance.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=800, bbox_inches='tight')
    plt.close()
    
    return np.argsort(window_importance)[-20:]

def analyze_sequences(sequences, labels, seq_ids, model, tokenizer, label_to_idx, output_dir, device):
    """分析所有序列的特征重要性"""
    os.makedirs(output_dir, exist_ok=True)
    subtype_patterns = {}
    
    for i, (sequence, label, seq_id) in enumerate(zip(sequences, labels, seq_ids)):
        subtype = [k for k, v in label_to_idx.items() if v == label][0]
        print(f"\n分析序列 {i+1}/{len(sequences)}: {seq_id} ({subtype})")
        
        try:
            window_importance = analyze_feature_importance(model, sequence, tokenizer, label, device)
            
            if window_importance is not None:
                important_windows = visualize_importance(
                    sequence, window_importance, seq_id, subtype, output_dir
                )
                
                if subtype not in subtype_patterns:
                    subtype_patterns[subtype] = []
                
                for window_start in important_windows:
                    window_seq = sequence[window_start:window_start+5]
                    subtype_patterns[subtype].append({
                        'window_start': window_start,
                        'window_sequence': window_seq,
                        'importance': float(window_importance[window_start])
                    })
            else:
                print(f"跳过序列 {seq_id}：无法计算特征重要性")
                        
        except Exception as e:
            print(f"处理序列 {seq_id} 时发生错误: {str(e)}")
            continue
    
    return subtype_patterns

def generate_summary_report(subtype_patterns, output_dir):
    """生成分析报告"""
    report_file = os.path.join(output_dir, "feature_importance_summary.txt")
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Window Importance Analysis Summary\n\n")
        
        for subtype, patterns in subtype_patterns.items():
            f.write(f"## {subtype} Analysis\n\n")
            
            # 统计重要窗口信息
            window_stats = {}
            for pattern in patterns:
                window_start = pattern['window_start']
                if window_start not in window_stats:
                    window_stats[window_start] = {
                        'count': 0,
                        'sequences': {},
                        'avg_importance': 0
                    }
                
                window_stats[window_start]['count'] += 1
                window_seq = pattern['window_sequence']
                if window_seq not in window_stats[window_start]['sequences']:
                    window_stats[window_start]['sequences'][window_seq] = 0
                window_stats[window_start]['sequences'][window_seq] += 1
                window_stats[window_start]['avg_importance'] += pattern['importance']
            
            # 计算平均值并排序
            for window_start in window_stats:
                window_stats[window_start]['avg_importance'] /= window_stats[window_start]['count']
            
            sorted_windows = sorted(
                window_stats.items(),
                key=lambda x: x[1]['avg_importance'],
                reverse=True
            )
            
            f.write("Window Start\tImportance\tSequence Distribution\n")
            for window_start, stats in sorted_windows[:30]:
                seq_dist = ", ".join([
                    f"{seq}:{cnt}" for seq, cnt in stats['sequences'].items()
                ])
                f.write(f"{window_start}\t{stats['avg_importance']:.4f}\t{seq_dist}\n")
            
            f.write("\n")
    
    print(f"已生成汇总报告: {report_file}")
