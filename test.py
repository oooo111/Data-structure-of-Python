plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 22 
plt.rcParams['axes.titlesize'] = 22  
plt.rcParams['axes.labelsize'] = 22 
plt.rcParams['xtick.labelsize'] = 22  
plt.rcParams['ytick.labelsize'] = 22 
plt.rcParams['figure.dpi'] = 800  


def visualize_importance(sequence, importance_scores, seq_id, subtype, output_dir):
    """可视化特征重要性分布"""
    plt.figure(figsize=(20, 6))
    
    # 创建热图
    importance_matrix = importance_scores.reshape(1, -1)
    ax = sns.heatmap(
        importance_matrix,
        cmap='YlOrRd',  # 保持原配色
        xticklabels=False,
        yticklabels=False,
        cbar_kws={'label': 'Feature Importance', 'shrink': 0.8}  # 移除无效的 'labelsize'
    )
    
    # 获取色条并设置标签字体大小
    cbar = ax.collections[0].colorbar
    cbar.set_label('Feature Importance', fontsize=16)  # 手动设置色条标签字体大小
    
    # 设置标题和x轴标签
    plt.title(f"Feature Importance Distribution - {seq_id} ({subtype})", pad=20)
    plt.xlabel("Nucleotide Position")
    
    # 添加位置标记
    seq_length = len(sequence)
    step = 50
    positions = range(0, seq_length, step)
    plt.xticks(positions, positions, rotation=45)
    
    # 添加重要区域标注
    threshold = np.percentile(importance_scores, 90)
    important_regions = np.where(importance_scores > threshold)[0]
    
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
    
    # 保存图片
    output_file = os.path.join(output_dir, f"{subtype}_{seq_id}_importance.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=800, bbox_inches='tight')
    plt.close()
    
    return np.argsort(importance_scores)[-20:]


