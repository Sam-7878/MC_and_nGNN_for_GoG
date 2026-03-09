import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_ngnn_comparison_plot(chain='polygon'):
    RESULT_PATH = f"../../_data/results/fraud_detection_ngnn"
    csv_path = f'{RESULT_PATH}/ngnn_comparison_{chain}.csv'
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: CSV file not found at {csv_path}")
        return

    # 데이터 로드
    df = pd.read_csv(csv_path)
    
    # 중복된 모델 평가가 있다면 가장 최근(마지막) 결과만 유지
    df = df.drop_duplicates(subset=['Model'], keep='last')
    
    # 시각화를 위해 데이터 형태 변환 (Melt)
    df_melted = df.melt(id_vars=['Model'], 
                        value_vars=['Test AUC', 'Test AP', 'Test F1'], 
                        var_name='Metric', value_name='Score')

    # 논문용 스타일 설정
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    plt.figure(figsize=(10, 6))
    
    # 색상 지정 (Base는 회색톤, MC는 강조색)
    palette = {"Base nGNN": "#95a5a6", "MC nGNN (Data+Model)": "#e74c3c"}
    
    ax = sns.barplot(data=df_melted, x='Metric', y='Score', hue='Model', 
                     palette=palette, edgecolor='black', linewidth=1.2)
    
    plt.title(f'Performance Comparison: Base vs MC nGNN ({chain.upper()})', 
              fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Evaluation Metrics', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.ylim(0.0, 1.15) # 최대값을 약간 여유있게 설정
    
    # 막대 위에 수치 표기
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.3f}", 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', 
                        fontsize=12, fontweight='bold',
                        xytext=(0, 5), textcoords='offset points')
            
    plt.legend(title='Architecture', loc='upper right', frameon=True)
    plt.tight_layout()
    
    # 저장
    png_path = f'{RESULT_PATH}/ngnn_comparison_{chain}_paper.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✅ Success! Plot saved as {png_path}")

if __name__ == "__main__":
    generate_ngnn_comparison_plot(chain='polygon')  