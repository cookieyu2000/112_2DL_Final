import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 讀取數據
data = pd.read_csv('data/augmented_labels.csv')

# 計算各標籤的數量
label_counts = data['label'].value_counts()

# 自定義顯示格式，顯示數量和百分比
def autopct_format(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return '{p:.1f}% ({v:d})'.format(p=pct, v=val)
    return my_autopct

# 圓餅圖顯示
plt.figure(figsize=(10, 5))
plt.pie(label_counts, labels=label_counts.index, autopct=autopct_format(label_counts), startangle=90, counterclock=False)
plt.title('Distribution of Labels')
plt.legend(title="Labels", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
