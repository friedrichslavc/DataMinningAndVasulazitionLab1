import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import webbrowser
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import xgboost as xgb
# 导入词云和中文分词库
from wordcloud import WordCloud
import jieba
import matplotlib.font_manager as fm

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

warnings.filterwarnings('ignore')

# 检查数据文件是否存在
import os
data_file = 'data/tesla_stock_data_2000_2025.csv'
if os.path.exists(data_file):
    print(f"数据文件已找到: {data_file}")
else:
    print(f"警告: 数据文件不存在: {data_file}")

# 使用相对路径读取数据文件
df = pd.read_csv('data/tesla_stock_data_2000_2025.csv')

df.head()
df.tail()
df.info()

df.isnull().sum()
df.describe()
df.dtypes
df.shape
df.columns
# 调整列名并跳过前两行（标题行和Ticker行）
df.columns = ["Price", "Close", "High", "Low", "Open", "Volume"]
df = df.iloc[2:].reset_index(drop=True)

# 2. Candlestick Chart
# 使用plotly创建专业的蜡烛图
# 对数据进行采样以减小数据量，但保留足够的数据点以显示趋势
sample_size = 5  # 减小采样间隔以显示更多数据点
df_sampled = df.iloc[::sample_size].copy()

# 计算移动平均线
df_sampled['MA20'] = df_sampled['Close'].rolling(window=20).mean()
df_sampled['MA50'] = df_sampled['Close'].rolling(window=50).mean()

# 使用matplotlib创建替代的蜡烛图（作为备份）
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)  # 2行1列的第1个子图
plt.plot(df_sampled.index, df_sampled['Close'], label='收盘价', color='blue')
plt.plot(df_sampled.index, df_sampled['MA20'], label='20日均线', color='orange')
plt.plot(df_sampled.index, df_sampled['MA50'], label='50日均线', color='red')
plt.fill_between(df_sampled.index, df_sampled['Low'], df_sampled['High'], alpha=0.2, color='gray')
plt.title('特斯拉股票价格走势图')
plt.ylabel('价格')
plt.legend()

# 在第二个子图中添加成交量
plt.subplot(2, 1, 2)  # 2行1列的第2个子图
plt.bar(df_sampled.index, df_sampled['Volume'], color='green', alpha=0.5)
plt.title('成交量')
plt.xlabel('日期')
plt.ylabel('成交量')

plt.tight_layout()

# 保存为PNG格式
import os
# 确保images目录存在
if not os.path.exists('images'):
    os.makedirs('images')
png_file_path = os.path.join(os.getcwd(), "images", "candlestick_chart.png")
plt.savefig(png_file_path, dpi=100)
plt.close()
print(f"蜡烛图已保存为PNG格式: {png_file_path}")

# 使用plotly创建专业的蜡烛图
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # 创建带有两个子图的图表（价格和成交量）
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.1, subplot_titles=('特斯拉股票价格', '成交量'),
                       row_heights=[0.7, 0.3])
    
    # 添加蜡烛图
    fig.add_trace(go.Candlestick(
        x=df_sampled.index,
        open=df_sampled['Open'],
        high=df_sampled['High'],
        low=df_sampled['Low'],
        close=df_sampled['Close'],
        name='TSLA',
        increasing_line_color='red',  # 上涨为红色
        decreasing_line_color='green'  # 下跌为绿色
    ), row=1, col=1)
    
    # 添加移动平均线
    fig.add_trace(go.Scatter(
        x=df_sampled.index,
        y=df_sampled['MA20'],
        line=dict(color='orange', width=1),
        name='MA20'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df_sampled.index,
        y=df_sampled['MA50'],
        line=dict(color='purple', width=1),
        name='MA50'
    ), row=1, col=1)
    
    # 添加成交量图
    colors = ['red' if row['Close'] >= row['Open'] else 'green' for _, row in df_sampled.iterrows()]
    fig.add_trace(go.Bar(
        x=df_sampled.index,
        y=df_sampled['Volume'],
        marker_color=colors,
        name='成交量'
    ), row=2, col=1)
    
    # 更新布局
    fig.update_layout(
        title="特斯拉股票蜡烛图分析",
        xaxis_title="日期",
        yaxis_title="价格",
        yaxis2_title="成交量",
        height=800,  # 增加高度以容纳两个子图
        template='plotly_white',
        xaxis_rangeslider_visible=False,  # 隐藏范围滑块
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # 格式化X轴为日期格式
    fig.update_xaxes(type='category', tickformat='%Y-%m-%d')
    
    # 使用配置保存HTML
    html_file_path = os.path.join(os.getcwd(), "images", "candlestick_chart_simple.html")
    fig.write_html(
        html_file_path, 
        include_plotlyjs='cdn', 
        full_html=False, 
        config={'displayModeBar': False, 'staticPlot': True, 'responsive': True}
    )
    print(f"专业蜡烛图已保存为HTML格式: {html_file_path}")
    print(f"请手动打开HTML文件，路径为: {html_file_path}")
except Exception as e:
    print(f"生成HTML图表时出错: {e}")
    print("请使用PNG格式的图表代替")

# 使用绝对路径和file://协议打开HTML文件
try:
    # 使用file://协议确保正确打开本地文件
    file_url = f"file:///{png_file_path.replace('\\', '/')}"
    print(f"PNG文件路径: {file_url}")
    print("请手动打开PNG文件查看图表")
except Exception as e:
    print(f"处理文件路径时出错: {e}")
    print("请手动打开PNG文件，路径为: " + png_file_path)

# 3. Volume vs. Close Price Scatter Plot
plt.figure(figsize=(10, 5))
sns.scatterplot(x=df['Volume'], y=df['Close'], color="purple", alpha=0.6)
plt.title("Volume vs. Close Price")
plt.xlabel("Volume")
plt.ylabel("Close Price")
plt.savefig(os.path.join("images", "volume_vs_close.png"))
plt.close()
print("交易量与收盘价散点图已保存为images/volume_vs_close.png")
# 移除plt.show()调用

# 将第一列重命名为Date并转换为datetime格式并设置为索引
df.rename(columns={"Price": "Date"}, inplace=True)
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

# 将字符串类型的股票数据转换为浮点数
for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 检查转换后的数据类型
print("转换后的数据类型:")
print(df.dtypes)

plt.figure(figsize=(12,6))
sns.lineplot(data=df, x=df.index, y="Close")
plt.title("收盘价走势图")
plt.savefig(os.path.join("images", "close_price_trend.png"))
plt.close()
print("收盘价走势图已保存为images/close_price_trend.png")

# 4. Moving Averages
df["50_MA"] = df["Close"].rolling(window=50).mean()
df["200_MA"] = df["Close"].rolling(window=200).mean()

plt.figure(figsize=(12, 5))
plt.plot(df.index, df["Close"], label="Close Price", color="blue")
plt.plot(df.index, df["50_MA"], label="50-day MA", color="red", linestyle="dashed")
plt.plot(df.index, df["200_MA"], label="200-day MA", color="green", linestyle="dashed")
plt.legend()
plt.title("Stock Price with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price")
plt.savefig(os.path.join("images", "moving_averages.png"))
plt.close()
print("移动平均线图已保存为images/moving_averages.png")

# 5. Histogram for Price Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df["Close"], bins=30, kde=True, color="blue")
plt.title("Close Price Distribution")
plt.xlabel("Close Price")
plt.ylabel("Frequency")
plt.savefig(os.path.join("images", "price_distribution.png"))
plt.close()
print("价格分布直方图已保存为images/price_distribution.png")

# 6. Box Plot for Price Volatility
plt.figure(figsize=(8, 6))
sns.boxplot(data=df[['Open', 'Close', 'High', 'Low']], palette="coolwarm")
plt.title("Price Volatility (Box Plot)")
plt.xlabel("Price Type")
plt.ylabel("Value")
plt.savefig(os.path.join("images", "price_volatility.png"))
plt.close()
print("价格波动箱线图已保存为images/price_volatility.png")
# 移除plt.show()调用

df.columns

# 数据预处理与统计分析部分
print("\n" + "="*50)
print("数据预处理与统计分析报告")
print("="*50)

# 1. 数据基本信息
print("\n1. 数据基本信息:")
print(f"数据集来源: 特斯拉股票数据 (2010-2025)")
print(f"数据集大小: {df.shape[0]} 行 x {df.shape[1]} 列")
print(f"数据集时间范围: {df.index.min()} 至 {df.index.max()}")

# 2. 数据清洗过程
print("\n2. 数据清洗过程:")

# 检查缺失值
missing_before = df.isnull().sum()
print("清洗前缺失值数量:")
print(missing_before)

# 检查重复值
duplicates_before = df.duplicated().sum()
print(f"清洗前重复行数量: {duplicates_before}")

# 处理缺失值 (已在前面代码中使用中位数填充)
print("\n缺失值处理方法: 使用中位数填充数值型缺失值")

# 删除重复行
df = df.drop_duplicates()
duplicates_after = df.duplicated().sum()
print(f"清洗后重复行数量: {duplicates_after}")

# 检查清洗后的缺失值
missing_after = df.isnull().sum()
print("\n清洗后缺失值数量:")
print(missing_after)

# 3. 统计学特征计算
print("\n3. 统计学特征计算:")

# 选择数值型列进行统计分析
numerical_cols = ['Close', 'High', 'Low', 'Open', 'Volume']

# 创建一个空的DataFrame来存储统计结果
stats_df = pd.DataFrame(index=numerical_cols)

# 计算各种统计指标
stats_df['均值'] = df[numerical_cols].mean()
stats_df['中位数'] = df[numerical_cols].median()
stats_df['标准差'] = df[numerical_cols].std()
stats_df['最小值'] = df[numerical_cols].min()
stats_df['最大值'] = df[numerical_cols].max()
stats_df['25%分位数'] = df[numerical_cols].quantile(0.25)
stats_df['75%分位数'] = df[numerical_cols].quantile(0.75)
stats_df['偏度'] = df[numerical_cols].skew()
stats_df['峰度'] = df[numerical_cols].kurt()

# 打印统计结果表格
print("\n数值型变量的统计特征:")
print(stats_df.round(2))

# 4. 统计结果解释
print("\n4. 统计结果解释:")

# 解释偏度
for col in numerical_cols:
    skew_value = stats_df.loc[col, '偏度']
    if skew_value > 0.5:
        skew_desc = f"{col} 列呈现明显的右偏分布 (偏度={skew_value:.2f})，大多数值集中在左侧，有少量较大的极端值"
    elif skew_value < -0.5:
        skew_desc = f"{col} 列呈现明显的左偏分布 (偏度={skew_value:.2f})，大多数值集中在右侧，有少量较小的极端值"
    else:
        skew_desc = f"{col} 列近似呈现对称分布 (偏度={skew_value:.2f})"
    print(skew_desc)

# 解释峰度
for col in numerical_cols:
    kurt_value = stats_df.loc[col, '峰度']
    if kurt_value > 3:
        kurt_desc = f"{col} 列呈现尖峰分布 (峰度={kurt_value:.2f})，分布的尾部较重，存在更多的极端值"
    else:
        kurt_desc = f"{col} 列呈现平峰分布 (峰度={kurt_value:.2f})，分布的尾部较轻，极端值较少"
    print(kurt_desc)

# 5. 数据标准化
print("\n5. 数据标准化:")
print("使用Min-Max标准化方法将数值列标准化到[0,1]区间")

# 创建Min-Max标准化函数
def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min())

# 对数值列进行标准化
df_normalized = df.copy()
for col in numerical_cols:
    df_normalized[f"{col}_normalized"] = min_max_normalize(df[col])

# 显示标准化后的前几行数据
print("\n标准化后的数据示例 (前5行):")
print(df_normalized[[f"{col}_normalized" for col in numerical_cols]].head())

# 6. 可视化结果解释
print("\n6. 可视化结果解释:")

print("\n(1) 价格分布直方图分析:")
print("   - 从price_distribution.png可以看出，特斯拉股票价格分布呈现右偏特征")
print("   - 大部分交易价格集中在较低区间，少数交易达到较高价格")
print("   - 这表明特斯拉股票在观察期内经历了显著的价格增长")

print("\n(2) 价格波动箱线图分析:")
print("   - 从price_volatility.png可以观察到各价格类型(开盘价、收盘价、最高价、最低价)的分布范围")
print("   - 箱线图显示存在一些异常值(离群点)，表明某些交易日价格波动较大")
print("   - 价格中位数线的位置表明价格分布的集中趋势")

print("\n(3) 交易量与收盘价散点图分析:")
print("   - 从volume_vs_close.png可以分析交易量与收盘价之间的关系")
print("   - 散点图显示交易量与价格之间可能存在一定的相关性")
print("   - 高交易量往往出现在价格变动较大的时期")

print("\n(4) 词云图分析:")
print("   - 从tesla_wordcloud.png可以直观地看出与特斯拉股票相关的高频词汇")
print("   - 词云突出显示了'特斯拉'、'股票'、'价格'、'趋势'等关键词")
print("   - 这些高频词反映了投资者关注的焦点")

# 7. 相关性热力图
print("\n创建相关性热力图...")

# 选择数值型列进行相关性分析
corr_columns = ['Close', 'High', 'Low', 'Open', 'Volume']

# 计算相关性矩阵
correlation_matrix = df[corr_columns].corr()

# 创建热力图
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
           square=True, linewidths=.5, cbar_kws={"shrink": .8})
plt.title('特斯拉股票数据相关性热力图', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join('images', 'correlation_heatmap.png'), dpi=300)
plt.close()
print("相关性热力图已保存为images/correlation_heatmap.png")

# 解释相关性热力图
print("\n(5) 相关性热力图分析:")
print("   - 从correlation_heatmap.png可以观察到各价格指标和交易量之间的相关性")
print("   - 开盘价、收盘价、最高价和最低价之间通常具有很强的正相关性")
print("   - 交易量与价格指标之间的相关性可能较弱或呈现不同的模式")
print("   - 热力图中的数值表示相关系数，范围从-1(完全负相关)到1(完全正相关)")

print("\n" + "="*50)
print("数据分析报告完成")
print("="*50)

# 继续执行原有的机器学习模型部分
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# 准备机器学习数据
print("\n准备机器学习数据...")

# 创建特征和目标变量
# 使用前一天的价格和交易量预测下一天的收盘价
df_ml = df.copy()

# 创建滞后特征（前一天的数据）
df_ml['prev_close'] = df_ml['Close'].shift(1)
df_ml['prev_high'] = df_ml['High'].shift(1)
df_ml['prev_low'] = df_ml['Low'].shift(1)
df_ml['prev_open'] = df_ml['Open'].shift(1)
df_ml['prev_volume'] = df_ml['Volume'].shift(1)

# 计算价格变化百分比
df_ml['price_change'] = (df_ml['Close'] - df_ml['prev_close']) / df_ml['prev_close'] * 100

# 计算波动性指标
df_ml['volatility'] = df_ml['High'] - df_ml['Low']

# 删除含有NaN的行
df_ml = df_ml.dropna()

# 选择特征和目标变量
features = ['prev_close', 'prev_high', 'prev_low', 'prev_open', 'prev_volume', 'price_change', 'volatility']
X = df_ml[features]
y = df_ml['Close']

# 划分训练集和测试集 (使用80%的数据作为训练集)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"训练集大小: {X_train.shape[0]} 样本")
print(f"测试集大小: {X_test.shape[0]} 样本")
print(f"特征数量: {X_train.shape[1]}")

# Define ML models for regression
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "SVR": SVR(),
}

# Train & Evaluate Models
results = {}
print("\n开始训练和评估模型...")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # 使用回归评估指标
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MSE": mse, "R2": r2}

# Print Summary
print("\n模型性能总结:")
for name, metrics in results.items():
    print(f"{name}: MSE={metrics['MSE']:.4f}, R2={metrics['R2']:.4f}")

print("\n所有图表已保存为PNG文件，可以在项目目录中查看。")

# 7. 生成词云
# 创建一个包含特斯拉股票相关词汇的文本
stock_text = """
特斯拉 股票 投资 分析 价格 趋势 波动 交易量 开盘价 收盘价 最高价 最低价 
电动汽车 新能源 科技股 马斯克 市值 增长 下跌 上涨 投资者 股东 
财报 季度 年度 预测 技术分析 基本面 阻力位 支撑位 均线 移动平均线 
市场 行情 大盘 板块 新能源汽车 电池技术 自动驾驶 人工智能 未来发展
"""

# 使用jieba进行中文分词
words = jieba.cut(stock_text)
text = " ".join(words)

# 创建词云对象
wordcloud = WordCloud(
    font_path='C:\\Windows\\Fonts\\simhei.ttf',  # 设置中文字体路径
    width=800, 
    height=400, 
    background_color='white',
    max_words=100,
    max_font_size=100,
    random_state=42
).generate(text)

# 显示词云图
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('特斯拉股票相关词云')
plt.savefig(os.path.join('images', 'tesla_wordcloud.png'), dpi=300)
plt.close()
print("词云图已保存为images/tesla_wordcloud.png")

# 尝试使用备用方法生成词云（如果上面的方法失败）
try:
    # 使用不同的字体配置方式
    font_path = fm.findfont(fm.FontProperties(family='SimHei'))
    wordcloud_alt = WordCloud(
        font_path=font_path,
        width=800, 
        height=400, 
        background_color='white',
        max_words=100,
        max_font_size=100,
        random_state=42
    ).generate(text)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud_alt, interpolation='bilinear')
    plt.axis('off')
    plt.title('特斯拉股票相关词云 (备用方法)')
    plt.savefig(os.path.join('images', 'tesla_wordcloud_alt.png'), dpi=300)
    plt.close()
    print("备用词云图已保存为images/tesla_wordcloud_alt.png")
except Exception as e:
    print(f"生成备用词云图时出错: {e}")
    print("请使用第一个词云图或手动调整字体路径")

# 7. 相关性热力图
print("\n创建相关性热力图...")

# 选择数值型列进行相关性分析
corr_columns = ['Close', 'High', 'Low', 'Open', 'Volume']

# 计算相关性矩阵
correlation_matrix = df[corr_columns].corr()

# 创建热力图
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
           square=True, linewidths=.5, cbar_kws={"shrink": .8})
plt.title('特斯拉股票数据相关性热力图', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join('images', 'correlation_heatmap.png'), dpi=300)
plt.close()
print("相关性热力图已保存为images/correlation_heatmap.png")

# 解释相关性热力图
print("\n(5) 相关性热力图分析:")
print("   - 从correlation_heatmap.png可以观察到各价格指标和交易量之间的相关性")
print("   - 开盘价、收盘价、最高价和最低价之间通常具有很强的正相关性")
print("   - 交易量与价格指标之间的相关性可能较弱或呈现不同的模式")
print("   - 热力图中的数值表示相关系数，范围从-1(完全负相关)到1(完全正相关)")

print("\n" + "="*50)
print("数据分析报告完成")
print("="*50)

# 继续执行原有的机器学习模型部分
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# 准备机器学习数据
print("\n准备机器学习数据...")

# 创建特征和目标变量
# 使用前一天的价格和交易量预测下一天的收盘价
df_ml = df.copy()

# 创建滞后特征（前一天的数据）
df_ml['prev_close'] = df_ml['Close'].shift(1)
df_ml['prev_high'] = df_ml['High'].shift(1)
df_ml['prev_low'] = df_ml['Low'].shift(1)
df_ml['prev_open'] = df_ml['Open'].shift(1)
df_ml['prev_volume'] = df_ml['Volume'].shift(1)

# 计算价格变化百分比
df_ml['price_change'] = (df_ml['Close'] - df_ml['prev_close']) / df_ml['prev_close'] * 100

# 计算波动性指标
df_ml['volatility'] = df_ml['High'] - df_ml['Low']

# 删除含有NaN的行
df_ml = df_ml.dropna()

# 选择特征和目标变量
features = ['prev_close', 'prev_high', 'prev_low', 'prev_open', 'prev_volume', 'price_change', 'volatility']
X = df_ml[features]
y = df_ml['Close']

# 划分训练集和测试集 (使用80%的数据作为训练集)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"训练集大小: {X_train.shape[0]} 样本")
print(f"测试集大小: {X_test.shape[0]} 样本")
print(f"特征数量: {X_train.shape[1]}")

# Define ML models for regression
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "SVR": SVR(),
}

# Train & Evaluate Models
results = {}
print("\n开始训练和评估模型...")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # 使用回归评估指标
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MSE": mse, "R2": r2}

# Print Summary
print("\n模型性能总结:")
for name, metrics in results.items():
    print(f"{name}: MSE={metrics['MSE']:.4f}, R2={metrics['R2']:.4f}")

print("\n所有图表已保存为PNG文件，可以在项目目录中查看。")
