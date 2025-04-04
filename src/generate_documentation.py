import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64
import io


def extract_code_block(file_path, start_marker, end_marker=None, line_limit=30):
    """从文件中提取代码块"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 处理特殊情况：如果start_marker包含换行符或特殊字符，需要转义
    escaped_start_marker = re.escape(start_marker)
    
    if end_marker:
        escaped_end_marker = re.escape(end_marker)
        pattern = f"{escaped_start_marker}[\\s\\S]*?{escaped_end_marker}"
    else:
        pattern = f"{escaped_start_marker}[\\s\\S]*?"
    
    matches = re.findall(pattern, content)
    if matches:
        code_block = matches[0]
        # 限制代码行数
        lines = code_block.split('\n')
        if len(lines) > line_limit:
            return '\n'.join(lines[:line_limit]) + '\n# ... 更多代码省略 ...'
        return code_block
    
    # 如果没有找到匹配，尝试使用更宽松的匹配方式
    if '\n' in start_marker:
        # 对于多行标记，尝试只匹配第一行
        first_line = start_marker.split('\n')[0]
        return extract_code_block(file_path, first_line, end_marker, line_limit)
    
    return "未找到相关代码"


def get_image_base64(image_path):
    """将图像转换为base64编码"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"无法读取图像 {image_path}: {e}")
        return ""


def generate_html_documentation():
    """生成HTML格式的文档"""
    main_py_path = os.path.join(os.path.dirname(__file__), "main.py")
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "visualization_documentation.html")
    
    # 检查main.py是否存在
    if not os.path.exists(main_py_path):
        print(f"错误: {main_py_path} 文件不存在")
        return
    
    # 图表信息
    # 获取项目根目录路径
    project_root = os.path.dirname(os.path.dirname(__file__))
    images_dir = os.path.join(project_root, "images")
    
    charts = [
        {
            "name": "蜡烛图",
            "file": os.path.join(images_dir, "candlestick_chart.png"),
            "html_file": os.path.join(images_dir, "candlestick_chart_simple.html"),
            "code_marker": "# 2. Candlestick Chart",
            "end_marker": "# 3. Volume vs. Close Price Scatter Plot",
            "description": "展示了特斯拉股票的价格走势，包括开盘价、收盘价、最高价和最低价。红色蜡烛表示上涨，绿色蜡烛表示下跌。"
        },
        {
            "name": "交易量与收盘价散点图",
            "file": os.path.join(images_dir, "volume_vs_close.png"),
            "code_marker": "# 3. Volume vs. Close Price Scatter Plot",
            "end_marker": "# 将第一列重命名为Date",
            "description": "展示了交易量与收盘价之间的关系，帮助分析交易量与价格变动的相关性。"
        },
        {
            "name": "收盘价走势图",
            "file": os.path.join(images_dir, "close_price_trend.png"),
            "code_marker": "plt.figure(figsize=(12,6))",
            "end_marker": "# 4. Moving Averages",
            "description": "展示了特斯拉股票收盘价随时间的变化，可以清晰地显示长期趋势。"
        },
        {
            "name": "移动平均线图",
            "file": os.path.join(images_dir, "moving_averages.png"),
            "code_marker": "# 4. Moving Averages",
            "end_marker": "# 5. Histogram for Price Distribution",
            "description": "展示了股票价格与50日和200日移动平均线的关系，帮助识别中长期趋势。"
        },
        {
            "name": "价格分布直方图",
            "file": os.path.join(images_dir, "price_distribution.png"),
            "code_marker": "# 5. Histogram for Price Distribution",
            "end_marker": "# 6. Box Plot for Price Volatility",
            "description": "展示了特斯拉股票收盘价的频率分布，帮助分析价格的集中趋势和分散程度。"
        },
        {
            "name": "价格波动箱线图",
            "file": os.path.join(images_dir, "price_volatility.png"),
            "code_marker": "# 6. Box Plot for Price Volatility",
            "end_marker": "# 移除plt.show()调用",
            "description": "展示了开盘价、收盘价、最高价和最低价的分布情况，帮助分析价格的波动范围和异常值。"
        },
        {
            "name": "相关性热力图",
            "file": os.path.join(images_dir, "correlation_heatmap.png"),
            "code_marker": "# 7. 相关性热力图",
            "end_marker": "# 解释相关性热力图",
            "description": "展示了各价格指标和交易量之间的相关性强度，帮助分析变量间的关系。"
        },
        {
            "name": "特斯拉股票词云图",
            "file": os.path.join(images_dir, "tesla_wordcloud.png"),
            "code_marker": "# 7. 生成词云",
            "end_marker": "# 尝试使用备用方法生成词云",
            "description": "直观地展示了与特斯拉股票相关的高频词汇，反映了投资者关注的焦点。"
        }
    ]
    
    # 提取统计特征计算代码
    stats_code = extract_code_block(main_py_path, "# 3. 统计学特征计算:", "# 4. 统计结果解释")
    
    # 如果没有找到统计特征计算代码，尝试不带冒号的标记
    if stats_code == "未找到相关代码":
        stats_code = extract_code_block(main_py_path, "# 3. 统计学特征计算", "# 4. 统计结果解释")
    
    # HTML文档头部
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>数据可视化文档</title>
        <style>
            body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #2980b9; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 10px; }}
            h3 {{ color: #3498db; }}
            .chart-container {{ background-color: #f9f9f9; border-radius: 5px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .chart-image {{ max-width: 100%; height: auto; display: block; margin: 20px auto; border: 1px solid #ddd; }}
            pre {{ background-color: #f8f8f8; border: 1px solid #ddd; border-radius: 3px; padding: 15px; overflow-x: auto; font-family: Consolas, Monaco, 'Andale Mono', monospace; }}
            code {{ font-family: Consolas, Monaco, 'Andale Mono', monospace; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .analysis {{ background-color: #e8f4fc; padding: 15px; border-radius: 5px; margin-top: 15px; }}
            .toc {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            .toc ul {{ list-style-type: none; padding-left: 20px; }}
            .toc a {{ text-decoration: none; color: #3498db; }}
            .toc a:hover {{ text-decoration: underline; }}
            .stats {{ background-color: #f0f7fa; padding: 15px; border-radius: 5px; margin-top: 15px; }}
        </style>
    </head>
    <body>
        <h1>数据可视化文档</h1>
        <p>本文档详细记录了特斯拉股票数据分析过程中生成的各个图表，包括预处理步骤、统计学特征计算结果和可视化分析。</p>
        
        <div class="toc">
            <h2>目录</h2>
            <ul>
    """
    
    # 生成目录
    for i, chart in enumerate(charts, 1):
        html_content += f'                <li><a href="#chart-{i}">{chart["name"]}</a></li>\n'
    
    html_content += """
            </ul>
        </div>
    """
    
    # 添加统计学特征计算部分
    html_content += """
        <h2 id="stats">统计学特征计算结果</h2>
        <div class="stats">
            <h3>数值型变量的统计特征</h3>
            <pre><code>{}</code></pre>
            
            <p>统计特征解释：</p>
            <ul>
                <li><strong>均值</strong>：数据的平均值，反映数据的集中趋势</li>
                <li><strong>中位数</strong>：将数据排序后的中间值，不受极端值影响</li>
                <li><strong>标准差</strong>：反映数据的离散程度，值越大表示数据越分散</li>
                <li><strong>最小值/最大值</strong>：数据的范围边界</li>
                <li><strong>分位数</strong>：25%和75%分位数反映数据的分布特征</li>
                <li><strong>偏度</strong>：描述分布的对称性，正值表示右偏，负值表示左偏</li>
                <li><strong>峰度</strong>：描述分布的尖峰程度，值越大表示分布越尖锐</li>
            </ul>
        </div>
    """.format(stats_code)
    
    # 为每个图表生成文档
    for i, chart in enumerate(charts, 1):
        code_block = extract_code_block(main_py_path, chart["code_marker"], chart["end_marker"])
        
        html_content += f"""
        <h2 id="chart-{i}">{chart["name"]}</h2>
        <div class="chart-container">
            <h3>预处理步骤和代码截图</h3>
            <pre><code>{code_block}</code></pre>
            
            <h3>可视化图表</h3>
        """
        
        # 添加图片
        if "html_file" in chart and os.path.exists(chart["html_file"]):
            # 对于HTML格式的图表，嵌入iframe
            html_content += f"""
            <iframe src="{chart['html_file']}" width="100%" height="600" frameborder="0"></iframe>
            """
        elif os.path.exists(chart["file"]):
            # 对于图片格式的图表，嵌入图片
            img_base64 = get_image_base64(chart["file"])
            if img_base64:
                html_content += f"""
                <img src="data:image/png;base64,{img_base64}" alt="{chart['name']}" class="chart-image">
                """
            else:
                html_content += f"""
                <p>图片 {chart['file']} 无法加载</p>
                <img src="{chart['file']}" alt="{chart['name']}" class="chart-image">
                """
        else:
            html_content += f"""
            <p>图片 {chart['file']} 不存在</p>
            """
        
        # 添加分析
        html_content += f"""
            <h3>分析</h3>
            <div class="analysis">
                <p>{chart["description"]}</p>
            </div>
        </div>
        """
    
    # HTML文档尾部
    html_content += """
    </body>
    </html>
    """
    
    # 写入HTML文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"文档已生成: {output_path}")


def generate_markdown_documentation():
    """生成Markdown格式的文档"""
    main_py_path = os.path.join(os.path.dirname(__file__), "main.py")
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "visualization_documentation.md")
    
    # 检查main.py是否存在
    if not os.path.exists(main_py_path):
        print(f"错误: {main_py_path} 文件不存在")
        return
    
    # 图表信息
    # 获取项目根目录路径
    project_root = os.path.dirname(os.path.dirname(__file__))
    images_dir = os.path.join(project_root, "images")
    
    charts = [
        {
            "name": "蜡烛图",
            "file": os.path.join(images_dir, "candlestick_chart.png"),
            "html_file": os.path.join(images_dir, "candlestick_chart_simple.html"),
            "code_marker": "# 2. Candlestick Chart",
            "end_marker": "# 3. Volume vs. Close Price Scatter Plot",
            "description": "展示了特斯拉股票的价格走势，包括开盘价、收盘价、最高价和最低价。红色蜡烛表示上涨，绿色蜡烛表示下跌。"
        },
        {
            "name": "交易量与收盘价散点图",
            "file": os.path.join(images_dir, "volume_vs_close.png"),
            "code_marker": "# 3. Volume vs. Close Price Scatter Plot",
            "end_marker": "# 将第一列重命名为Date",
            "description": "展示了交易量与收盘价之间的关系，帮助分析交易量与价格变动的相关性。"
        },
        {
            "name": "收盘价走势图",
            "file": os.path.join(images_dir, "close_price_trend.png"),
            "code_marker": "plt.figure(figsize=(12,6))",
            "end_marker": "# 4. Moving Averages",
            "description": "展示了特斯拉股票收盘价随时间的变化，可以清晰地显示长期趋势。"
        },
        {
            "name": "移动平均线图",
            "file": os.path.join(images_dir, "moving_averages.png"),
            "code_marker": "# 4. Moving Averages",
            "end_marker": "# 5. Histogram for Price Distribution",
            "description": "展示了股票价格与50日和200日移动平均线的关系，帮助识别中长期趋势。"
        },
        {
            "name": "价格分布直方图",
            "file": os.path.join(images_dir, "price_distribution.png"),
            "code_marker": "# 5. Histogram for Price Distribution",
            "end_marker": "# 6. Box Plot for Price Volatility",
            "description": "展示了特斯拉股票收盘价的频率分布，帮助分析价格的集中趋势和分散程度。"
        },
        {
            "name": "价格波动箱线图",
            "file": os.path.join(images_dir, "price_volatility.png"),
            "code_marker": "# 6. Box Plot for Price Volatility",
            "end_marker": "# 移除plt.show()调用",
            "description": "展示了开盘价、收盘价、最高价和最低价的分布情况，帮助分析价格的波动范围和异常值。"
        },
        {
            "name": "相关性热力图",
            "file": os.path.join(images_dir, "correlation_heatmap.png"),
            "code_marker": "# 7. 相关性热力图",
            "end_marker": "# 解释相关性热力图",
            "description": "展示了各价格指标和交易量之间的相关性强度，帮助分析变量间的关系。"
        },
        {
            "name": "特斯拉股票词云图",
            "file": os.path.join(images_dir, "tesla_wordcloud.png"),
            "code_marker": "# 7. 生成词云",
            "end_marker": "# 尝试使用备用方法生成词云",
            "description": "直观地展示了与特斯拉股票相关的高频词汇，反映了投资者关注的焦点。"
        }
    ]
    
    # 提取统计特征计算代码
    stats_code = extract_code_block(main_py_path, "# 3. 统计学特征计算:", "# 4. 统计结果解释")
    
    # 如果没有找到统计特征计算代码，尝试不带冒号的标记
    if stats_code == "未找到相关代码":
        stats_code = extract_code_block(main_py_path, "# 3. 统计学特征计算", "# 4. 统计结果解释")
    
    # Markdown文档头部
    md_content = """# 数据可视化文档

本文档详细记录了特斯拉股票数据分析过程中生成的各个图表，包括预处理步骤、统计学特征计算结果和可视化分析。

## 目录

"""
    
    # 生成目录
    for i, chart in enumerate(charts, 1):
        md_content += f"{i}. [{chart['name']}](#{chart['name']})\n"
    
    # 添加统计学特征计算部分
    md_content += """
## 统计学特征计算结果

```python
{}
```

统计特征解释：
- **均值**：数据的平均值，反映数据的集中趋势
- **中位数**：将数据排序后的中间值，不受极端值影响
- **标准差**：反映数据的离散程度，值越大表示数据越分散
- **最小值/最大值**：数据的范围边界
- **分位数**：25%和75%分位数反映数据的分布特征
- **偏度**：描述分布的对称性，正值表示右偏，负值表示左偏
- **峰度**：描述分布的尖峰程度，值越大表示分布越尖锐

""".format(stats_code)
    
    # 为每个图表生成文档
    for i, chart in enumerate(charts, 1):
        code_block = extract_code_block(main_py_path, chart["code_marker"], chart["end_marker"])
        
        md_content += f"""## {chart["name"]}

### 预处理步骤和代码截图

```python
{code_block}
```

### 可视化图表

![{chart["name"]}]({chart["file"]})

### 分析

{chart["description"]}

"""
    
    # 写入Markdown文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    print(f"文档已生成: {output_path}")


if __name__ == "__main__":
    print("开始生成数据可视化文档...")
    generate_html_documentation()
    generate_markdown_documentation()
    print("文档生成完成！")