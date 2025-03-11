import pandas as pd
import time
import boto3
from botocore.exceptions import ClientError

# 创建 Bedrock Runtime 客户端
client = boto3.client("bedrock-runtime", region_name="us-west-2")
model_id = "us.deepseek.r1-v1:0"

def get_response(prompt):
    """
    使用 AWS Bedrock 从 Deepseek 模型获取响应
    """
    try:
        # 构建消息，将system提示改为user提示
        messages = [
            {
                "role": "user",
                "content": [{"text": "直接返回正确答案选项，不要解释。\n\n" + prompt}]
            }
        ]
        
        # 调用模型
        response = client.converse(
            modelId=model_id,
            messages=messages,
            inferenceConfig={"maxTokens": 2000, "temperature": 0.6, "topP": 0.9},
        )
        
        # 提取响应文本
        response_text = response["output"]["message"]["content"][0]["text"]
        print(f"响应: {response_text}")
        
        # 添加延迟以避免速率限制
        time.sleep(2)
        
        return response_text
    
    except (ClientError, Exception) as e:
        print(f"错误: 无法调用 '{model_id}'。原因: {e}")
        return None

def evaluate_model():
    """
    在数据集上评估模型性能
    """
    try:
        # 加载数据集
        df = pd.read_json("data/data.json", lines=True)
        
        # 对数据集中的每个提示应用模型
        df['answer'] = df['prompt'].apply(get_response)
        
        # 计算准确率
        accuracy = sum(df['answer'] == df['referenceResponse']) / len(df)
        print(f"准确率: {accuracy:.4f}")
        
        # 保存结果
        df.to_csv("data/deepseek.csv", index=False)
        
        return accuracy, df
    
    except Exception as e:
        print(f"评估过程中出错: {e}")
        return None, None

if __name__ == "__main__":
    accuracy, results = evaluate_model()