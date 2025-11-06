import os
import json
import boto3
from langchain_aws import ChatBedrock
from botocore.exceptions import ClientError
#from langchain_core.messages import HumanMessage
import random

boto3_bedrock = boto3.client('bedrock-runtime')

# S3 클라이언트 생성
s3 = boto3.client('s3')

# 모델 Id 선언
model_id = os.environ.get('MODEL_ID', 'amazon.nova-lite-v1:0')


# Bucket 이름 선언
bucket_name = os.environ.get('BUCKET_NAME', 'aiassistant-bucket-0088')
aws_region = os.environ.get('AWS_REGION', 'us-east-1')

def get_llm(max_tokens=512, temperature=0.8, top_k=125, top_p=1):
    """LLM 인스턴스 생성"""
    model_kwargs = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "stop_sequences": ["\n\nHuman"],
    }

    return ChatBedrock(
        region_name=aws_region,  # 환경 변수 사용
        model_id=model_id,
        streaming=False,
        model_kwargs=model_kwargs
    )


def invoke_llm(prompt):
    """LLM 호출"""
    messages = [HumanMessage(content=prompt)]
    llm = get_llm()
    result = llm.invoke(messages)
    return result.content


def get_info(id):
    """S3에서 사용자 정보 가져오기"""
    json_data = {}
    file_key = f'info/{id}_info.json'
    
    try:
        print(f"Reading info file: {file_key}")
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        json_data = json.loads(response['Body'].read())
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            print(f"{file_key} file not found.")
        else:
            print(f"Error reading info file ({error_code}): {e}")
    
    return json_data


def lambda_handler(event, context):
    try:
        # 파라미터 검증
        if 'queryStringParameters' not in event or not event['queryStringParameters']:
            return {
                'statusCode': 400,
                'headers': {
                    "Content-Type": "application/json; charset=UTF-8",
                    "Access-Control-Allow-Origin": "*"
                },
                'body': json.dumps({'error': 'Missing required parameters'}, ensure_ascii=False)
            }

        id = event["queryStringParameters"].get('id')
        if not id:
            return {
                'statusCode': 400,
                'headers': {
                    "Content-Type": "application/json; charset=UTF-8",
                    "Access-Control-Allow-Origin": "*"
                },
                'body': json.dumps({'error': 'id is required'}, ensure_ascii=False)
            }

        # 정보 가져오기
        info = get_info(id)

        ai_info = f"Name: {info.get('ai-name', '')}, Personality: {info.get('ai-character', '')}, Look: {info.get('ai-prompt', '')}"
        user_info = f"Name: {info.get('my-name', '')}, Age: {info.get('my-age', '')}, Hobby: {info.get('my-hobby', '')}, Like: {info.get('my-like', '')}, Pre-input_Prompt: {info.get('my-etc', '')}"

        prompt = f"""
Remember that you are a helpful and thoughtful AI assistant. Show appreciation for being created to the user.

<ai-info>
{ai_info}
</ai-info> 

<user>
{user_info}
</user>

Information about the AI is in <ai-info> tag and user is in <user> tag.
Try to use the information that you know the most rather than using something you don't know.
"""

        # LLM 호출
        answer = invoke_llm(prompt)

        result = {"answer": answer}

        return {
            'statusCode': 200,
            'headers': {
                "Content-Type": "application/json; charset=UTF-8",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*"
            },
            'body': json.dumps(result, ensure_ascii=False)
        }

    except Exception as e:
        print(f"Error in lambda_handler: {e}")
        return {
            'statusCode': 500,
            'headers': {
                "Content-Type": "application/json; charset=UTF-8",
                "Access-Control-Allow-Origin": "*"
            },
            'body': json.dumps({'error': str(e)}, ensure_ascii=False)
        }