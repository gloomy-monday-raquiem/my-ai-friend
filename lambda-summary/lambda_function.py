import os
import json
import boto3
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
import random

boto3_bedrock = boto3.client('bedrock-runtime')

# S3 클라이언트 생성
s3 = boto3.client('s3')

# 모델 Id 선언
model_id = 'amazon.nova-lite-v1:0'

# Bucket 이름 선언
bucket_name = "aiassistant-bucket-0088"

def get_llm(max_tokens=512, temperature=0.8, top_k=125, top_p=1):
    model_kwargs = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "stop_sequences": ["\n\nHuman"],
    }

    # Sonnet
    selected_region = "us-east-1"
    print(f"Selected region is {selected_region}")

    return ChatBedrock(
        region_name=selected_region,
        model_id=model_id,
        streaming=False,
        # callbacks=[StreamingStdOutCallbackHandler()],
        model_kwargs=model_kwargs
    )


def invoke_llm(prompt):
    messages = [
        HumanMessage(
            content=prompt
        )
    ]
    llm = get_llm()
    result = llm.invoke(messages)

    return result.content


def get_info(id):
    json_data = {}
    # Read and Update info file#####
    file_key = f'info/{id}_info.json'
    try:

        # S3 파일 경로 설정
        s3.head_object(Bucket=bucket_name, Key=file_key)
        print(f"{file_key} file exists in {bucket_name}.")

        # S3에서 JSON 파일 읽기
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        json_data = json.loads(response['Body'].read())

    except s3.exceptions.ClientError as e:
        # 404 에러가 발생하면 파일이 없는 것
        if e.response['Error']['Code'] == '404':
            print(f"{file_key} file not in {bucket_name}.")
        else:
            # 다른 에러가 발생한 경우 예외 처리
            print(f"Error: {e}")
    return json_data


def lambda_handler(event, context):
    # file_path = f"tmp/titan/ai_info_{id}.txt"
    # info = read_ai_info(file_path=file_path)

    id = event["queryStringParameters"]['id']

    # Read and Update info file#####
    info = get_info(id)

    ai_info = f"Name: {info.get('ai-name', '')}, Personality: {info.get('ai-character', '')}, Look: {info.get('ai-prompt', '')}"
    user_info = f"Name: {info.get('my-name', '')}, Age: {info.get('my-age', '')}, Hobby: {info.get('my-hobby', '')}, Like: {info.get('my-like', '')}, Pre-input_Prompt: {info.get('my-etc', '')}"

    prompt = f"""
    Remember that you are an helpful and thoughtful AI assistant. Show appreciation of creating you to the user.


<ai-info>
{ai_info}
</ai-info> 

<user>
{user_info}
</user>

information about the ai is in <ai-info> tag and user is in <user> tag.
try to use the information that you know the most rather than using something you don't know.
"""

    # Get answer
    answer = invoke_llm(prompt)

    result = {
        "answer": answer
    }

    return {
        'statusCode': 200,
        'headers': {
            "Content-Type": "application/json; charset=UTF-8",
            "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,X-Amz-Security-Token,Authorization,X-Api-Key,X-Requested-With,Accept,Access-Control-Allow-Methods,Access-Control-Allow-Origin,Access-Control-Allow-Headers",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "X-Requested-With": "*"
        },
        'body': json.dumps(result, ensure_ascii=False)
    }
