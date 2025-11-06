import os
import io
import json
import boto3
from langchain_aws import ChatBedrock
from botocore.exceptions import ClientError
#from langchain_core.messages import HumanMessage

# S3 클라이언트 생성
s3 = boto3.client('s3')

# 모델 Id 선언
model_id = os.environ.get('MODEL_ID', 'amazon.nova-lite-v1:0')

# Bucket 이름 선언
bucket_name = os.environ.get('BUCKET_NAME', 'aiassistant-bucket-0088')
aws_region = os.environ.get('AWS_REGION', 'us-east-1')


def get_info(file_key):
    """S3에서 사용자 정보 JSON 파일 읽기"""
    json_data = {}

    try:
        print(f"Reading info file: {file_key}")

        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        json_data = json.loads(response['Body'].read())
        print(f"{file_key} file loaded successfully")

    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            print(f"{file_key} file not found in {bucket_name}")
        else:
            print(f"Error reading file ({error_code}): {e}")
    
    return json_data


def get_history(file_key):
    text_data = ""

    try:
        print(f"Reading history file: {file_key}")

        response_body = s3.get_object(Bucket=bucket_name, Key=file_key)
        file_content = response_body['Body'].read().decode('utf-8')

        print(file_content)

        stream = io.StringIO(file_content)
        lines = stream.readlines()

        # 마지막 20줄만 가져오기 (메모리 절약)
        num_lines = min(20, len(lines))
        for line in lines[-num_lines:]:
            text_data = text_data + line.strip() + "\n"

    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            print(f"{file_key} file not found in {bucket_name}")
        else:
            print(f"Error reading history ({error_code}): {e}")
    
    return text_data


def create_prompt(info, history, query):
    """AI 프롬프트 생성"""
    ai_info = f"Name: {info.get('ai-name', '')}, Personality: {info.get('ai-character', '')}, Look: {info.get('ai-prompt', '')}"
    user_info = f"Name: {info.get('my-name', '')}, Age: {info.get('my-age', '')}, Hobby: {info.get('my-hobby', '')}, Like: {info.get('my-like', '')}, Pre-input_Prompt: {info.get('my-etc', '')}"

    prompt = f"""
You are a kind and thoughtful AI assistant. Your info can be found in <ai> tag.
Information of the person who created you can be found in <user> tag.
When you answer, please reference the information about the <ai> and <user> tag. Don't need to mention about it all the time.

<ai>
{ai_info}
</ai> 

<user>
{user_info}
</user>

Current conversation:
<history>
{history}
</history>

<query>
{query}
</query>

Based on the conversation history in the <history> tag, answer the question in the <query> tag.
Remove all tags when you answer.
Provide the answer in a succinct manner within 2~3 sentences unless prompted differently.
"""
    return prompt


def invoke_llm(prompt):
    """LLM 호출"""
    try:
        llm = ChatBedrock(
            region_name=aws_region,  
            model_id=model_id,
            streaming=False,
            model_kwargs={
                "max_tokens": 2024,
                "temperature": 1,
                "top_k": 250,
                "top_p": 1,
            }
        )

        messages = [
            HumanMessage(content=prompt)  
        ]

        result = llm.invoke(messages)
        return result.content
    
    except Exception as e:
        print(f"Error invoking LLM: {e}")
        raise


def update_history(file_key, answer, history, query):
    """대화 히스토리 업데이트 및 S3 저장"""
    try:
        current_history = f"""Human: {query}
        AI: {answer}
        """
        next_history = history + current_history
        
        s3.put_object(
            Body=next_history.encode('utf-8'),  
            Bucket=bucket_name,
            Key=file_key,
            ContentType='text/plain; charset=utf-8',
            ACL='public-read'  
        )
        print(f"History successfully saved in {file_key}.")
    
    except Exception as e:
        print(f"Error updating history: {e}")
        raise

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

        params = event['queryStringParameters']
        id = params.get('id')
        query = params.get('query')

        if not id or not query:
            return {
                'statusCode': 400,
                'headers': {
                    "Content-Type": "application/json; charset=UTF-8",
                    "Access-Control-Allow-Origin": "*"
                },
                'body': json.dumps({'error': 'id and query are required'}, ensure_ascii=False)
            }

        # 히스토리 삭제 명령
        if query.strip() == 'DELETE HISTORY':
            file_key = f'info/{id}_history.txt'
            s3.put_object(
                Body=b'',  # ✅ bytes로 변경
                Bucket=bucket_name,
                Key=file_key,
                ContentType='text/plain; charset=utf-8',
                ACL='public-read'
            )
            return {
                'statusCode': 200,
                'headers': {
                    "Content-Type": "application/json; charset=UTF-8",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "*",
                    "Access-Control-Allow-Headers": "*"
                },
                'body': json.dumps({'message': 'History deleted'}, ensure_ascii=False)
            }

        # 정보 및 히스토리 읽기
        info = get_info(f'info/{id}_info.json')
        history = get_history(f'info/{id}_history.txt')

        # 프롬프트 생성 및 LLM 호출
        prompt = create_prompt(info=info, history=history, query=query)
        answer = invoke_llm(prompt=prompt)

        # 히스토리 업데이트
        update_history(
            file_key=f'info/{id}_history.txt',
            answer=answer,
            history=history,
            query=query
        )

        result = {
            "answer": answer,
            "query": query,
        }

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