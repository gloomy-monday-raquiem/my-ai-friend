import os
import io
import json
import boto3
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage

# S3 클라이언트 생성
s3 = boto3.client('s3')

# 모델 Id 선언
model_id = 'amazon.nova-lite-v1:0'

# Bucket 이름 선언
bucket_name = "aiassistant-bucket-0088"


def get_info(file_key):
    json_data = {}

    try:
        # S3 파일 경로 설정
        s3.head_object(Bucket=bucket_name, Key=file_key)
        print(f"{file_key} file exists in the {bucket_name}.")

        # S3에서 JSON 파일 읽기
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        json_data = json.loads(response['Body'].read())

    except s3.exceptions.ClientError as e:
        # 404 에러가 발생하면 파일이 없는 것
        if e.response['Error']['Code'] == '404':
            print(f"{file_key} file not in the {bucket_name}.")
        else:
            # 다른 에러가 발생한 경우 예외 처리
            print(f"Error: {e}")
    return json_data


def get_history(file_key):
    text_data = ""

    try:
        # S3 버킷 이름과 파일 경로 설정
        s3.head_object(Bucket=bucket_name, Key=file_key)
        print(f"{file_key} file exists in the {bucket_name}.")

        # S3에서 JSON 파일 읽기
        response_body = s3.get_object(Bucket=bucket_name, Key=file_key)
        file_content = response_body['Body'].read().decode('utf-8')

        # 파일 내용 출력
        print(file_content)

        stream = io.StringIO(file_content)
        lines = stream.readlines()

        # 마지막 5줄 출력
        num_lines = min(20, len(lines))
        for line in lines[-num_lines:]:
            text_data = text_data + line.strip() + "\n"

    except s3.exceptions.ClientError as e:
        # 404 에러가 발생하면 파일이 없는 것
        if e.response['Error']['Code'] == '404':
            print(f"{file_key} file not in the {bucket_name}..")
        else:
            # 다른 에러가 발생한 경우 예외 처리
            print(f"Error: {e}")
    return text_data


def create_prompt(info, history, query):
    ai_info = f"Name: {info.get('ai-name', '')}, Personality: {info.get('ai-character', '')}, Look: {info.get('ai-prompt', '')}"
    user_info = f"Name: {info.get('my-name', '')}, Age: {info.get('my-age', '')}, Hobby: {info.get('my-hobby', '')}, Like: {info.get('my-like', '')}, Pre-input_Prompt: {info.get('my-etc', '')}"

    prompt = f"""
    You are a kind and thoughtful AI assistant. Your info can be found in <ai> tag
    Information of the person who created you can be found in <user> atg.
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

    <history> 
    Based on the conversation history in the tag, answer the question in the <query> tag.
    Remove all tags when you answer.
    Provide the answer in a succinct manner within 2~3 sentences unless prompted differently.
    """
    return prompt


def invoke_llm(prompt):
    llm = ChatBedrock(
        model_id=model_id,
        streaming=False,
        model_kwargs={
            "max_tokens": 2024,
            "temperature": 1,
            "top_k": 250,
            "top_p": 1,
            #"stop_sequences": ["\n\nHuman"],
        }
    )

    messages = [
        HumanMessage(
            content=prompt
        )
    ]

    result = llm.invoke(messages)

    return result.content


def update_history(file_key, answer, history, query):
    # Make history
    # prev_query = query
    # prev_answer = answer
    current_history = f"""Human: {query}
    AI: {answer}
    """
    # Save history
    next_history = history + current_history
    s3.put_object(Body=next_history, Bucket=bucket_name, Key=file_key)
    print(f"JSON file successfully saved in {file_key}.")


def lambda_handler(event, context):
    id = event["queryStringParameters"]['id']
    query = event["queryStringParameters"]['query']

    if query.strip() == 'DELETE HISTORY':
        file_key = f'info/{id}_history.txt'
        s3.put_object(Body="", Bucket=bucket_name, Key=file_key)
        return {
            'statusCode': 200,
            'headers': {
                "Content-Type": "application/json; charset=UTF-8",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,X-Amz-Security-Token,Authorization,X-Api-Key,X-Requested-With,Accept,Access-Control-Allow-Methods,Access-Control-Allow-Origin,Access-Control-Allow-Headers",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "X-Requested-With": "*"
            },
            'body': 'DELETE'
        }

    # Read info file
    info = get_info(f'info/{id}_info.json')

    # Read history file
    history = get_history(f'info/{id}_history.txt')

    # Create Prompt
    prompt = create_prompt(info=info, history=history, query=query)

    # Get answer
    answer = invoke_llm(prompt=prompt)

    # Update History
    update_history(file_key=f'info/{id}_history.txt', answer=answer, history=history, query=query)

    result = {
        "answer": answer,
        "query": query,
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
