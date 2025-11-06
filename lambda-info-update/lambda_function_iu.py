import os
import json
import boto3
from botocore.exceptions import ClientError

# S3 클라이언트 생성
s3 = boto3.client('s3')

# Bucket 이름 선언
bucket_name = os.environ.get('BUCKET_NAME', 'aiassistant-bucket-0088')

def update_if_exist(event, key, data):
    """파라미터가 존재하고 비어있지 않으면 데이터 업데이트"""
    if 'queryStringParameters' in event:
        query_params = event['queryStringParameters']
        if key in query_params:
            value = query_params[key]
            if value and value.strip() != "":
                print(f"{key}: {value}")
                data[key] = value.strip()
        else:
            print(f"{key} Not Found")
    else:
        print("queryStringParameters Not Found.")
    return data


def lambda_handler(event, context):
    try:
        # 파라미터 검증
        if 'queryStringParameters' not in event or not event['queryStringParameters']:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json; charset=UTF-8',
                    "Access-Control-Allow-Origin": "*"
                },
                'body': json.dumps({'error': 'Missing required parameters'}, ensure_ascii=False)
            }

        # id 가져오기
        id = event["queryStringParameters"].get('id')
        if not id:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json; charset=UTF-8',
                    "Access-Control-Allow-Origin": "*"
                },
                'body': json.dumps({'error': 'id is required'}, ensure_ascii=False)
            }

        # S3 파일 경로 설정
        file_key = f'info/{id}_info.json'
        json_data = {}

        try:
            print(f"Reading info file: {file_key}")
            response = s3.get_object(Bucket=bucket_name, Key=file_key)
            json_data = json.loads(response['Body'].read())
            print("Existing info file loaded")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                print("Info file not found, creating new one")
            else:
                print(f"Error reading info file ({error_code}): {e}")

        # JSON 데이터 업데이트
        update_if_exist(event, 'ai-name', json_data)
        update_if_exist(event, 'ai-character', json_data)
        update_if_exist(event, 'my-name', json_data)
        update_if_exist(event, 'my-age', json_data)
        update_if_exist(event, 'my-hobby', json_data)
        update_if_exist(event, 'my-like', json_data)
        update_if_exist(event, 'my-etc', json_data)

        # 업데이트된 JSON 데이터를 문자열로 변환
        updated_json_data = json.dumps(json_data, ensure_ascii=False)

        # S3에 업로드
        print(f"Updating info file: {file_key}")
        s3.put_object(
            Body=updated_json_data,
            Bucket=bucket_name,
            Key=file_key,
            ContentType='application/json'
        )
        print(f"JSON File {file_key} successfully updated.")

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json; charset=UTF-8',
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*"
            },
            'body': updated_json_data
        }

    except ClientError as e:
        print(f"S3 Error occurred: {e.response['Error']['Message']}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json; charset=UTF-8',
                "Access-Control-Allow-Origin": "*"
            },
            'body': json.dumps({'error': 'S3 error occurred'}, ensure_ascii=False)
        }
    except Exception as e:
        print(f"Unexpected Error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json; charset=UTF-8',
                "Access-Control-Allow-Origin": "*"
            },
            'body': json.dumps({'error': 'Unexpected error occurred'}, ensure_ascii=False)
        }