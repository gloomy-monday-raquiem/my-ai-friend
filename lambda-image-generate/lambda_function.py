# External dependencies
import json
import os
import io
import base64
import boto3
from PIL import Image
from botocore.exceptions import ClientError

# S3 클라이언트 생성
s3 = boto3.client('s3')

# Image 생성을 위한 Model Id 선언
image_model_id = os.environ.get('imageModelId')

# Bedrock 클라이언트 생성
bedrock = boto3.client('bedrock-runtime')

# Bucket 이름 선언
bucket_name = os.environ.get('assetsBucketName')


def save_image(image, path):
    # Save
    os.makedirs("data/titan", exist_ok=True)
    img1 = Image.open(
        io.BytesIO(
            base64.decodebytes(
                bytes(image, "utf-8")
            )
        )
    )
    img1.save(path)


def invoke_titan_image(prompt, seed):
    """Titan Image Generator G1 모델 호출"""

    try:
        body = {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": prompt
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "quality": "standard",
                "cfgScale": 8,
                "height": 1024,
                "width": 1024,
                "seed": seed,
            }
        }

        response = bedrock.invoke_model(
            modelId=image_model_id,
            body=json.dumps(body)
        )

        response_body = json.loads(response["body"].read())
        images = response_body.get("images", [])

        if not images:
            raise ValueError("Titan Image Generator 응답에 이미지가 없습니다.")

        base64_image_data = images[0]

        return base64_image_data

    except ClientError as error:
        print("Couldn't invoke Titan Image Generator", error)
        raise


def lambda_handler(event, context):
    id = event["queryStringParameters"]['id']
    prompt = event["queryStringParameters"]['prompt']

    img_b64 = invoke_titan_image(prompt, 0)

    # 파일이름
    filename = f'{id}_image.png'

    # 객체 키 (파일 이름)
    object_key = f'data/{filename}'

    # S3에 업로드
    s3.put_object(
        Bucket=bucket_name,
        Key=object_key,
        Body=io.BytesIO(
            base64.decodebytes(
                bytes(img_b64, "utf-8")
            )
        )
    )

    file_url = f"/default/data/{filename}"
    json_data = {}

    # Read and Update info file#####
    file_key = f'info/{id}_info.json'
    try:

        # head_object 메서드를 사용하여 파일 존재 여부 확인
        s3.head_object(Bucket=bucket_name, Key=file_key)
        print(f"{file_key} 파일이 {bucket_name} 버킷에 존재합니다.")

        # S3에서 JSON 파일 읽기
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        json_data = json.loads(response['Body'].read())

    except s3.exceptions.ClientError as e:
        # 404 에러가 발생하면 파일이 없는 것
        if e.response['Error']['Code'] == '404':
            print(f"{file_key} 파일이 {bucket_name} 버킷에 없습니다.")
        else:
            # 다른 에러가 발생한 경우 예외 처리
            print(f"에러 발생: {e}")

    # 이미지 주소, Prompt 저장
    json_data['ai-prompt'] = prompt
    json_data['ai-image'] = file_url
    updated_json_data = json.dumps(json_data, ensure_ascii=False)

    # S3에 업데이트된 JSON 파일 업로드
    s3.put_object(Body=updated_json_data, Bucket=bucket_name, Key=file_key)

    ####################################

    result = {
        "url": file_url,
        "prompt": prompt
    }

    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,X-Amz-Security-Token,Authorization,X-Api-Key,X-Requested-With,Accept,Access-Control-Allow-Methods,Access-Control-Allow-Origin,Access-Control-Allow-Headers",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "X-Requested-With": "*"
        },
        'body': json.dumps(result, ensure_ascii=False)
    }
