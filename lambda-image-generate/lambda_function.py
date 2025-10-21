# External dependencies
import json
import os
import io
import base64
import boto3
from PIL import Image
from botocore.exceptions import ClientError
import random

# S3 클라이언트 생성
s3 = boto3.client('s3')

# Image 생성을 위한 Model Id 선언
image_model_id = "amazon.titan-image-generator-v2:0"

# Bucket 이름 선언
bucket_name = "aiassistant-bucket-0088"


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


def invoke_titan(prompt, seed, style_preset=None):
    """
    Invokes the Amazon tita-image-generator model to create an image using
    the input provided in the request body.

    :param prompt: The prompt that you want Stable Diffusion  to use for image generation.
    :param seed: Random noise seed (omit this option or use 0 for a random seed)
    :param style_preset: Pass in a style preset to guide the image model towards
                         a particular style.
    :return: Base64-encoded inference response from the model.
    """

    try:
        # The different model providers have individual request and response formats.
        # For the format, ranges, and available style_presets of Stable Diffusion models refer to:
        # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-stability-diffusion.html

        selected_region = 'us-east-1'
        boto3_bedrock = boto3.client(
            'bedrock-runtime',
            region_name=selected_region
        )

        body = {
            "text_prompts": [{"text": prompt}],
            "seed": seed,
            "cfg_scale": 8,
            "steps": 50,
        }

        if style_preset:
            body["style_preset"] = style_preset

        response = boto3_bedrock.invoke_model(
            modelId=image_model_id, body=json.dumps(body)
        )

        response_body = json.loads(response["body"].read())
        base64_image_data = response_body["artifacts"][0]["base64"]

        return base64_image_data

    except ClientError:
        print("Couldn't invoke Titan image generator")
        raise


def lambda_handler(event, context):
    id = event["queryStringParameters"]['id']
    prompt = event["queryStringParameters"]['prompt']

    img_b64 = invoke_titan(prompt, 0)

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
