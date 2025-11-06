# External dependencies
import json
import os
import io
import base64
import boto3
#from PIL import Image
from botocore.exceptions import ClientError
import random

# S3 클라이언트 생성
s3 = boto3.client('s3')

# Image 생성을 위한 Model Id 선언

image_model_id = os.environ.get('IMAGE_MODEL_ID', 'amazon.titan-image-generator-v2:0')
# Bucket 이름 선언
bucket_name = os.environ.get('BUCKET_NAME', 'aiassistant-bucket-0088')
aws_region = os.environ.get('AWS_REGION', 'us-east-1')

# def save_image(image, path):
#     # Save
#     os.makedirs("data/titan", exist_ok=True)
#     img1 = Image.open(
#         io.BytesIO(
#             base64.decodebytes(
#                 bytes(image, "utf-8")
#             )
#         )
#     )
#     img1.save(path)


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

        boto3_bedrock = boto3.client(
            'bedrock-runtime',
            region_name=aws_region
        )

        body = {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": prompt
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "height": 1024,
                "width": 1024,
                "cfgScale": 8.0,
                "seed": seed
            }
        }

        response = boto3_bedrock.invoke_model(
            modelId=image_model_id,
            body=json.dumps(body)
        )

        response_body = json.loads(response["body"].read())
        base64_image_data = response_body["images"][0]

        return base64_image_data

    except ClientError as e:
        print(f"Couldn't invoke Titan image generator: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise


def lambda_handler(event, context):
    try:
        # 파라미터 검증
        if 'queryStringParameters' not in event or not event['queryStringParameters']:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "*",
                    "Access-Control-Allow-Headers": "*"
                },
                'body': json.dumps({'error': 'Missing required parameters'})
            }

        params = event["queryStringParameters"]
        id = params.get('id')
        prompt = params.get('prompt')

        if not id or not prompt:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    "Access-Control-Allow-Origin": "*"
                },
                'body': json.dumps({'error': 'id and prompt are required'})
            }
        print(f"ID: {id}, Prompt: {prompt}")
        # 이미지 생성
        print(f"Invoking Titan model: {image_model_id}")
        img_b64 = invoke_titan(prompt, 0)
        print("Image generated successfully")

        # S3 업로드
        filename = f'{id}_image.png'
        object_key = f'data/{filename}'

        print(f"Uploading to S3: {bucket_name}/{object_key}")
        s3.put_object(
            Bucket=bucket_name,
            Key=object_key,
            Body=io.BytesIO(base64.decodebytes(bytes(img_b64, "utf-8"))),
            ContentType='image/png'
        )
        print("Image uploaded successfully")

        file_url = f"data/{filename}"

        # JSON 파일 업데이트
        file_key = f'info/{id}_info.json'
        json_data = {}

        # Read and Update info file#####
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

        json_data['ai-prompt'] = prompt
        json_data['ai-image'] = file_url
        updated_json_data = json.dumps(json_data, ensure_ascii=False)

        print(f"Updating info file: {file_key}")
        s3.put_object(
            Body=updated_json_data,
            Bucket=bucket_name,
            Key=file_key,
            ContentType='application/json'
        )
        print("Info file updated successfully")

        result = {
            "url": file_url,
            "prompt": prompt
        }
        ####################################

        result = {
            "url": file_url,
            "prompt": prompt
        }

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
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
                'Content-Type': 'application/json',
                "Access-Control-Allow-Origin": "*"
            },
            'body': json.dumps({'error': str(e)})
        }
