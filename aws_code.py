### Upload to AWS

# You'll need to set your environment with AWS variables. Add the commented lines (without the #) into a new file at ~/.aws/credentials

#  AWS_ACCESS_KEY_ID = "abc"
#  AWS_SECRET_ACCESS_KEY = "dfg"
#  AWS_REGION = "us-east-1"

import sagemaker
import pandas as pd
import boto3
import os 

s3 = boto3.client('s3')
s3.create_bucket(Bucket = "weratedogs")

all_images = []
for path in ["resized_images/validation/image_directory", "resized_images/train/image_directory", "resized_images/holdout/image_directory"]:
  all_images.append([os.path.join(path, f) for f in os.listdir(path)])

all_images = [item for s in all_images for item in s]

for file in all_images: 
  s3.upload_file(Bucket = "weratedogs", Filename = file, Key =  file.removeprefix("resized_images/"))

s3.upload_file(Bucket = "weratedogs",
              Filename = "resized_images/validation/validation_lst.lst", 
              Key = "validation_lst/validation_lst.lst")

s3.upload_file(Bucket = "weratedogs",
              Filename = "resized_images/train/train_lst.lst",
              Key = "train_lst/train_lst.lst")

"""
### Get model ready 
To create an IAM Role with the right permissions: 
- Log onto the console -> IAM -> Roles -> Create Role
- Create a service-linked role with sagemaker.amazonaws.com
- Give the role AmazonSageMakerFullAccess
- Give the role AmazonS3FullAccess 
- Name is "sagemaker_role"
From https://github.com/aws/sagemaker-python-sdk/issues/300
"""

iam = boto3.client('iam')

role_arn = iam.get_role(RoleName = "sagemaker_role")['Role']['Arn']

# commented code below doesn't work, so found it manually from https://docs.aws.amazon.com/sagemaker/latest/dg/ecr-us-east-2.html#image-classification-us-east-2.title 
# training_image = sagemaker.image_uris.retrieve(framework = "image_classification", region ="us-east-1")

training_image = "811284229777.dkr.ecr.us-east-1.amazonaws.com/image-classification:1"

sess = sagemaker.Session(boto3.session.Session(region_name = "us-east-1"))

ic = sagemaker.estimator.Estimator(
  training_image, 
  role_arn,
  instance_count = 1,
  instance_type = "ml.p2.xlarge",
  volume_size = 50,
  input_mode = "File",
  output_path = "s3://dogrates/ic-fulltraining/output",
  sagemaker_session = sess
)

len_train_set = len(os.listdir("resized_images/train/image_directory"))

ic.set_hyperparameters(
    num_layers="18",
    image_shape="3,1000,750",
    num_training_samples=len_train_set,
    num_classes="2",
    mini_batch_size="16",
    epochs="5",
    top_k="2",
    precision_dtype="float32"
)

def training_input(s3_data):
  return sagemaker.inputs.TrainingInput(
    s3_data,
    distribution="FullyReplicated",
    content_type="application/x-image",
    s3_data_type="S3Prefix"
  )

train_data = training_input("s3://weratedogs/train/")
validation_data = training_input("s3://weratedogs/validation/")
train_data_lst = training_input("s3://weratedogs/train_lst/")
validation_data_lst = training_input("s3://weratedogs/validation_lst/")

data_channels = dict(zip(["train", "validation", "train_lst", "validation_lst"],
[train_data, validation_data, train_data_lst, validation_data_lst]))

# Need to follow instructions on https://medium.com/data-science-bootcamp/amazon-sagemaker-ml-p2-xlarge-8b9cbc0dd7d to get access to machine that can be used with the image classification model. 

ic.fit(inputs = data_channels, logs = True)

### Get Predictions

# You have to do a batch transform job to figure out how your model performed in terms of actual predictions - can get accuracy from the logs but maybe just predicted everything is one class. 

transformer = ic.transformer(instance_count=1, instance_type='ml.c5.9xlarge')
transformer.transform("s3://weratedogs/validation/")

import numpy as np
import json
from urllib.parse import urlparse

s3_client = boto3.client("s3")
s3validation = "s3://weratedogs/validation/"
bucket = "weratedogs"
object_categories = ["low", "high"]
batch_job_name = CHANGE

def list_objects(s3_client, bucket, prefix):
    response = s3_client.list_objects(Bucket=bucket, Prefix=prefix)
    objects = [content["Key"] for content in response["Contents"]]
    return objects
  
def get_label(s3_client, bucket, prefix):
    filename = prefix.split("/")[-1]
    s3_client.download_file(bucket, prefix, filename)
    with open(filename) as f:
        data = json.load(f)
        index = np.argmax(data["prediction"])
        probability = data["prediction"][index]
    return object_categories[index], probability

inputs = list_objects(s3_client, bucket, urlparse(s3validation).path.lstrip("/"))
print("Sample inputs: " + str(inputs[:2]))

outputs = list_objects(s3_client, "sagemaker-us-east-1-461249631240", batch_job_name + "/image_directory")
print("Sample output: " + str(outputs[:2]))

labels = [get_label(s3_client, "sagemaker-us-east-1-461249631240", prefix) for prefix in outputs]

[x[0] for x in labels]

