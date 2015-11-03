#!/bin/bash

AMI="ami-e60d6286"
TYPE="g2.2xlarge"

aws ec2 run-instances \
--region us-west-1 \
--image-id $AMI \
--count 1 \
--instance-type $TYPE \
--key-name theano \
--security-groups theano

# Then ssh tino ubuntu@