#!/bin/bash

# http://markus.com/install-theano-on-aws/
# has Theano and CUDA pre-installed I think?
AMI="ami-b141a2f5"
TYPE="g2.2xlarge"

aws ec2 run-instances \
--region us-west-1 \
--image-id $AMI \
--count 1 \
--instance-type $TYPE \
--key-name theano \
--security-groups theano

# Then ssh tino ubuntu@