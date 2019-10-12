#!/bin/sh

aws s3 sync --exact-timestamps s3://prd-prsn-er-pet-projects/upscaling/example_outputs upscaling/example_images
aws s3 sync --exact-timestamps s3://prd-prsn-er-pet-projects/upscaling/losses upscaling/losses
aws s3 sync --exact-timestamps s3://prd-prsn-er-pet-projects/upscaling/models upscaling/trained_model
aws s3 sync --exact-timestamps s3://prd-prsn-er-pet-projects/upscaling/images images
