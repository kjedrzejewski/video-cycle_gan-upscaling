#!/bin/sh

aws s3 sync --exact-timestamps upscaling/example_images s3://prd-prsn-er-pet-projects/upscaling/example_outputs
aws s3 sync --exact-timestamps upscaling/losses s3://prd-prsn-er-pet-projects/upscaling/losses
aws s3 sync --exact-timestamps upscaling/trained_model s3://prd-prsn-er-pet-projects/upscaling/models
aws s3 sync --exact-timestamps images s3://prd-prsn-er-pet-projects/upscaling/images
