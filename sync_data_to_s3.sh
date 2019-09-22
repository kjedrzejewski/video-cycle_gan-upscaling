#!/bin/sh

aws s3 sync upscaling/example_images s3://prd-prsn-er-pet-projects/upscaling/example_outputs
aws s3 sync upscaling/trained_model s3://prd-prsn-er-pet-projects/upscaling/models
aws s3 sync images s3://prd-prsn-er-pet-projects/upscaling/images
