# 🍌 Stable Diffusion 1.5 COMPLETE on Banana

This repo gives a basic framework for serving Stable Diffusion 1.5 in production using simple HTTP servers.

My plan is to include stable-diffusion text2img, img2img, and inpainting in this repo so that every bananese (banana user) uses same public template for sd 1.5, which means no 10-20 second cold boot cost (For now only available txt2img with negative prompt and multiple schedulers).

# Quickstart

**[Follow the quickstart guide in Banana's documentation to use this repo](https://docs.banana.dev/banana-docs/quickstart).**

_(choose "GitHub Repository" deployment method)_

### Additional Steps

1. Create huggingface account to get permission to download and run [Stable Diffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) text-to-image model.

- Accept terms and conditions for the use of [Stable Diffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)

2. After deploying the model on Banana, be sure to set the HF_AUTH_TOKEN build argument in the settings page of the model on Banana

# Helpful Links

Understand the 🍌 [Serverless framework](https://docs.banana.dev/banana-docs/core-concepts/inference-server/serverless-framework) and functionality of each file within it.

Generalize this framework to [deploy anything on Banana](https://docs.banana.dev/banana-docs/resources/how-to-serve-anything-on-banana).

## Use Banana for scale.
