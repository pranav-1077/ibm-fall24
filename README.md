# ibm-fall24

# Satellite Image Captioning with Novel Text Augmentation Method using Street View Alignment

This project demonstrates a cutting-edge approach to satellite image captioning by leveraging satellite image tiles from the [LandCover.AI](https://landcover.ai/) dataset. Instead of relying on manual annotation, which is time-intensive and costly, we introduce a novel dataset creation method that utilizes street view image alignment to generate captions for training vision-language models (VLMs). 

By aligning corresponding street-level imagery with satellite image tiles, the method automates the generation of meaningful textual descriptions for satellite imagery, enabling scalable and robust training of VLMs for satellite image captioning.

## Features

- **Dataset Creation Automation**: Utilizes geospatial alignment of satellite and street view images to replace manual captioning.
- **LandCover.AI Integration**: Leverages high-quality satellite image tiles from the LandCover.AI dataset for diverse land cover types.
- **Vision-Language Model Training**: Demonstrates training of a satellite image captioning model with the automatically generated dataset.
- **End-to-End Workflow**: Includes data preprocessing, alignment pipeline, model training, and evaluation.

## Methodology

1. **Image Alignment**:
   - Extract geographic coordinates of satellite image tiles from the LandCover.AI dataset.
   - Retrieve corresponding street view imagery using APIs such as Google Street View Static API.
   - Align the images based on their geospatial metadata.

2. **Caption Generation**:
   - Use street view images and openAI 4o API to synthesize augmented captions generated by blip-rsicd 

3. **Model Training**:
   - Fine-tune satellite image captioning models using the generated dataset.
   - Evaluate model performance against the current SOTA VLM model.

4. **Scalability**:
   - Demonstrates the potential of the method to scale dataset creation for large-scale VLM satellite imagery datasets.

