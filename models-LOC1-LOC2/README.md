# Models
Contains trained models various tasks
- classification
- decoder
- GAN
- location
- triplet-vae
- vae
- website

## Formats
### Classification
Divided into location classification and website classification
- location
- website: \
     {Locations}-{baseNetwork}-epochs{number_of_epochs}-train_samples{number of websites used for training}-triplet_samples{number of triplets generated per anchor}-{if or not domain invariant with GRL}-l{lambda_parameter}.keras

