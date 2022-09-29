# Master Thesis repository
This repo contains the code used in the master thesis with the title: 
'Towards intra-operative assessment of resection margins in colorectal cancer'.
The thesis concerned intra-operative resection margin assessment with Diffuse
Reflectance Spectroscopy (DRS) and Ultrasound (US). Data is not publicly
available. The thesis text is available on: t.b.a.

All models were tracked with Weights and Biases and all results are available on: 
https://wandb.ai/spruijssers/projects. Trained models are not publicly available.

## DRS Classification
Directory contains python code for tissue type classification of fat, tumor,
fibrosis and colorectal wall with CNNs and RNNs. Scripts for multi-layer 
and single-layer tissue classification are available.

## US classification
Directory contains python code for tissue type classification of fat, tumor,
fibrosis and colorectal wall with a region of interest corresponding to DRS
measurements. Methods for tumor detection on colorectal ultrasound is also
available. 

## US segmentation
Directory contains matlab code for a custom training experiment for semantic
segmentation of colorectal ultrasound. Transfer learning and ensemble learning
was used. 

Author: C.W.A. Pruijssers
