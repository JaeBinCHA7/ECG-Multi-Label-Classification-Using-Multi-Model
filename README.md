# ECG-Multi-Label-Classification-Using-Multi-Model

[[paper](https://paper.cricit.kr/user/listview/ieie2018/cart_rdoc.asp?URL=files/filename%3Fnum%3D431313%26db%3DRD_R&dn=431313&db=RD_R&usernum=0&seid=)]
 In this project, we will perform 12-lead ECG Multi-label Classification. Specifically, we will design a multi-model utilizing the characteristics of diagnoses from the Shaoxing and Ningbo databases.

## Update
* **2024.05.08**

* ## Requirements 
This repo is implemented in Ubuntu 22.04, PyTorch 2.3.0, Python3.11, and CUDA12.0. For package dependencies, you can install them by:

```
pip install -r requirements.txt    
```

## Dataset Installation 
In this paper, we used a large-scale ECG database consisting of 45,152 12-lead electrocardiograms [18, 19]. The database is annotated with a total of 94 labels through the Systematized Nomenclature of Medicine Clinical Terms (SNOMED CT). For ease of classification, we selected 41 SNOMED CT codes, including those for arrhythmias and ischemic cardiovascular diseases, from the 94 codes based on the hierarchical structure of SNOMED CT codes and the Minnesota Code Manual. These were then consolidated into 20 groups.  
https://physionet.org/content/ecg-arrhythmia/1.0.0/

## Getting Started
1. Install the necessary libraries.   
2. Set directory paths for your dataset. ([options.py](https://github.com/JaeBinCHA7/ECG-Multi-Label-Classification-Using-Multi-Model/blob/main/options.py))    
3. Run [train.py](https://github.com/JaeBinCHA7/ECG-Multi-Label-Classification-Using-Multi-Model/blob/main/train.py)

## Architecture [3]
<center><img src = "https://github.com/JaeBinCHA7/ECG-Multi-Label-Classification-Using-Multi-Model/assets/87358781/d7bfad44-5df6-4e32-a7e0-015f5ebd7e7c" width="100%" height="100%"></center>

## Results 
<center><img src = "https://github.com/JaeBinCHA7/ECG-Multi-Label-Classification-Using-Multi-Model/assets/87358781/b85722b5-4445-4951-a9c3-505a3602df06" width="100%" height="100%"></center>
<center><img src = "https://github.com/JaeBinCHA7/ECG-Multi-Label-Classification-Using-Multi-Model/assets/87358781/5a7a2cd2-b040-4931-bbf5-7e6cdbcbb12e" width="100%" height="100%"></center>

## References   
[1] Zheng, Jianwei, et al. "A 12-lead electrocardiogram database for arrhythmia research covering more than 10,000 patients." Scientific data 7.1 (2020): 48.  
[2] Zheng, J. "Optimal multi-stage arrhythmia classification approach Sci." Reports 101 (2020): 1-17.  
[3] Hwang, Seorim, et al. "Multi-label ECG Abnormality Classification Using A Combined ResNet-DenseNet Architecture with ResU Blocks." 2023 IEEE EMBS Special Topic Conference on Data Science and Engineering in Healthcare, Medicine and Biology. IEEE, 2023.** [[paper]](https://ieeexplore.ieee.org/abstract/document/10404234)
[[code]](https://github.com/seorim0/Multi-label-12-lead-ECG-abnormality-classification)   

## Contact  
E-mail: jbcha7@yonsei.ac.kr
