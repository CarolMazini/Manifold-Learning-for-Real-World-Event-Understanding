# Manifold Learning for Real-World Event Understanding
## Authors: Caroline Mazini Rodrigues, Aurea Soriano-Vargas, Bahram Lavi, Anderson Rocha, and Zanoni Dias


The code was developed in Python (64-bit) 3.7.4, Numpy 1.17.1, Keras 2.2.4, TensorFlow 1.13.1 and Pytorch 1.2.0.

---

The method envolves three main steps for images comparison which need to be followed sequentially:

1) extract_features;
2) train_combination;
3) rankings_comparison or train_with_manifold.



## Supporting scripts:
- reduction: for dimensionality reduction;
- projections: with the plot function including scatter plot and line plots;
- classifiers: with the implementation of the networks training methods for combination, and also the fine-tuning scripts for the extractor networks;
- rankings_utils: script for visualizing top of rankings, ranking metrics, aggregation methods and generation of ESS features (based on rankings);
- parameter_analysis: script for calculating the points of ranking recall
- utils: normalizations, augmentation of images and argument parsing.

## Output folders:
- out_files/checkpoints: files of network weights;
- out_files/features: files of extracted features (and dataset splits);
- out_files/graphs: files of generated graphs;
- out_files/rankings: files of obtained image rankings;


## Datasets folder:
- dataset: folder in which the datasets should be placed.


We will detail the content of the three main steps.
## 1) extract_features: 

The extraction of features uses three different networks
	
**Imagenet to represent objets**

```
@InProceedings{2016_szegedy, Title = {Inception-v4, {I}nception-{ResNet} and the Impact of Residual Connections on Learning}, Author = {Christian Szegedy and Sergey Ioffe and Vincent Vanhoucke and Alexander A. Alemi}, Booktitle = {31st International Conference on Artificial Intelligence (AAAI)}, Year = {2017}, Address = {San Francisco, California, USA}, Pages = {4278-4284}}

@article{2015_olga, Author = {Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei}, Title = {{ImageNet Large Scale Visual Recognition Challenge}}, Year = {2015}, journal   = {International Journal of Computer Vision},doi = {10.1007/s11263-015-0816-y}, volume={115}, number={3},pages={211-252}}
```
**Places to represent places**
```
@InProceedings{2015_simonyan, Title = {Very Deep Convolutional Networks for Large-Scale Image Recognition}, Author = {Christian Szegedy and Sergey Ioffe and Vincent Vanhoucke}, Booktitle = {3rd International Conference on Learning Representations (ICLR)}, Year = {2015}, Address = {San Diego, California, USA}, Pages = {1-14}}

@Article{2016_places,Title = {Places: An Image Database for Deep Scene Understanding}, Author = {Bolei Zhou and  Aditya Khosla and  Agata Lapedriza and Antonio Torralba and Aude Oliva}, Journal = {Journal of Vision}, Year = {2017}, pages = {1-12}, volume = 17,number = 10}
```
**PCB to represent people**
```
@InProceedings{2018_suny, Title = {Beyond Part Models: Person Retrieval with Refined Part Pooling and A Strong Convolutional Baseline}, Author = {Yifan Suny and Liang Zhengz and Yi Yangz and Qi Tianx and Shengjin Wang}, Booktitle = {15th European Conference on Computer Vision (ECCV)}, Year = {2018}, Address = {Munich, Germany}, Pages = {1-17}}

@InProceedings{2015_zheng, Title = {Scalable Person Re-identification: A Benchmark}, Author = {Liang Zheng and Liyue Shen and Lu Tian and Shengjin Wang and Jingdong Wang and Qi Tian}, Booktitle = {20th IEEE International Conference on Computer Vision (ICCV)}, Year = {2015}, Address = {Santiago, Chile}, Pages = {1116-1124}}
```

Place into *dataset* folder, a folder with the name of the event (ex: "dataset/bombing/") with two subfolders (*positive* and *negative*) containing the dataset images. It is also possible to include an unlabeled test (as the datasets museu_nacional and bangladesh_fire). For that, include also a folder named *unlabeled*.

The main script is called *main_extract_all_features.py*. It reads three csv's (*train*, *val* and *test*) from the event dataset folder (ex: "dataset/bombing/") with the path of images. The csv's should contain the 'path' and 'label' of the images. For datasets with unlabeled images, it should be provided a fourth .csv only with 'path' of images.

The extracted features will be placed on "out_files/features/*EVENT*" (ex: "out_files/features/bombing/").

**Example**

Without augmentation

```
python main_extract_all_features.py --dataset bombing --gpu_ids 0

```

With augmentation
```
python main_extract_all_features.py --dataset bombing --gpu_ids 0 --aug _aug

```


## 2) train_combination: 

It is important to have the base features extracted (step 1) for this step. We implemented networks with three different loss functions:
- Cross-Entropy;
  
 

- Contrastive loss;
  
  

-  Triplet loss.
  
 

For these three losses we provide two main training experiments:
	
1) Varying architecture of the network: 512_128, 512_128_64, 1024_512, 1024_512_128
	- Inside training_different_architectures we provided one script for each loss function in order to compare up to the four architectures. For training and extracting combined features for specific architectures (among the implemented ones) just specify the one you will be using;
	- According to the experiments, architecture *1024_512* (type 2) presented best results for small training sets;
	- Among the three loss functions the metric leaning ones presented better accuracies: *contrastive* and *triplet*.
  
  **Example**

  Without augmentation
  ```
  python train_classifier_triplet_deep.py --dataset bombing --arch 2 --gpu_ids 0

  ```

  With augmentation
  ```
  python train_classifier_triplet_deep.py --dataset bombing --arch 2 --gpu_ids 0 --aug _aug

  ```


  2) Varying the size of training set: 10,20,50,100,200 (size of positive class without augmentation)
	  - Inside training_with_different_sizes we provided one script for each loss function. The experiments were performed only with the shallower networks (*1024_512* and *512_128*) because of the good results for small training sets. The data used to these experiments was augmented.
	  - By changing the *train_numbers* you can use different set sizes (considering just original images, in other words, the ones not resultant of augmented process).
  
  **Example**

  Experiment only performed with augmentation.
  ```
  python train_classifier_triplet_vary_training.py --dataset bombing --arch 2 --gpu_ids 0

  ```

These experiments extract image features using the trained networks for combination (using the deep features extracted from the pre-trained networks). The extracted features will be placed on "out_files/features/*TYPE_REPRESENTATION*/*EVENT*" (ex: "out_files/features/triplet/bombing/").

	

## 3) rankings_comparison: 

For this step it is required to have all features we want to compare already extracted. 


Here, we can compare:
1) different combination architectures with compare_depth_query_train_mean_distance.py.
  
  **Example**

  Without augmentation
  ```
  python train_classifier_triplet_deep.py --dataset bombing --method triplet --arch0 --arch1 --arch2 --arch3

  ```

  With augmentation
  ```
  python train_classifier_triplet_deep.py --dataset bombing --aug _aug --method triplet --arch0 --arch1 --arch2 --arch3 

  ```
	

2) different training sizes for leaning combination with compare_ess_combination_vary.py. 
	
  **Example**

  Experiment performed with augmentation
  ```
  python compare_ess_combination_vary.py --dataset bombing --method triplet

  ```


3) different feature representations (including the extracted by learning combination) with compare_all_embedding_query_mean_distance.py. Here we compare six feature representations:

	- concatenated: simple concatenation of objects, places and people features (raw features obtained from *extract_features*);
	- fine-tuned: simple concantenation of objects, places and people features, but after fine-tuning the extractor networks (it is possible to fine-tune the models using the script "classifiers/finetuning_nets.py");
	- ESS: features obtained by Event Semantic Space method generating using the training images as ERIs;
    ```
    @inproceedings{rodrigues:WIFS:2019, author = {Caroline Mazini Rodrigues and Luis Pereira and Anderson Rocha and Zanoni Dias}, booktitle = {11th IEEE International Workshop on Information Forensics and Security (WIFS)}, pages = {1-6}, publisher = {IEEE}, title = {Image Semantic Representation for Event Understanding}, year = 2019, doi={10.1109/WIFS47025.2019.9035102}}
    ```
	- Cross-Entropy: using the cross-entropy loss to train the combination network for feature extraction ("/train_combination/training_different_architectures/train_classifier_cross_entropy_deep.py" or "/train_combination/training_with_different_sizes/train_classifier_cross_entropy_vary_training.py");
	- Contrastive: using the contrastive loss to train the combination network for feature extraction ("/train_combination/training_different_architectures/train_classifier_contrastive_deep.py" or "/train_combination/training_with_different_sizes/train_classifier_contrastive_vary_training.py");
	- Triplet: using the triplet loss to train the combination network for feature extraction ("/train_combination/training_different_architectures/train_classifier_triplet_deep.py" or "/train_combination/training_with_different_sizes/train_classifier_triplet_vary_training.py");
  
  **Example**
  
  Concatenated features are compared by default.
	
  Without augmentation
  ```
  python compare_all_embedding_query_train_mean_distance.py --dataset bombing --ESS --cross_entropy --contrastive --triplet

  ```

  With augmentation
  ```
  python compare_all_embedding_query_train_mean_distance.py --dataset bombing --aug _aug --ESS --cross_entropy --contrastive --triplet 

  ```

All the experiments generate the rankings placed on "out_files/rankings/*TYPE_REPRESENTATION*/*EVENT*" (ex: "out_files/features/triplet/bombing/") and a comparison graph placed on "out_files/graphs/*EVENT*" (ex: "out_files/graphs/bombing/").

The ranking file contain the positions ordered according to the images distances to the queries. These positions are related to the path position of the images in the name files. 


## 3) train_with_manifold:
For this step it is required to have all features we want to compare already extracted.

Here, we can use the features obtained from the trained combination (or other features we want), to train:
	
- an MLP ("train_with_manifold/train_mlp.py");
- an SVM ("train_with_manifold/train_svm.py").

**Example**

Multilayer perceptron classifier for features without augmentation.
```
python train_mlp.py --dataset bombing --method triplet

```

Support vector machine for features with augmentation.
```
python train_svm.py --dataset bombing --aug _aug --method triplet

```


## Citation

The presented code is the implementation of the paper entitled *Manifold Learning for Real-World Event Understanding*. If you find it useful in your research, please cite our paper:
```
@article{rodrigues:TIFS:2021, Author = {Caroline Mazini Rodrigues and Aurea Soriano-Vargas and Bahram Lavi and Anderson Rocha and Zanoni Dias}, Title = {{Manifold Learning for Real-World Event Understanding}}, Year = {2021}, journal   = {IEEE Transactions on Infomation Forensics & Security}, doi = {10.1109/TIFS.2021.3070431}}

@article{rodrigues:supplementaryTIFS:2021, Author = {Caroline Mazini Rodrigues and Aurea Soriano-Vargas and Bahram Lavi and Anderson Rocha and Zanoni Dias}, Title =  {{Manifold Learning for Real-World Event Understanding}}, Year = 2021, doi = {10.5281/zenodo.4633316}, url = {https://doi.org/10.5281/zenodo.4633316}}

```
