# phd_fas_uncertainty
___
Repository of codes and models used to develop a facial anti-spoofing architecture with uncertainty treatment.

This repository aims to present an end-to-end system for facial anti-spoofing (FAS). The system is made up of 3 stages. The first corresponds to a facial detector, which is used to find the region of the person's face. The second stage consists of a face image quality classifier, making it possible to reject the image if it is not within the acceptable parameters. Finally, the stage that performs the classification if the face image corresponds to a real face or a spoof is executed. The FAS model was developed to perform uncertainty treatment, configuring to be more restrictive than permissive in presentation attacks.

The repository is divided into two sections. The first section corresponds to all the experimentation files that led to the selection of the models to form the end-to-end system executed in the second section.

All the codes of the experiments carried out to select the best-analyzed models can be seen in the ***experiments*** folder. The codes for each element of the proposed end-to-end system are found in notebook format.

On the other hand, the second section corresponds to the codes of the scripts that make up the library (***fasuncertainty***) created to implement the end-to-end system. For more information, see (or run) the ***"run_example.ipynb"*** notebook.
