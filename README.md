![LOGO][Logo]
# LIWI
Language Independent Writer Identification

Automatic offline text-independent writer identification is very important for forensic
analysis, documents authorization, and calligraphic relic’s identification, etc. The offline textindependent
writer identification is to determine the writer of a text among a number of known
writers using their handwriting images. Extensive researches have been conducted in this field. In
addition, a series of international writer identification contests have been successfully organized.
In general, the existing approaches of offline text-independent writer identification can be
roughly divided into two categories: texture-based approaches and structure-based approaches.
In this project we considered three different approaches for identifying writers, a texturebased
approach, and two structure-based approach, the main aim was to compare the results and
come up with the best approach for a writer identification application, we then implemented a user
friendly interface to make it easier for the users to try out our model.

This model can be tested by using [LIWI Desktop](https://github.com/Shaalan31/LIWIDesktop)



## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

run the following command to install all the dependencies
```
pip install -r requirements.txt
```
## Authors

* **Omar Shaalan**  - [Shaalan31](https://github.com/Shaalan31)

See also the list of [contributors](https://github.com/Shaalan31/LIWI/contributors) who participated in this project.

## References

[1] U.V. Marti, R. Messerli and H. Bunke. Writer identification using text line based features.
Proceedings of 6th ICDAR. pp. 101–105. 2001. 680, 684, 685

[2] Hertel, Caroline & Bunke, Horst. (2003). A Set of Novel Features for Writer Identification.
2688. 679-687. 10.1007/3-540-44887-X_79

[3] Singh, Priyanka & Roy, Partha & Raman, Balasubramanian. (2018). Writer identification using
texture features: A comparative study. Computers & Electrical Engineering. 71. 1-12.
10.1016/j.compeleceng.2018.07.003.

[4] Wu, Xiangqian & Tang, Yuan & Bu, Wei. (2014). Offline Text-Independent Writer
Identification Based on Scale Invariant Feature Transform. Information Forensics and Security,
IEEE Transactions on. 9. 526-536. 10.1109/TIFS.2014.2301274.


[block]: https://github.com/Shaalan31/LiwiD/blob/master/readme_images/block.png "Block Diagram"


[Horst]: https://github.com/Shaalan31/LiwiD/blob/master/readme_images/horst.png "Horst Model"

[Horst-3NN]: https://github.com/Shaalan31/LiwiD/blob/master/readme_images/horst-3NN.png "Horest Accuracy 3NN"

[Horst-3NN]: https://github.com/Shaalan31/LiwiD/blob/master/readme_images/horst-5NN.png "Horest Accuracy 5NN"

[Horst-NN]: https://github.com/Shaalan31/LiwiD/blob/master/readme_images/horst-NN.png "Horest Accuracy NN"

[logo]: https://github.com/Shaalan31/LiwiD/blob/master/readme_images/logo.png "LI Writer Indeitification"

[PreProcessing]: https://github.com/Shaalan31/LiwiD/blob/master/readme_images/preprocessing.png "Preprocessing Block Diagram"

[SIFT]: https://github.com/Shaalan31/LiwiD/blob/master/readme_images/sift.png "SIFT Model"

[AR_Centers]: https://github.com/Shaalan31/LiwiD/blob/master/readme_images/sift_ar_centers.png "CodeBook Centers Arabic"

[Sift-Acc]: https://github.com/Shaalan31/LiwiD/blob/master/readme_images/sift_en.png "Sift Accuracy"

[EN_Centers]: https://github.com/Shaalan31/LiwiD/blob/master/readme_images/sift_en_centers.png "CodeBook Centers English"

[SIFT_Val_T1]: https://github.com/Shaalan31/LiwiD/blob/master/readme_images/sift_val_T1.png "Validation on Sift Hyperparameters with T=1"

[SIFT_Val_T50]: https://github.com/Shaalan31/LiwiD/blob/master/readme_images/sift_val_T50.png "Validation on Sift Hyperparameters with T=50"

[SIFT_Val_T150]: https://github.com/Shaalan31/LiwiD/blob/master/readme_images/sift_val_T150.png "Validation on Sift Hyperparameters with T=150"

[SIFT_Val_T225]: https://github.com/Shaalan31/LiwiD/blob/master/readme_images/sift_val_T225.png "Validation on Sift Hyperparameters with T=225"

[Texture]: https://github.com/Shaalan31/LiwiD/blob/master/readme_images/texture.png "Texture Model"

[Texture_Results]: https://github.com/Shaalan31/LiwiD/blob/master/readme_images/texture_en.PNG "Texture Model Results"

[Texture_Val_H]: https://github.com/Shaalan31/LiwiD/blob/master/readme_images/texture_Validation-H.png "View 1"

[Texture_Val_H2]: https://github.com/Shaalan31/LiwiD/blob/master/readme_images/texture_Validation-H2.png "View 2"

