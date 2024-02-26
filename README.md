<b><h1><center><font size="6">Cats or Dogs - using CNN with Transfer Learning</font></center></h1></b>
## <b>1. | Introduction</b> 👋
  * Problem Overview 👨‍💻 </br>
    * 👉 <mark><b>Classifying cats and dogs images has been <mark><b>a popular task in the deep learning field</b></mark>.
    * 👉 There are <mark><b>many different "Cats and Dogs" datasets</b></mark>.
    * 👉 The objective is the same: <mark><b>train a convolutional neural network (CNN) able to successfully differentiate cats from dogs</b></mark>.
    * 👉 Therefore, <mark><b>it is a binary classification</b></mark>.
  * Dataset Description 🤔 </br>
    * 👉 The <mark><b>Dogs and Cats Dataset is taken from Kaggle 'CatsDogs' Dataset</b></mark>, <a href="https://www.kaggle.com/datasets/sanjoybijoy/catsanddogs">CatsDogs</a>.
    * 👉 This Dataset <mark><b>provides training folder and test folder with inside each folder has both dogs folder and cats folder respectively.</b></mark>.
  * Methods 🧾 </br>
    * 👉 can <mark><b>use TensorFlow to create a data generator with a validation set split and focus on training the dense layers of the model</b></mark>.
    * 👉 <a href="https://en.wikipedia.org/wiki/Transfer_learning">Transfer learning</a> <mark><b>consists of copying the weights and architecture of a network, while maintaining or retraining some of its layers for particular needs</b></mark>. It is <mark><b>recurrently used to save model building time by using weights from models already trained in other more general datasets</b></mark>.
    * 👉 In our case, <mark><b>cats and dogs are our classes</b></mark>, which are also part of the more general <a href="https://www.image-net.org/">ImageNet</a> dataset. This means that we <mark><b>can pick any CNN trained using ImageNet to get a warm start at training our own model</b></mark>.
  * Analysis Introduction 🔎 </br>
    * 👉 <a href="https://en.wikipedia.org/wiki/Residual_neural_network">ResNet-50</a> is a somewhat old, but still very popular, CNN. Its <mark><b>popularity come from the fact that it was the CNN that introduced the residual concept in deep learning</mark></b>. It <mark><b>also won the</mark></b> <a href="https://towardsdatascience.com/review-resnet-winner-of-ilsvrc-2015-image-classification-localization-detection-e39402bfa5d8">ILSVRC 2015</a> <mark><b>image classification contest</mark></b>. Since it is a well-known and very solid CNN, we decided to use it for our transfer learning task.
    * 👉 As the <mark><b>original ResNet-50 was trained on ImageNet</mark></b>, <mark><b>its last layer outputs 1000 probabilities for a tested image to belong to the 1000 different ImageNet classes</mark></b>. Therefore, we <mark><b>cannot directly use it in our binary classification problem with only cats and dogs as classes</mark></b>. Nonetheless, <mark><b>using the original weights of the network would give us a model that is too generalistic and not really built to understand cats and dogs</mark></b>.
    * 👉 We <mark><b>first transfer a base ResNet-50 CNN</mark></b>, that is, <mark><b>a ResNet-50 without its fully connected layers</mark></b>. Later, <mark><b>by freezing the base ResNet-50 weights</mark></b>, we <mark><b>add new layers and train them without changing anything in the convolutional section of the network</mark></b>.
    * 👉 In this case, <mark><b>the convolutional section becomes just an image feature extractor</mark></b> and <mark><b>the actual job of classifying the features is performed by the newly added fully connected layers</mark></b>.
    * 👉 After many experiments, an optimal architecture was found. It <mark><b>achieves 95% train accuracy and 91% validation accuracy in our cats and dogs training set</mark></b>. In this project's notebook, we show how to build and train this CNN.

## <b>2. | Accuracy of Best Model</b> 🧪
Transfer Learning ResNet50
- Training Accuracy achieved: 95.46%
- Validation Accuracy achieved: 91.71%

## <b>3. | Conclusiion </b> 📤
- In this study respectively,
- We have loaded a ResNet-50 model trained using the ImageNet dataset.
- With this transferred ResNet-50 we can perform tests using any image having 224x224 resolution.
- After loading the image that we want to use, we need to preprocess the image using the same method used in ResNet-50's training. Fortunately, TensorFlow gives us a function to do exactly so.
- Lastly, we can use the trained ResNet-50 to predict the class of the preprocessed image. Since it is trained on ImageNet, it is going to return 1000 scores in a list, one for each ImageNet class.

## <b>5. | Reference</b> 🔗
<ul><b><u>Github Notebook 📚</u></b>
        <li><a style="color: #3D5A80" href="https://github.com/guilhermedom/resnet50-transfer-learning-cats-and-dogs">resnet50-transfer-learning-cats-and-dogs by GUILHERMEDOM</a></li>
</ul>
<ul><b><u>Online Articles 🌏</u></b>
      <li><a style="color: #3D5A80" href="https://en.wikipedia.org/wiki/Transfer_learning">Transfer Learning by WIKIPEDIA</a></li>
      <li><a style="color: #3D5A80" href="https://en.wikipedia.org/wiki/Residual_neural_network">Residual neural network by WIKIPEDIA</a></li>
      <li><a style="color: #3D5A80" href="https://towardsdatascience.com/review-resnet-winner-of-ilsvrc-2015-image-classification-localization-detection-e39402bfa5d8">Review: ResNet — Winner of ILSVRC 2015 (Image Classification, Localization, Detection) by SIK-HO TSANG</a></li>
</ul>
<ul><b><u>Online Learning Channel 🌏</u></b>
        <li><a style="color: #3D5A80" href="https://www.udemy.com/course/artificial-intelligence-in-python-/learn/lecture/26598012#overview">Master Artificial Intelligence 2022 : Build 6 AI Projects by Dataisgood Academy</a></li>   
</ul>
