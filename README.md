# Potato Leaf Disease Classification

Potato plants are highly susceptible to various diseases, which can significantly impact crop yield and quality. Early detection of these diseases is essential for effective management and prevention. In this project, we developed a machine learning-based approach to classify potato leaf diseases using Convolutional Neural Networks (CNNs). We implemented two models: 
- the first model directly classifies the uploaded leaf image, while
- the second model divides the image into five regions, applies quantization, and then classifies each region separately.

Both models are integrated into a Flask web application, allowing users to upload an image and receive predictions from both models simultaneously. This project aims to compare the effectiveness of the two approaches in disease detection and provide an accessible tool for identifying potato leaf diseases.



### Flask App UI:

<img src="https://github.com/user-attachments/assets/214912a0-65e0-411f-bd54-f0e17d152255" width="700">

### Upload of image:

<img src="https://github.com/user-attachments/assets/319cd306-0044-4419-a1d1-8dfde8382986" width="700">

### Result(Output):

<img src="https://github.com/user-attachments/assets/adfe8e12-64ba-4c69-8110-0807c327b11d" width="700">
