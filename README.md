# Fake Review Detector

This project leverages machine learning to detect fake reviews, providing a reliable tool for identifying genuine and fraudulent reviews. The Fake Review Detector analyzes various features of reviews, such as text sentiment, review length, helpful votes ratio, and more, to classify reviews as either "FAKE" or "GENUINE."

## Table of Contents
  - [Introduction](#introduction)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Model](#model)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)

## Introduction

  The Fake Review Detector project aims to enhance the credibility of online reviews by distinguishing between genuine and fake reviews. By preprocessing the review text and using a trained classifier, the project provides an effective solution to identify fraudulent reviews.

## Features

  - Data preprocessing and cleaning
  - Feature extraction (review length, word count, sentiment analysis, etc.)
  - Model training and evaluation
  - Fake review detection

## Installation

  Clone the repository and install the necessary dependencies:

  ```
  git clone https://github.com/the-developer-306/Fake-Review-Detector.git
cd Fake-Review-Detector
pip install -r requirements.txt
  ```

## Usage

  1. **Run the Application**: 
	
	
  - Use the following command to start the Flask app:

    ```
    python app.py
    ```

## Model

The project employs several machine learning techniques to classify reviews. The models explored include:

- **Logistic Regression**: A statistical method to predict the probability of a binary outcome.
- **Vectorization Pipeline**: Transforming text data into numerical format using TF-IDF.
- **Sentiment Analysis**: Using VADER sentiment analysis to extract sentiment scores.

Performance is evaluated using metrics such as accuracy, precision, and recall.

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
   
## License

This project is licensed under the MIT License.

## Contact

For any questions or suggestions, feel free to reach out:

- GitHub: [the-developer-306](https://github.com/the-developer-306)
- Email: [whilealivecode127.0.0.1@gmail.com](mailto:whilealivecode127.0.0.1@gmail.com)

