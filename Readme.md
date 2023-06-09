# AI-Based Multiple-Choice Question Generator

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

<div>
  <img src="./images/logo.svg" alt="Logo" width="330" height="330" align="right" style="margin-left: 40px;">
  
  <p align="left">
    <em>Welcome to the AI-Based Question Generator!</em> This project is a graduation project that aims to simplify the process of generating multiple-choice questions from text using artificial intelligence. It provides a versatile platform that can benefit various user groups, including educators, students, and professionals in different domains.
  </p>
</div>

## Live Repository

Check out the live repository of the AI-Based Multiple-Choice Question Generator: [Live Repo](https://asmaa-elfatayry.github.io/Quera-MCQs-Generator/)

## Tools and Technologies Used

### Backend Tools

The AI-Based Question Generator leverages the following backend tools:

- **Python**: The project is implemented primarily in Python programming language, serving as the backbone for the entire system.

- **Flask**: Flask is used to create a lightweight and scalable web framework for building the backend server of the AI-Based Question Generator.

- **Flask API**: Flask API is utilized to create a RESTful API that exposes the question generation functionality, allowing for easy integration with other applications or systems.

- **Docker**: Docker is used for containerization, providing an efficient and consistent environment for deploying the backend server and its dependencies.

- **Similarity Library**: A similarity library, such as scikit-learn or scipy, is utilized to compute text similarity scores and identify relevant information for question generation.

- **Torch**: The Torch library is employed for its powerful deep learning capabilities, allowing for the integration of neural network models in the question generation process.

- **PKE Model**: The PKE (Python Keyphrase Extraction) model is used for keyword extraction from the provided text, aiding in generating contextually relevant questions.

- **T5 Model**: The T5 (Text-to-Text Transfer Transformer) model is leveraged to perform text generation tasks, facilitating the creation of multiple-choice questions.

- **s2Vec Model**: The s2Vec model is utilized to represent textual information in a vector space, enabling efficient similarity calculations and text analysis and used to generate distractors.

- **Transformers**: The Transformers library is employed for its extensive collection of pre-trained models, including various language models that enhance the question generation process.

- **Flashtext**: Flashtext is used for efficient and scalable keyword search and replacement in the provided text, contributing to the question generation workflow.

- **Scikit-learn**: Scikit-learn is employed for additional machine learning functionalities and algorithms, supporting various data preprocessing and analysis tasks.

### Frontend/UI Tools

The AI-Based Question Generator utilizes the following UI/frontend tools:

- **UI Design Tools**: Various UI design tools, such as Figma, Adobe Photoshop, Adobe XD and google fonts , are utilized to design and prototype the user interface of the application, ensuring a user-friendly experience.

- **HTML, CSS, and JavaScript**: These web technologies are used for frontend development to create an appealing and responsive user interface for the AI-Based Question Generator.

- **Streamlit**: Streamlit is used as a web application framework to create an intuitive and interactive user interface for the AI-Based Question Generator.

## Getting Started

To run the AI-Based Question Generator, you have two options:

### Option 1: Running the Backend (Backend-only Mode)

To run only the backend of the AI-Based Question Generator, follow these steps:

1. Download the repository or navigate to the project directory.
2. Install the required dependencies by running the following command:
   pip install -r requirements.txt
3. In the same shell, start the backend server and launch the frontend by running the following command:
   streamlit run app.py

### Option 2: Running the Full Application

To run the full AI-Based Question Generator application, including both the frontend and backend, follow these steps:

1. Clone the repository to your local machine
2. Install the required dependencies for the project by running the following command:
   pip install -r requirements.txt
3. Start the Flask backend server
4. In your web browser, open the `index.html` file
5. The application will open in your web browser, and the frontend will automatically connect to the backend server through Flask's built-in

# Authors

The AI-Based Question Generator project was developed by:

- [Amina Zahra](https://github.com/Aminazahra20/Aminazahra20)
- [Asmaa Elfatayry](https://github.com/asmaa-elfatayry)
- [Rehab Adel](https://github.com/Rehab-Adel)
- [Manar Ali](https://github.com/Manaraligomaa)
- [Naden Abozayed ](https://github.com/)

We would like to express our gratitude to all the contributors who helped in various aspects of the project.
