# AI-Generated-Text-Catcher

This project is dedicated to helping English teachers identify essays and other homework assignments that have been generated by AI models. It provides tools and resources to analyze and detect AI-generated text using a pretrained model and tokenizer.

## Web Application

The AI-Generated-Text-Catcher web application is publicly hosted and available at:

**[https://ai-text-catcher.duckdns.org/](https://ai-text-catcher.duckdns.org/)**

If you encounter any issues or need assistance, feel free to contact the contributors of the project.

Use this web application to analyze text and identify AI-generated content quickly and efficiently.

## Project Structure

The repository contains the following files and directories:

- `model/`: Contains the model-related files, including datasets, Jupyter notebooks, and pretrained model binaries.
  - `requirements.txt`: File listing the dependencies required for the model.
  - `model.ipynb`: Jupyter notebook that explains the step-by-step creation of the model and tokenizer.
  - `process_df.ipynb`: Notebook for preprocessing datasets.
  - `data/`: Directory containing ODS files with datasets used for analysis and model training.
    - `processed.ods`: Processed dataset.
    - `writings.ods`: Original dataset of writings.
  - `models/`: Directory with binary files of the pretrained model and tokenizer vocab.
    - `pytorch_distilbert_writings.bin`: Pretrained model binary.
    - `vocab_distilbert_writings.bin`: Tokenizer vocabulary.

- `app/`: Contains the code and resources for the web application.
  - `nginx_webapp`: Configuration files for hosting the application.
  - `services/`: Backend services for handling predictions.
    - `predict_api.py`: REST API for making predictions using the pretrained model.
    - `pytorch_distilbert.bin`: Model binary used for predictions.
    - `vocab_distilbert_writings.bin`: Tokenizer vocabulary used for predictions.
    - `model_utils.py`: Utilities for loading the model and tokenizer.
    - `requirements.txt`: Dependencies required for the backend services.
  - `webapp/`: Frontend code for the web application.
    - `app.js`: JavaScript for handling frontend logic.
    - `index.html`: Main webpage for the application.
    - `results.html`: Page for displaying results of the analysis.
    - `styles.css`: Stylesheet for the web application.
  - `start_services.sh`: Script to start the web application.

## Usage

To use this project locally, follow these steps:

1. **Clone the Repository:**
   - Clone the repository to your local machine:
     ```bash
     git clone https://github.com/LuisAxel/AI-Generated-Text-Catcher
     cd AI-Generated-Text-Catcher
     ```

2. **Set Up Your Environment:**
   - Create a virtual environment (recommended) and activate it.
   - Install the required dependencies for both the model and the app:
     ```bash
     pip install -r model/requirements.txt
     pip install -r app/services/requirements.txt
     ```

3. **Explore the Model Creation Process:**
   - Open `model/model.ipynb` to explore how the model and tokenizer were created.

4. **Run the Web Application Locally:**
   - Start the web application and microservice by running the `start_services.sh` script:
     ```bash
     bash app/start_services.sh
     ```
   - This will:
     - Start the web app on `http://localhost:9000/`.
     - Set up and activate the virtual environment for the microservice.
     - Install the necessary dependencies if the virtual environment does not exist (app/services/env).
     - Start the microservice on port `8000`.
   - You can access the web application in your browser at `http://localhost:9000/`.

5. **Stop the Services:**
   - If you need to stop the services, simply terminate the script by pressing `Ctrl+C`. This will stop both the web app and the microservice.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
