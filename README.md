# AI-Generated-Text-Catcher

This project is dedicated to helping English teachers identify essays and other homework assignments that have been generated by AI models. It provides tools and resources to analyze and detect AI-generated text using a pretrained model and tokenizer.

## Project Structure

The repository contains the following files and directories:

- `data/`: Directory containing ODS files with writings. These are datasets used for analysis and model training.
- `models/`: Directory with binary files for the pretrained model and tokenizer vocab, created from `model.ipynb`.
- `demo.ipynb`: Jupyter notebook that demonstrates how to use the pretrained model and tokenizer for detecting AI-generated text.
- `model.ipynb`: Jupyter notebook that explains the step-by-step creation of the model and tokenizer.
- `requirements.txt`: File listing the dependencies required to set up the virtual environment.

## Usage

To use this project, follow these steps:

1. **Clone the Repository:**
   - First, clone the repository to your local machine:
     ```bash
     git clone https://github.com/LuisAxel/AI-Generated-Text-Catcher
     cd AI-Generated-Text-Catcher
     ```

2. **Set Up Your Environment:**
   - Create a virtual environment (recommended) and activate it.
   - Install the required dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. **Explore the Model Creation Process:**
   - Open `model.ipynb` to understand how the model and tokenizer were created.

4. **Run the Demo:**
   - Open `demo.ipynb` to see the pretrained model in action. This notebook demonstrates how to load the model and tokenizer from the `models/` directory and use them for text detection.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
