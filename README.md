# Piraeus Vice

![Piraeus Vice](vice.png)

In this case, we are going to investigate what happens in the Piraeus and who is the killer in this city, exploring patterns in homicide cases.

## Running the project

To run the project, follow these steps
1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Create a virtual environment using `python3 -m venv .venv`.
4. Activate the virtual environment:
    - On Windows: `.venv\Scripts\activate`
    - On macOS/Linux: `source .venv/bin/activate`
5. Install the required packages using `python3 -m pip install -r requirements.txt`.

## Execution Workflow

Before running the core analysis, please follow these steps to ensure the data is properly set up and processed:

1. **Prepare the Data Directory:** You need to create a folder named `data/` in the root directory of the project. Place your dataset (`crimes.csv`) inside this folder, as all the `.py` files are configured to look for the data there.
2. **Data Preprocessing:** You **must** run the normalization script before executing any of the analysis scripts (Q1-Q8). This step handles the necessary data standardization and one-hot encoding.
    ```bash
    python3 normalization.py
    ```
3. **Run the Analysis:** After normalization, you can safely run the scripts for each assignment question (e.g., `Q1.py`, `Q2.py`, etc.).
4. **Generate the Submission File:** To build the final `submission.csv` file required for the assignment deliverables, run the following script:
    ```bash
    python3 construct_submission_CSV_file.py
    ```