# Placement Prediction with Logistic Regression

A compact C++ machine learning project that predicts whether a student is likely to be placed based on academic and behavioral indicators such as DSA score, project count, IQ, and attendance.

This project demonstrates a simple binary classification model using logistic regression, feature normalization, and a training loop implemented from scratch.

## Overview

The model learns from a CSV dataset of students and predicts a binary outcome:

- 1 = placed
- 0 = not placed

The logic is implemented in two stages:

1. `placement_model.cpp` trains the model from the data
2. `prediction.cpp` uses the learned weights to predict placement for sample students

## Features Used

Each training row contains:

- `name`
- `dsa`
- `projects`
- `iq`
- `attendance`
- `placed`

The classifier uses these values to estimate the probability of placement using a sigmoid function.

## Project Structure

```text
Binary_Classifier/
├── placement_model.cpp    # Trains the logistic regression model
├── prediction.cpp         # Uses trained weights for predictions
├── students.csv           # Dataset used for training
├── placement_model.exe    # Compiled Windows executable (generated)
├── README.md              # Project documentation
```

## How the Model Works

The training pipeline does the following:

- loads student data from `students.csv`
- computes feature min/max values for normalization
- normalizes the input features to a range between 0 and 1
- applies a weighted linear combination
- passes the result through the sigmoid function
- updates weights iteratively using gradient descent

The decision rule is:

- if probability >= 0.5 → placed
- otherwise → not placed

## Data Format

The dataset is structured like this:

```csv
name,dsa,projects,iq,attendance,placed
Manish0,57,0,95,61,0
Isha1,52,10,129,41,1
Priya2,15,1,87,59,0
```

Each row corresponds to one student profile.

## Build and Run

### 1. Compile the training model

On Linux/macOS:

```bash
g++ placement_model.cpp -o placement_model
```

On Windows with MinGW or g++:

```bash
g++ placement_model.cpp -o placement_model.exe
```

### 2. Run the trainer

```bash
./placement_model
```

This trains the logistic regression model and prints the learned weights and training accuracy.

### 3. Compile the prediction script

```bash
g++ prediction.cpp -o prediction
```

### 4. Run predictions

```bash
./prediction
```

The script prints the computed probability and final placement result for sample students.

## Example Output

```text
─── Final Weights ───
w_dsa:        6.6007
w_projects:   4.9706
w_iq:         3.7706
w_attendance: 2.6825
bias:         -8.8813
```

And for prediction:

```text
Student        : Arjun
Probability      : 0.8824
Result           : PLACED ✓
```

## Notes

- The model uses internally calculated normalization stats instead of hardcoded values.
- The training code is intentionally simple and educational, making it a good starting point for learning logistic regression in C++.
- The prediction file contains fixed example values; you can replace them with your own student profiles to test the model.

## Future Improvements

Some useful enhancements would be:

- add support for reading student data from user input
- evaluate on a separate test dataset
- implement cross-validation
- add a more robust model with regularization
- convert the project into a reusable CLI tool

## License

This project is intended for learning and academic use.

## Author

Built as a beginner-friendly C++ machine learning example for placement prediction.
