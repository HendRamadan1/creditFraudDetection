# Credit Card Fraud Detection

[View on GitHub](https://github.com/HendRamadan1/creditFraudDetection)

A machine learning project that detects fraudulent credit card transactions using various classification algorithms. This project includes a clean, interactive GUI built with [Streamlit](https://streamlit.io/) to make it accessible for both technical and non-technical stakeholders.

---

## Description

The project processes real-world transactional data to classify whether a given transaction is fraudulent or not. It uses supervised machine learning models, explores feature importance, and implements key techniques like feature scaling, cross-validation, and model evaluation.

A core focus is usability—via an interactive web interface—and reproducibility—through a modular code structure and accessible setup.

---

## Key Techniques Used

* **[StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)** for feature scaling to improve model convergence.
* **[Train/Test Split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)** and **[Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html)** to ensure reliable evaluation.
* **[Confusion Matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)**, ROC-AUC, and classification reports for model assessment.
* **[Streamlit Widgets](https://docs.streamlit.io/library/api-reference/widgets)** for dynamic user inputs in the GUI.
* **[Pandas Profiling](https://github.com/ydataai/pandas-profiling)** for automated exploratory data analysis.
* **[XGBoost](https://xgboost.readthedocs.io/)** and other ML algorithms like Random Forest, Logistic Regression, and SVM.
* GUI interactivity uses **[Streamlit Layout API](https://docs.streamlit.io/library/api-reference/layout)** for clear and guided user flow.
* Visualization through **[Matplotlib](https://matplotlib.org/)** and **[Seaborn](https://seaborn.pydata.org/)** for data insights.

---

## Non-Obvious Tools and Libraries

* [Streamlit](https://streamlit.io/) for creating a web UI without HTML/JS.
* [Imbalanced-learn](https://imbalanced-learn.org/stable/) for handling class imbalance using techniques like SMOTE.
* [Joblib](https://joblib.readthedocs.io/) for model persistence.
* [Pandas Profiling](https://github.com/ydataai/pandas-profiling) to generate a quick statistical overview of the dataset.

---

## Design Patterns

The project applies several object-oriented design patterns to keep the structure modular and maintainable:

* **Abstract Factory Pattern**: Used to instantiate different types of models in a unified interface.
* **Factory Pattern**: Helps encapsulate the logic of object creation for different model types.
* **Builder Pattern**: Used to configure machine learning pipelines step by step.
* **Observer Pattern**: Can be extended for tracking evaluation metrics and triggering updates in the UI.
* **Strategy Pattern**: Applied for switching between different preprocessing or modeling techniques.

These patterns help in managing complexity and enhancing flexibility when testing different models and workflows.

---

## GUI Highlights

The GUI simplifies access to the model and predictions:

* Allows CSV upload for batch predictions
* Displays prediction results instantly
* Includes model evaluation metrics
* Non-technical users can explore model outputs without writing code

It’s designed to work seamlessly with different screen sizes and encourages feedback and iteration.

To run the GUI, make sure you have all required libraries installed, then run:

```bash
streamlit run app.py
```

This will launch the Streamlit app in your browser.

---

## Project Structure

```markdown
creditFraudDetection/
│
├── app.py
├── model.py
├── utils.py
├── requirements.txt
├── README.md
├── data/
│   └── creditcard.csv
├── images/
│   └── confusion_matrix.png
├── notebooks/
│   └── EDA.ipynb
├── saved_models/
│   └── best_model.pkl
```

* `data/`: Raw dataset used for training and testing models
* `images/`: Output images like evaluation plots (e.g., confusion matrix)
* `notebooks/`: Exploratory and preprocessing work in Jupyter Notebook
* `saved_models/`: Serialized model files for deployment
* `app.py`: Streamlit GUI app
* `model.py`: Contains model training logic
* `utils.py`: Helper functions for preprocessing and metrics

---

## Contributors

* [Hend Ramadan](https://github.com/HendRamadan1)
* [Yousef Fawzi](https://github.com/Losif01)
