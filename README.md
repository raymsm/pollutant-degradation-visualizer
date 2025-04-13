# Pollutant Degradation Kinetics Visualizer

A Streamlit-based application for analyzing and visualizing pollutant degradation kinetics data.

## Features

- Simple and intuitive interface for data input
- Pseudo-first-order kinetic model fitting
- Real-time visualization of experimental data and fitted curves
- Basic error analysis (R² and RMSE)
- Support for catalyst loading and light intensity parameters

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pollutant-degradation-visualizer.git
cd pollutant-degradation-visualizer
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Enter your experimental parameters:
   - Initial concentration (C₀)
   - Time-concentration data pairs
   - Catalyst loading
   - Light intensity

4. View the results:
   - Fitted curve plot
   - Rate constant (k)
   - R² value
   - RMSE
   - Half-life

## Development

The project is structured as follows:
```
pollutant-degradation-visualizer/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Dependencies
├── README.md                   # Project documentation
├── src/                        # Source code
│   ├── models/                 # Kinetic models
│   │   └── first_order.py      # First-order model implementation
└── tests/                      # Unit tests
    └── test_models.py          # Tests for kinetic models
```

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.