standard_synthetic = {
    "nodes": ['V8', 'V2', 'V4', 'Y', 'V6', 'V3', 'V5', 'V7', 'V9', 'V1'],
    "edges": [('V2', 'V8'), ('V2', 'V4'), ('Y', 'V8'), ('V4', 'Y'), ('V2', 'V3'), ('V3', 'Y'), ('V2', 'V5'), ('V3', 'V5'), ('Y', 'V7'), ('V8', 'V9'), ('V7', 'V9'), ('V6', 'Y'), ('V1', 'V3'), ('V1', 'Y'), ('V3', 'V4')]
}

diabetes = {
    "nodes": ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
    "edges": [('Age', 'BloodPressure'), ('Age', 'Glucose'), ('Age', 'Outcome'), ('Age', 'Pregnancies'), ('BMI', 'Outcome'), ('BloodPressure', 'BMI'), ('DiabetesPedigreeFunction', 'Outcome'), ('Glucose', 'BloodPressure'), ('Glucose', 'Insulin'), ('Glucose', 'Outcome'), ('Insulin', 'BMI'), ('Insulin', 'DiabetesPedigreeFunction'), ('SkinThickness', 'BMI'), ('SkinThickness', 'Insulin')]
}