import pandas as pd

df = pd.read_csv('data/dataset_train.csv')
course_columns = df.columns[6:]
deviants = []

def identify_deviants_by_house(df, house, course):
    
    house_df = df[df['Hogwarts House'] == house]

    mean_course = house_df[course].mean()
    std_course = house_df[course].std()

    deviants_df = house_df[(house_df[course] > mean_course + 3 * std_course) | (house_df[course] < mean_course - 3 * std_course)]

    for _, row in deviants_df.iterrows():
        deviants.append({
            'Hogwarts House': house,
            'First Name': row['First Name'],
            'Last Name': row['Last Name'],
            'Course': course,
            'Score': row[course],
            'Mean': mean_course,
            'Std Dev': std_course,
            'Deviation': (row[course] - mean_course) / std_course
        })

for house in df['Hogwarts House'].unique():
    for course in course_columns:
        identify_deviants_by_house(df, house, course)

deviants_df = pd.DataFrame(deviants)
deviants_df.to_csv('deviant.csv', index=False)

deviants_df = pd.read_csv('deviant.csv')

def group_deviants_by_student(df):
    grouped_deviants = {}
    
    for _, row in df.iterrows():
        student_key = (row['Hogwarts House'], row['First Name'], row['Last Name'])
        
        if student_key in grouped_deviants:
            grouped_deviants[student_key]['Courses'].append(row['Course'])
        else:
            grouped_deviants[student_key] = {
                'Hogwarts House': row['Hogwarts House'],
                'First Name': row['First Name'],
                'Last Name': row['Last Name'],
                'Courses': [row['Course']]
            }

    grouped_data = []
    for student_key, student_info in grouped_deviants.items():
        grouped_data.append({
            'Hogwarts House': student_info['Hogwarts House'],
            'First Name': student_info['First Name'],
            'Last Name': student_info['Last Name'],
            'Courses': ', '.join(student_info['Courses'])
        })

    return pd.DataFrame(grouped_data)
grouped_deviants_df = group_deviants_by_student(deviants_df)

grouped_deviants_df.to_csv('grouped_deviant.csv', index=False)
print("Les résultats regroupés ont été enregistrés dans 'grouped_deviant.csv'.")