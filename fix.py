import pandas as pd

def fix_graph(nodes, edges):
    new_edges = []
    is_dag = False
    is_admg = False
    for e in edges:
        start, direction, end = e
        
        if direction == "---":
            is_dag = True
            
            while True:
                print(f"Which orientation is best:\n1. {start} --> {end}\n2. {end} --> {start} ")
                response = input("Enter number of selected orientation: ")
                
                if response == "1":
                    new_edges.append((start, "-->", end))
                    break
                if response == "2":
                    new_edges.append((end, "-->", start))
                    break
                if response == "q":
                    return
                else:
                    print("Invalid Choice, select again")
            
        elif direction == "o->":
            is_admg = True
            
            while True:
                print(f"Which orientation is best for edge [{start} {direction} {end}]:\n1. {start} --> {end}\n2. {start} <-> {end}\n3. both 1 and 2\n")
                response = input("Enter number of selected orientation: ")
                
                if response == "1":
                    new_edges.append((start, "-->", end))
                    break
                if response == "2":
                    new_edges.append((start, "<->", end))
                    break
                if response == "3":
                    new_edges.append((start, "-->", end))
                    new_edges.append((start, "<->", end))
                    break
                if response == "q":
                    return
                else:
                    print("Invalid Choice, select again")
                    
        elif direction == "o-o":
            is_admg = True
            
            while True:
                print(f"Which orientation is best for edge [{start} {direction} {end}]:\n1. {start} --> {end}\n2. {end} --> {start}\n3. {start} <-> {end}\n4. both 1 and 3\n5. both 2 and 3")
                response = input("Enter number of selected orientation: ")
                
                if response == "1":
                    new_edges.append((start, "-->", end))
                    break
                if response == "2":
                    new_edges.append((end, "-->", start))
                    break
                if response == "3":
                    new_edges.append((start, "<->", end))
                    break
                if response == "4":
                    new_edges.append((start, "-->", end))
                    new_edges.append((start, "<->", end))
                    break
                if response == "5":
                    new_edges.append((end, "-->", start))
                    new_edges.append((start, "<->", end))
                    break
                if response == "q":
                    return
                else:
                    print("Invalid Choice, select again")
        else:
            new_edges.append(e)
            
    assert is_dag + is_admg < 2
        

    di_edges = [(x[0], x[2]) for x in new_edges if x[1] == "-->"]
    bi_edges = [(x[0], x[2]) for x in new_edges if x[1] == "<->"]

    if is_admg: 
        return nodes, di_edges, bi_edges
    else:
        return nodes, di_edges

def fix_DAG(nodes, edges):
    new_edges = []
    for e in edges:
        start, direction, end = e
        
        if direction == "---":
            while True:
                print(f"Which orientation is best:\n1. {start} --> {end}\n2. {end} --> {start} ")
                response = input("Enter number of selected orientation: ")
                
                if response == "1":
                    new_edges.append((start, "-->", end))
                    break
                if response == "2":
                    new_edges.append((end, "-->", start))
                    break
                else:
                    print("Invalid Choice, select again")
        else:
            new_edges.append(e)
            
    return nodes, new_edges

def fix_ADMG(nodes, edges):
    new_edges = []
    for e in edges:
        start, direction, end = e
        
        if direction == "o->":
            while True:
                print(f"Which orientation is best for edge [{start} {direction} {end}]:\n1. {start} --> {end}\n2. {start} <-> {end}\n3. both 1 and 2\n")
                response = input("Enter number of selected orientation: ")
                
                if response == "1":
                    new_edges.append((start, "-->", end))
                    break
                if response == "2":
                    new_edges.append((start, "<->", end))
                    break
                if response == "3":
                    new_edges.append((start, "-->", end))
                    new_edges.append((start, "<->", end))
                    break
                else:
                    print("Invalid Choice, select again")
                    
        elif direction == "o-o":
            while True:
                print(f"Which orientation is best for edge [{start} {direction} {end}]:\n1. {start} --> {end}\n2. {end} --> {start}\n3. {start} <-> {end}\n4. both 1 and 3\n5. both 2 and 3")
                response = input("Enter number of selected orientation: ")
                
                if response == "1":
                    new_edges.append((start, "-->", end))
                    break
                if response == "2":
                    new_edges.append((end, "-->", start))
                    break
                if response == "3":
                    new_edges.append((start, "<->", end))
                    break
                if response == "4":
                    new_edges.append((start, "-->", end))
                    new_edges.append((start, "<->", end))
                    break
                if response == "5":
                    new_edges.append((end, "-->", start))
                    new_edges.append((start, "<->", end))
                    break
                if response == "q":
                    return
                else:
                    print("Invalid Choice, select again")
        else:
            new_edges.append(e)
            
    return nodes, new_edges

def remove_strs(csvfile, outfile=None):
    df = pd.read_csv(csvfile)

    first_row_array = df.iloc[0].values
    new = df.copy()
    # print(len(df.columns))
    for i in range(len(df.columns)):

        if isinstance(first_row_array[i], str):
            new = new.drop(df.columns[i], axis=1)

    if outfile is None:
        new.to_csv(csvfile, index=False)
    else:
        new.to_csv(outfile, index=False)


def main():
    nodes = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    edges = [('Age', '---', 'Glucose'), ('Age', '-->', 'Outcome'), ('BMI', '-->', 'Outcome'), ('BloodPressure', '---', 'Age'), ('BloodPressure', '-->', 'BMI'), ('BloodPressure', '---', 'Glucose'), ('DiabetesPedigreeFunction', '-->', 'Outcome'), ('Glucose', '-->', 'Insulin'), ('Glucose', '-->', 'Outcome'), ('Insulin', '-->', 'BMI'), ('Insulin', '-->', 'DiabetesPedigreeFunction'), ('Pregnancies', '---', 'Age'), ('SkinThickness', '-->', 'BMI'), ('SkinThickness', '-->', 'Insulin')]
    x, y = fix_DAG(nodes, edges)
    print(x, y)


if __name__ == "__main__":
    main()