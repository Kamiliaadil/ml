from dash import Dash, html, dcc, callback, Output, Input, State, no_update
import pandas as pd 
from dash_iconify import DashIconify
import numpy as np

from sklearn import svm
from sklearn.model_selection import train_test_split 
from features import feature_list
# print(feature_list)
import dash_mantine_components as dmc
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df = pd.read_csv("cleaned_data.csv", index_col=0)
X= df.drop(['Target'],axis=1)
y=df['Target']
svc = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo', probability=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

print(X_test.iloc[42])



X_train = sc.fit_transform(X_train.values)
X_test = sc.transform(X_test.values)
svc.fit(X_train, y_train)




# y = df['Target']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# model = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo')
# model.fit(X_train, y_train)

    
def label_value (label, value):
    return dmc.Paper(
        style = {'display':'flex'},  p = 3,
        children = [
            dmc.Text(f'Input {label}', color='gray',  w=320),
            value
        ]



    )
drop_downs = []
for key, value in feature_list.items():
    _id = key.replace("'",'')
    _id = _id.replace(" ",'_')
    _id = _id.replace("(",'_')
    _id = _id.replace(")",'_')

    if type(value) == list:
        if type(value[0]) == float:
            numveric = dmc.NumberInput(
                    # label="Number input with decimal steps",
                    id= _id,
                    value=value[0],
                    precision=2,
                    min=value[0],
                    # step=0.05,
                    max=value[1],
                    style={"width": 300},
                    
                )
            
            drop_downs.append(label_value (key, numveric))

        else:
            options = [ {'label': str(i),  'value':i} for i in range(value[0], value[1])]
            drop = dmc.Select(
                data=options,
                searchable=True,
                nothingFound="No options found",
                style={"width": 300},
                id=_id,
                value = options[0]['value']
            )
            drop_downs.append(label_value (key, drop))

    else:
        options = [{'label': key, 'value': value} for key, value in value.items()]

        drop = dmc.Select(
            data=options,
            searchable=True,
            nothingFound="No options found",
            style={"width": 300},
            value = options[0]['value'],
            id=_id
        )
        drop_downs.append(label_value (key, drop))

    # print(f"State('{_id}', 'value' ),")


app = Dash(__name__)

server = app.server

app.layout = html.Div(
children = [   
    dmc.Center(
        
        children =[
            html.Div(
                children = [
                    dmc.Text('Title here', size=22, fw=500),
                    html.Div(
                        style = {'display':'flex'},
                        children = [
                            dmc.Paper(drop_downs[:12], shadow='md', m = 10, p = 12),
                            dmc.Paper(drop_downs[12:], shadow='md', m = 10, p = 12),  
                        ]
                    ),
                    dmc.Button(
                        "Predict",   
                        id = 'predict',
                        mt = 10,
                        # mr = '20%',
                        style = {'float':'right'},
                        leftIcon=DashIconify(icon="carbon:machine-learning-model"),
                        variant="gradient",
                        gradient={"from": "teal", "to": "lime", "deg": 105},
                    ),
                    dmc.Text("Please enter the candidate's values and click 'Predict' to see whether the student is likely to graduate or drop out", color='gray'),
                    html.Div(id = 'prediction')
                ]
            )
          
        ]
    ),

    
])

@callback(
    Output("prediction", "children"), 
    
    State('Application_mode', 'value' ),
    State('Application_order', 'value' ),
    State('Course', 'value' ),
    State('Previous_qualification__grade_', 'value' ),
    State('Mothers_qualification', 'value' ),

    State('Fathers_qualification', 'value' ),
    State('Mothers_occupation', 'value' ),
    State('Fathers_occupation', 'value' ),
    State('Admission_grade', 'value' ),
    State('Displaced', 'value' ),
    State('Debtor', 'value' ),

    State('Tuition_fees', 'value' ),
    State('Gender', 'value' ),
    State('Scholarship_holder', 'value' ),
    State('Age_at_enrollment', 'value' ),
    State('Curricular_units_1st_sem__evaluations_', 'value' ),

    State('Curricular_units_1st_sem__approved_', 'value' ),
    State('Curricular_units_1st_sem__grade_', 'value' ),
    State('Curricular_units_2nd_sem__evaluations_', 'value' ),
    State('Curricular_units_2nd_sem__approved_', 'value' ),
    State('Curricular_units_2nd_sem__grade_', 'value' ),

    State('Unemployment_rate', 'value' ),
    State('Inflation_rate', 'value' ),
    State('GDP', 'value' ),

    Input("predict", "n_clicks")
)
def checkbox(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, predict):
    d = {1:'Drop Out', 0:'Graduate'}
    if not predict:
        return no_update
    ob = [[v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24]]
    ob = np.array(ob) 
    # print(ob)
    # for ix, i in enumerate(range(100)):
    #     print(ix, svc.predict([X_test[i]]))
    print(svc.predict(X_test))
    ob = sc.transform(ob)
    pr = int(svc.predict(ob)[0])
    # print('HHHHH')
    # print(type(pr))
    # print(svc.predict_proba([ob]) [:,1])
    # print(d[pr])
    return dmc.Text(d[pr], color='gray')
                


if __name__ == '__main__':
     app.run(debug=True, port = 8730)