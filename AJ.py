from chat_bot import *

training = pd.read_csv('Training.csv')
testing= pd.read_csv('Testing.csv')
cols= training.columns
cols= cols[:-1]
x = training[cols]
y = training['prognosis']
y1= y
reduced_data = training.groupby(training['prognosis']).max()
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx    = testing[cols]
testy    = testing['prognosis']  
testy    = le.transform(testy)
clf1  = DecisionTreeClassifier()
clf = clf1.fit(x_train,y_train)


selected_symptom = input("symptom : ")
tree_,disease_input,feature_name = tree_to_code(clf,cols,selected_symptom)

num_days = input("days : ")
present_disease,symptoms_given, symptoms_exp, num_days = get(tree_,disease_input,feature_name,int(num_days))
print('_________________')

print('_________________')

##

print(present_disease)
print('__=========== ____')
print(list(symptoms_given))
print('_________________')
##print(num_days)
print(symptoms_exp)
disease_name, description_of_disease, precutions_measures = get_D_name(present_disease, symptoms_exp,num_days )

print('_________________')
print('_________________')
print('_________________')
print(disease_name)
print(description_of_disease)
print(precutions_measures)



