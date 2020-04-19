#import necessary modules
import pandas
result = pandas.read_csv('C:\\Data\\SoumenPers\\Python_WS\\MachineLearning_Course\\trainms.csv')


#import the necessary module
from sklearn import preprocessing
# create the Labelencoder object
le = preprocessing.LabelEncoder()
#convert the categorical columns into numeric

result['Gender_m'] = le.fit_transform(result['Gender'])
result['self_employed_m'] = le.fit_transform(result['self_employed'].tolist())
result['family_history_m'] = le.fit_transform(result['family_history'].tolist())
result['work_interfere_m'] = le.fit_transform(result['work_interfere'].tolist())
result['remote_work_m'] = le.fit_transform(result['remote_work'].tolist())
result['tech_company_m'] = le.fit_transform(result['tech_company'].tolist())
result['benefits_m'] = le.fit_transform(result['benefits'].tolist())
result['care_options_m'] = le.fit_transform(result['care_options'].tolist())
result['wellness_program_m'] = le.fit_transform(result['wellness_program'].tolist())
result['seek_help_m'] = le.fit_transform(result['seek_help'].tolist())
result['anonymity_m'] = le.fit_transform(result['anonymity'].tolist())
result['leave_m'] = le.fit_transform(result['leave'].tolist())
result['mental_health_consequence_m'] = le.fit_transform(result['mental_health_consequence'].tolist())
result['phys_health_consequence_m'] = le.fit_transform(result['phys_health_consequence'].tolist())
result['coworkers_m'] = le.fit_transform(result['coworkers'].tolist())
result['supervisor_m'] = le.fit_transform(result['supervisor'].tolist())
result['mental_health_interview_m'] = le.fit_transform(result['mental_health_interview'].tolist())
result['phys_health_interview_m'] = le.fit_transform(result['phys_health_interview'].tolist())
result['mental_vs_physical_m'] = le.fit_transform(result['mental_vs_physical'].tolist())
result['obs_consequence_m'] = le.fit_transform(result['obs_consequence'].tolist())

#display the initial records
print (result[["Gender", "Gender_m"]].head(2))

#print(result)
# select columns other than 'Opportunity Number','Opportunity Result'
cols = [col for col in result.columns if col not in
['Country','state','s.no','Timestamp','treatment','no_employees','comments']]
# dropping the 'Opportunity Number'and 'Opportunity Result' columns
data = result[cols]

print(data.columns)
#assigning the Oppurtunity Result column as target
target = result['treatment']
#print(type(data))

data.head(n=2)

test = pandas.read_csv('C:\\Data\\SoumenPers\\Python_WS\\MachineLearning_Course\\testms.csv')
le = preprocessing.LabelEncoder()
#convert the categorical columns into numeric

test['Gender'] = le.fit_transform(test['Gender'])
test['self_employed'] = le.fit_transform(test['self_employed'].tolist())
test['family_history'] = le.fit_transform(test['family_history'].tolist())
test['work_interfere'] = le.fit_transform(test['work_interfere'].tolist())
test['remote_work'] = le.fit_transform(test['remote_work'].tolist())
test['tech_company'] = le.fit_transform(test['tech_company'].tolist())
test['benefits'] = le.fit_transform(test['benefits'].tolist())
test['care_options'] = le.fit_transform(test['care_options'].tolist())
test['wellness_program'] = le.fit_transform(test['wellness_program'].tolist())
test['seek_help'] = le.fit_transform(test['seek_help'].tolist())
test['anonymity'] = le.fit_transform(test['anonymity'].tolist())
test['leave'] = le.fit_transform(test['leave'].tolist())
test['mental_health_consequence'] = le.fit_transform(test['mental_health_consequence'].tolist())
test['phys_health_consequence'] = le.fit_transform(test['phys_health_consequence'].tolist())
test['coworkers'] = le.fit_transform(test['coworkers'].tolist())
test['supervisor'] = le.fit_transform(test['supervisor'].tolist())
test['mental_health_interview'] = le.fit_transform(test['mental_health_interview'].tolist())
test['phys_health_interview'] = le.fit_transform(test['phys_health_interview'].tolist())
test['mental_vs_physical'] = le.fit_transform(test['mental_vs_physical'].tolist())
test['obs_consequence'] = le.fit_transform(test['obs_consequence'].tolist())

cols = [col for col in test.columns if col not in ['Country','state','s.no','Timestamp','treatment','no_employees','comments']]
# dropping the 'Opportunity Number'and 'Opportunity Result' columns
data_test = test[cols]
#assigning the Oppurtunity Result column as target
#target_test = test['treatment']

#import the necessary module
#from sklearn.model_selection import train_test_split
#split data set into train and test sets
#data_train, data_test, target_train, target_test = train_test_split(data,target, test_size = 0.30, random_state = 10)

# import the necessary module
from sklearn.metrics import accuracy_score
""" from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
pred = gnb.fit(data_train, target_train).predict(data_test)
"""
#import necessary modules
""" from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=10)
pred = neigh.fit(data_train, target_train).predict(data_test)
"""
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=35)
pred=rfc.fit(data, target).predict(data_test)

""" from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion="entropy", max_depth=50)
pred=dtree.fit(data_train, target_train).predict(data_test)
"""
#print the accuracy score of the model
#print("Naive-Bayes accuracy : ",accuracy_score(target_test, pred, normalize = True))
print(pred)
#print("Naive-Bayes accuracy : ",accuracy_score(target_test, pred, normalize = True))
#pandas.to_csv('samplems.csv')
import numpy as np;
s =pred.shape
s_no = np.arange(1,np.squeeze(s)+1,1)
prediction = pandas.DataFrame(pred, columns=["treatment"])
prediction.insert(loc=0,column="s.no",value=s_no)
prediction.to_csv('samplems.csv',header=True,index=False, sep=',')
