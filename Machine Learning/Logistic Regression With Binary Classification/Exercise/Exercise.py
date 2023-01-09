import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('HR_comma_sep.csv')
# print(df)
# plt.scatter(df.satisfaction_level, df.left, marker='+', color='red')
# plt.show()

salary=["low","medium","high"]
left=[0,0,0]
for i in range(len(df)):
    if df.salary[i]=="low":
        left[0]+=df.left[i]
    elif df.salary[i]=="medium":
        left[1]+=df.left[i]
    else:
        left[2]+=df.left[i]
# plt.bar(salary,left)
# plt.show()

department=df.Department.unique()
left_dept=[0 for _ in range(len(department))]
for i in range(len(df)):
    for j in range(len(department)):
        if df.Department[i]==department[j]:
            left_dept[j]+=df.left[i]

# plt.bar(department,left_dept)
# plt.show()

number_of_projects=df.number_project.unique()
left_project=[0 for _ in range(len(number_of_projects))]
for i in range(len(df)):
    for j in range(len(number_of_projects)):
        if df.number_project[i]==number_of_projects[j]:
            left_project[j]+=df.left[i]

# plt.bar(number_of_projects,left_project)
# plt.show()

average_montly_hours=df.average_montly_hours.unique()
left_hours=[0 for _ in range(len(average_montly_hours))]
for i in range(len(df)):
    for j in range(len(average_montly_hours)):
        if df.average_montly_hours[i]==average_montly_hours[j]:
            left_hours[j]+=df.left[i]

# plt.bar(average_montly_hours,left_hours)
# plt.show()

time_spend_company=df.time_spend_company.unique()
left_time=[0 for _ in range(len(time_spend_company))]
for i in range(len(df)):
    for j in range(len(time_spend_company)):
        if df.time_spend_company[i]==time_spend_company[j]:
            left_time[j]+=df.left[i]

# plt.bar(time_spend_company,left_time)
# plt.show()

work_accident=df.Work_accident.unique()
left_accident=[0 for _ in range(len(work_accident))]
for i in range(len(df)):
    for j in range(len(work_accident)):
        if df.Work_accident[i]==work_accident[j]:
            left_accident[j]+=df.left[i]

# plt.bar(work_accident,left_accident)
# plt.show()

promotion_last_5years=df.promotion_last_5years.unique()
left_promotion=[0 for _ in range(len(promotion_last_5years))]
for i in range(len(df)):
    for j in range(len(promotion_last_5years)):
        if df.promotion_last_5years[i]==promotion_last_5years[j]:
            left_promotion[j]+=df.left[i]

# plt.bar(promotion_last_5years,left_promotion)
# plt.show()

subdf = df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
salary_dummies=pd.get_dummies(subdf.salary,prefix='salary')

df_with_dummies = pd.concat([subdf,salary_dummies],axis='columns')
df_with_dummies.drop(['salary'],axis='columns',inplace=True)

X=df_with_dummies
Y=df.left

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
model = LogisticRegression()
model.fit(X_train, Y_train)
print(model.predict(X_test))
print(model.predict_proba(X_test))
print(model.score(X_test, Y_test))
