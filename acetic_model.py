from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler

scaling=StandardScaler()

x = df.iloc[:,0:13]
target = df.iloc[:,15]

# Use fit and transform method 
scaling.fit(x)
Scaled_data=scaling.transform(x)

for i in range(1,14):
# Set the n_components=3
    principal=PCA(n_components=i)
    principal.fit(Scaled_data)
    x=principal.transform(Scaled_data)
    x=pd.DataFrame(x)
    X=x
    Y=target
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

    max_Acc=0
    max_ind=-1
    for i in range(1,149):
        pipe = make_pipeline(MinMaxScaler(), KNeighborsClassifier(n_neighbors=i))
        pipe.fit(x_train, y_train)  # apply scaling on training data
        score=pipe.score(x_test, y_test)
        if score>max_Acc:
            max_Acc=score
            max_ind=i
    print("The maximum accuracy is :",max_Acc,"at n-neighbours =",max_ind)
    
