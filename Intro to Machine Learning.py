import mglearn
import matplotlib.pyplot as plt


def NearestNeighborFaces():
   from sklearn.neighbors import KNeighborsClassifier
   from sklearn.model_selection import train_test_split
   from sklearn.decomposition import PCA
   import numpy as np
   from sklearn.datasets import fetch_lfw_people
   
   people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
   
   image_shape = people.images[0].shape
   
   mask = np.zeros(people.target.shape, dtype=np.bool)
   for target in np.unique(people.target):
      mask[np.where(people.target == target)[0][:50]] = 1
      
   x_people = people.data[mask]
   y_people = people.target[mask]   
   #Split the data into training and test sets 
   X_train, X_test, y_train, y_test = train_test_split(x_people, y_people, stratify=y_people, random_state=0)
   pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)
   X_train_pca = pca.transform(X_train)
   X_test_pca = pca.transform(X_test)
   
   #Build a KNeighorsClassifier using one neighbor
   knn = KNeighborsClassifier(n_neighbors=1)
   knn.fit(X_train_pca, y_train)

   print('Test set accuracy: {:.2f}'.format(knn.score(X_test_pca, y_test)))
   print('X_train_pca_.shape: {}'.format(X_train_pca.shape))
   print('pca.components_.shape: {}'.format(pca.components_.shape))
   
   fix, axes = plt.subplots(3, 5, figsize=(15, 12), subplot_kw={'xticks': (), 'yticks': ()})
   
   for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
      ax.imshow(component.reshape(image_shape), cmap='viridis')
      ax.set_title('{}. component'.format((i + 1)))
      
   plt.show()
   mglearn.plots.plot_pca_faces(X_train, X_test, image_shape)
   plt.show()
   


def UnsupervisedLearningFaces():
   
   from sklearn.datasets import fetch_lfw_people
   import numpy as np
   people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
   image_shape = people.images[0].shape
   
   fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
   
   for target, image, ax in zip(people.target, people.images, axes.ravel()):
      ax.imshow(image)
      ax.set_title(people.target_names[target])
   
   mask = np.zeros(people.target.shape, dtype=np.bool)
   for target in np.unique(people.target):
      mask[np.where(people.target == target)[0][:50]] = 1
      
   x_people = people.data[mask]
   y_people = people.target[mask]
   
   #Scale the grayscale values to be between 0 and 1 instead of 0 and 255 for better numeric stability
   x_people = x_people / 255
      
   #Count how often each target appears
   counts = np.bincount(people.target)
   #Print counts next to target names
   for i, (count, name) in enumerate(zip(counts, people.target_names)):
      print('{0:25} {1:3}'.format(name, count), end='   ')
      if (i + 1) % 3 == 0:
         print()
         
   
      
   plt.show()
   

def UnsupervisedLearning():
   from sklearn.datasets import load_breast_cancer
   #from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler
   from sklearn.decomposition import PCA
   #from sklearn.svm import SVC
   cancer = load_breast_cancer()
   
   scaler = StandardScaler()
   scaler.fit(cancer.data)
   X_scaled = scaler.transform(cancer.data)
   
   #Keep the two principal components of the data
   pca = PCA(n_components=2)
   #Fit the PCA model to breast cancer data
   pca.fit(X_scaled)
   
   #Transform data onto the first two principal components
   X_pca = pca.transform(X_scaled)
   
   print('Original shape: {}'.format(str(X_scaled.shape)))
   print('Reduced shape: {}'.format(str(X_pca.shape)))
   
   #plot first vs second principal component, colored by class
   #plt.figure(figsize=(8, 8))
   mglearn.discrete_scatter(X_pca[:, 0], X_pca[:,1], cancer.target)
   #plt.legend(cancer.target_names, loc='best')
   #plt.gca().set_aspect('equal')
   #plt.xlabel('First principal component')
   #plt.ylabel('Second principal component')
   #plt.show()
   
   plt.matshow(pca.components_, cmap='viridis')
   plt.yticks([0,1], ['First component', 'Second component'])
   plt.colorbar()
   plt.xticks(range(len(cancer.feature_names)),cancer.feature_names,rotation=60, ha='left')
   plt.xlabel('Feature')
   plt.ylabel('Principal compenents')
   plt.show()
   
  # X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)   
   
   #svm = SVC(C=100)
   #svm.fit(X_train, y_train)
   
   #print('Test set accuracy: {: .2f}'.format(svm.score(X_test,y_test)))
   
   #Preprosessing using the 0-1 scaling
   #scaler = MinMaxScaler()
   #scaler.fit(X_train)
   
   #X_train_scaled = scaler.transform(X_train)
   #X_test_scaled = scaler.transform(X_test)
   
   #Learning an SVM on the scaled training data
   #svm.fit(X_train_scaled, y_train)
   
   #Scoring on the scaled test set 
   #print('Scaled test set accuracy; {:.2F}'.format(svm.score(X_test_scaled, y_test)))
   

def UncertaintyInMultiClassClass():
   from sklearn.datasets import load_iris
   from sklearn.model_selection import train_test_split
   from sklearn.ensemble import GradientBoostingClassifier
   
   iris = load_iris()
   X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)
   
   gbrt = GradientBoostingClassifier(learning_rate=0.01, random_state=0)
   gbrt.fit(X_train, y_train)
   
   
   


def UncertaintyEstimatesFromClassifiers():
   from sklearn.ensemble import GradientBoostingClassifier
   from sklearn.datasets import make_circles
   import numpy as np
   from sklearn.model_selection import train_test_split
   
   X, y = make_circles(noise=0.25, factor=0.5, random_state=1)
   
   #We rename the class "blue" and "red" for illustration purposes
   y_named = np.array(['blue','red'])[y]
   
   #We can call train_test_split with arbitrarily many arrays;
   #all will split in a consistent manner
   X_train, X_test, y_train_named, y_test_named, y_train, y_test =  train_test_split(X, y_named, y, random_state=0)
   
   #Build the gradient boosting model
   gbrt = GradientBoostingClassifier(random_state=0)
   gbrt.fit(X_train, y_train_named)
   
   print('X_test.shape: {}'.format(X_test.shape))
   print('Decision function shape: {}'.format(gbrt.decision_function(X_test).shape))
   
   #Show the first few entries of decision_function
   print('Decision function: \n{}'.format(gbrt.decision_function(X_test[:6])))
   
   print('Threshold decision function:\n{}'.format(gbrt.decision_function(X_test)>0))
   print('Predictions:\n{}'.format(gbrt.predict(X_test)))
                       
                       


def DeepLearning():
   from sklearn.neural_network import MLPClassifier
   from sklearn.datasets import make_moons
   from sklearn.model_selection import train_test_split
   
   X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
   
   X_train, x_test, y_train, y_test = train_test_split(X,y,stratify=y, random_state=42)
   
   fig, axes = plt.subplots(2, 4, figsize=(20, 8))
   
   for axx, n_hidden_nodes in zip(axes, [100, 1000]):
      for ax, alpha, in zip(axx, [0.0001, 0.01, 0.1, 1]):
         mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes], alpha=alpha)
         mlp.fit(X_train, y_train)
         mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
         mglearn.discrete_scatter(X_train[:, 0], X_train[:,1], y_train, ax=ax)
         ax.set_title('n_hidden=[{}, {}]\naplha={:.4f}'.format(n_hidden_nodes, n_hidden_nodes, alpha))
         
   plt.show()


def SupportVectorMachine():
   from sklearn.svm import SVC
   from sklearn.model_selection import train_test_split
   
   X,y = mglearn.tools.make_handcrafted_dataset()
   #The important parameter in kernel SVMs are the regularization parameter 'C'
   svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X,y)
   mglearn.plots.plot_2d_separator(svm, X, eps=.5)
   mglearn.discrete_scatter(X[:,0], X[:, 1], y)
   
   #Plot support vectors
   sv = svm.support_vectors_
   
   #Class labels of support vectors are given by the sign of the dual coefficiants
   sv_labels = svm.dual_coef_.ravel() > 0
   mglearn.discrete_scatter(sv[:,0], sv[:,1], sv_labels, s=15, markeredgewidth=3)
   
   plt.xlabel('Feature 0')
   plt.ylabel('Feature 1')

   fig, axes = plt.subplots(3, 3, figsize=(15, 10))
   
   for ax, C in zip(axes, [-1, 0, 3]):
      for a, gamma in zip(ax, range(-1, 2)):
         mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)
         
   axes[0, 0].legend(['class 0', 'class 1', 'sv class 0', 'sv class 1'], ncol=4, loc=(.9, 1.2))
   
   plt.show()
         
   
def RAM_Prices_Over_Time():
   import os
   import pandas as pd
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.tree import DecisionTreeRegressor
   from sklearn.linear_model  import LinearRegression
   import numpy as np
   
   ram_prices = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH,'ram_price.csv'))
   
   #Use historical data to forcast prices after the year 2000
   data_train = ram_prices[ram_prices.date < 2000]
   data_test = ram_prices[ram_prices.date >= 2000]
   
   #Predict prices based on date
   X_train = data_train.date[:,np.newaxis]
   #We use a log-transform to get a simpler relationship of data to target
   y_train = np.log(data_train.price)
   
   tree = DecisionTreeRegressor().fit(X_train,y_train)
   linear_reg = LinearRegression().fit(X_train,y_train)
   
   #Predict on all data
   X_all = ram_prices.date[:, np.newaxis]
   
   pred_tree = tree.predict(X_all)
   pred_lr = linear_reg.predict(X_all)
   
   #Undo log-transform
   price_tree = np.exp(pred_tree)
   price_lr = np.exp(pred_lr)
   
   
   plt.semilogy(data_train.date, data_train.price,label='Training data')
   plt.semilogy(data_test.date,data_test.price,label='Test data')
   plt.semilogy(ram_prices.date,price_tree,label='Tree prediction')
   plt.semilogy(ram_prices.date,price_lr,label='Linear prediciton')
   plt.legend()
   plt.show()


def Plot_Feature_Importances_Cancer_Model(model, cancer):
   import numpy as np
   
   n_features = cancer.data.shape[1]
   plt.barh(range(n_features),model.feature_importances_,align='center')
   plt.yticks(np.arange(n_features),cancer.feature_names)
   plt.xlabel('Feature importance')
   plt.ylabel('Feature')
   plt.ylim(-1,n_features)
   plt.show()


def Trees():
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.datasets import load_breast_cancer
   from sklearn.model_selection import train_test_split
   from sklearn.tree import export_graphviz
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.ensemble import GradientBoostingClassifier 

   #These different methods are in order of flexibility, accuracy, and generalization.

   cancer = load_breast_cancer()
   
   X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=0)
   
   #1
   #n_jobs is how many CPU cores we want to use. 1 = one core, 2 = two, and so one. -1 = all availale cores. Using a value more than 
   #the number of physical cores available will not help. Large data sets will require a lot of CPU power and RAM using random trees.
   #The important parameters to adjust are n_estimators and max_features, and possibly max_debth. For n_estimators, larger is always better.
   #Averaging more trees will always yeild a more robust ensemble by reducing overfitting. However, more trees require more memory and time to train.
   #Changing the random_state parameter to different value, or not setting it all will produce unreproducible results. It's import to set it's value
   #if you want repordicible results. 
   #max_features determines how random each tree is. It's recommended to set this to default (not setting it all). 
   forest = RandomForestClassifier(n_estimators=200,random_state=0,n_jobs=-1)
   forest.fit(X_train,y_train)
   
   #Using random forest and ensembles of decision trees. Random forests  forces the algorithm to consifer many possible explanations,
   #the result being that the random forest captures a much broader picture of the data than a swingle tree.
   Plot_Feature_Importances_Cancer_Model(forest,cancer)
   
   print('Accuracy on training set: {:.3f}'.format(forest.score(X_train,y_train)))
   print('Accuracy on test set: {:.3f}'.format(forest.score(X_test,y_test)))      
   
   
   #2
   #The main parameters if gradient boosted tree models are the number of trees, n_estimators, and the learning_rate, which controls the the degree to which 
   #each tree is allowed ot correct the mistakes of the previous trees. These two parameters are highly interconnected. Lowering learning_rate means that
   #more trees are needed to build a model of similar complexity. In contrast to random forests, where a higher n_estimators value is always better, increasing n_estimators
   #in gradie boosting leads to a more complex model, which may lead to overfitting. A common practice is to fit n_estimators depending on time the and memory budget
   # and then search over different learning_rate.
   #AAnother importtant parameter is max_depth or aleternatively max_leaf_nodes), to reduce the complexityof each tree. Usually max_depth is set very low for gradient boosted models,
   #often not deeper than five splits.
   gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
   gbrt.fit(X_train,y_train)
   
   print('Accuracy on training set: {:.3f}'.format(gbrt.score(X_train,y_train)))
   print('Accuracy on test set: {:.3f}'.format(gbrt.score(X_test,y_test))) 
   
   Plot_Feature_Importances_Cancer_Model(gbrt,cancer)
   
   #3  
   tree = DecisionTreeClassifier(max_depth=4, random_state=0)
   tree.fit(X_train,y_train)
   #Using a single decision tree. A single tree oftern overfits and offer poor generalization performance. 
   Plot_Feature_Importances_Cancer_Model(tree,cancer)
   
   print('Accuracy on training set: {:.3f}'.format(tree.score(X_train,y_train)))
   print('Accuracy on test set: {:.3f}'.format(tree.score(X_test,y_test)))
   
   print('Features importances:\n{}'.format(tree.feature_importances_))


def main():
   from sklearn.datasets import make_blobs
   from sklearn.svm import LinearSVC
   
   X,y = make_blobs(random_state=42)
   mglearn.discrete_scatter(X[:,0],X[:,1],y)
   plt.xlabel('Feature 0')
   plt.ylabel('Feature 1')
   plt.legend(['Class 0','Class 1','Class 2'])
   
   linear_svm = LinearSVC().fit(X,y)
   
   
   
   plt.show()
    
   
if __name__ == '__main__':
   NearestNeighborFaces()