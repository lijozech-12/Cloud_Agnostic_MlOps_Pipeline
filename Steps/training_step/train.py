import argparse
import joblib
import numpy as np
from sklearn.linear_model import SGDRegressor

def train_model(x_train, y_train):
    x_train_data = np.load(x_train)
    y_train_data = np.load(y_train)

#SGD stands for Stochastic Gradient Descent: the gradient of the loss is 
# estimated each sample at a time and the model is updated along the way with a 
# decreasing strength schedule (aka learning rate).

#The regularizer is a penalty added to the loss function that shrinks model parameters 
#towards the zero vector using either the squared euclidean norm L2 or the absolute norm L1 or 
#a combination of both (Elastic Net). If the parameter update crosses the 0.0 value because of 
# the regularizer, 
#the update is truncated to 0.0 to allow for learning sparse models and achieve online feature selection.
    model = SGDRegressor(verbose=1)
    model.fit(x_train_data, y_train_data)
    
    joblib.dump(model, 'model.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_train')
    parser.add_argument('--y_train')
    args = parser.parse_args()
    train_model(args.x_train, args.y_train)