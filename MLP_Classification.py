import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore")

#""""""""""""""""""""""""""""Data loading"""""""""""""""""""""""""""""""""""

X_raw = pd.read_csv("24microarraydata.csv")                             #우리 원래 데이터
X_raw_values = X_raw.values
scaler = StandardScaler()
X_raw_scaled = scaler.fit_transform(X_raw_values)                       #Normalization은 제가 한 버전을 사용하는 걸로 합시다.
X_raw_scaled_Transpose = X_raw_scaled.T
y_raw = np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1])     #1-12번은 정상, 13-24번은 환자

#"""""""""""Train, Cross-validation, Test set 분할 """""""""""""""""""""""

X_, X_test, y_, y_test = train_test_split(X_raw_scaled_Transpose, y_raw, test_size = 0.25, random_state = 31415926)
X_train, X_val, y_train, y_val = train_test_split(X_, y_, test_size = 0.22, random_state = 5)
print (X_train.shape, X_val.shape, X_test.shape)               # Train / C_val / Test set 분할 : 14, 4, 6개
print (y_train, y_val, y_test)

#"""""""""""""""""""Training set oversampling하기 (a.k.a. 뻥튀기)"""""""""""""""""""""""""""""""""
# X_train은 지금 14 x 32321 모양의 행렬입니다.
# X_train을 이용해서 X_train_oversample 이라는 140 x 32321 짜리 행렬로 만들건데, 다음과 같은 방식으로 합니다.
# 1) X_train_oversample의 처음 14행은 그대로 X_train을 씁니다.
# 2) X_train_oversample의 다음 14행은 X_train의 모든 항에 평균이 0이고 표준편차가 0.2인 정규분포를 따르는 난수를 더한 행렬로 채웁니다.
# 3) 다음 14행도 마찬가지로 채웁니다.
# 4) 총 9번 반복하게 되면, X_train_oversample이라는 140개의 행을 가진 행렬은
#    첫 14행은 X_train 원본, 나머지 126행은 X_train + 난수 가 더해진 14x32321짜리 행렬 총 9개가 채우게 됩니다.
# 5) y_train_oversample은 길이가 140짜리인 numpy array로, y_train 10개를 갖다 붙인 행렬입니다.

np.random.seed(0)
num_train = X_train.shape[0]                                                                # =14
X_train_oversample = np.zeros((num_train*10, X_train.shape[1]))

# 여기부터 여러분들이 채우면 됩니다.

#1) 첫 14행 채우기
X_train_oversample[0:num_train, ] = X_train

#2-4) X_train_oversample 만들기
# Hint : np.random.normal을 이용할 것
for i in range(1,10):
    store = []
    for j in range(num_train):
        store.append(X_train[j] + np.random.normal(loc=0, scale=0.2, size = X_train.shape[1]))
    X_train_oversample[num_train*i:num_train*(i+1), ] = store

#5) y_train_oversample 만들기
y_train_oversample = np.tile(y_train, 10)


# 여기를 넘어서 Training set oversampling 코드를 작성하지 마세요.

#"""""""""""""""""""모델 디자인"""""""""""""""""""""""""""""""""""""""""""""""""
# 여기부터 여러분들이 채우면 됩니다.
# scikit-learn을 이용해서 여러분들이 설계한 모델을 design하세요.
# Hint : https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html 참고.
# 모델 디자인에서 여러분들이 채워야 할 코드는 정확히 한 줄입니다.
# 반드시 solver = 'adam', warm_start = True, max_iter = 1 으로 setting하세요.

model = MLPClassifier(hidden_layer_sizes=(40,30,20,10), activation='tanh', solver='adam', warm_start=True, max_iter=1, learning_rate_init=3e-4)

# 여기를 넘어서 모델 디자인 코드를 작성하지 마세요.
# 여기까지 잘 작성했다면 이후 코드는 건드리지 않아도 코드가 돌아갑니다.

#"""""""""""""""""""모델 training 및 evaluation"""""""""""""""""""""""""""""""


def training_model(model, max_iters = 1000):
    is_overfit = False
    best_c_val_loss = 2020                                                                              # Random big number
    train_losses = []                                                                                   # Training_loss가 얼마인지 기록
    c_val_losses = []                                                                                   # C-val loss가 얼마인지 기록
    for i in range(max_iters):
        if i == 1 : tic = time.time()
        if i == 10 :
            toc = time.time()
            print ("Expected remaining time :" + str((round((toc-tic)*max_iters/10.0,2))) + "seconds") # TIME IS GOLD

        model.fit(X_train_oversample, y_train_oversample)                                              # Forward propagation 1번, Backward propagation 1번
        train_loss_ = log_loss(y_train_oversample, model.predict_proba(X_train_oversample))            # Train loss 얼마?
        c_val_loss = log_loss(y_val, model.predict_proba(X_val))                                       # C-val loss 얼마?
        train_losses.append(train_loss_)                                                               # 기록
        c_val_losses.append(c_val_loss)                                                                # 기록
        print ("Iteration #" + str(i) + " Training loss : "+ str(round(train_loss_,8))                 # 얼마인지 한 번 training 될 때마다 확인.
               +", Validation loss : " + str(round(c_val_loss,8)))

        if c_val_loss < best_c_val_loss :                                                              # Cross validation loss가 최소인 모델이 제일 성능이 좋을 것이라고 가정하고, 그걸 사용합시다.
            model_to_keep = model
            is_overfit = True
            best_c_val_loss = c_val_loss

    if not is_overfit :
        print ("Underfit")
        model_to_keep = model

    return model_to_keep, train_losses, c_val_losses, i


def plot_learning_curve(Loss1, Loss2, num) :                                                           # Plot Learning curve
    x = list(range(1, num+2))
    plt.plot(x, Loss1, label="Train loss")
    plt.plot(x, Loss2, label="C-val loss")
    plt.xlabel("#Iters")
    plt.ylabel("Losses")
    plt.legend()
    plt.show(10)


def plot_ROC_curve(X, y_true, model):                                                                 # Plot ROC curve
    y_preds = model.predict_proba(X)
    y_pred = y_preds[:, 1]
    fpr, tpr, threshold = roc_curve(y_true, y_pred)
    AUC = roc_auc_score(y_true, y_pred)
    plt.plot(fpr, tpr, label = 'ROC curve (area = %0.2f)' % AUC)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show(10)
    print ("AUC on the test set = ", str(AUC))


model, train_losses, c_val_losses, num_iters = training_model(model, max_iters = 1000)
plot_learning_curve (train_losses, c_val_losses, num_iters)
acc = model.score(X_test, y_test)
print ("Accuracy on test set = " + str(acc))
plot_ROC_curve(X_test, y_test, model)

