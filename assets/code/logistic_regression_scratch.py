from utils import *

data = pd.read_csv('game_prediction.csv')
x = data[['gold_difference(K)']]
y = data[['win_or_lose']]

x = np.r_[x]
y = np.r_[y]
N=len(x)

plt.scatter(x,y, c='navy')
plt.xlabel('global gold difference (K)')
plt.ylabel('lose(0) | win(1)')
plt.title('Predicting Win/Loss Based on Global Gold Difference')
plt.show()

W=10
b=10

x_plot = np.arange(-4,14,0.01)
y_plot = np.exp(W*x_plot + b)/(1+np.exp(W*x_plot + b))
plt.plot(x_plot, y_plot)
plt.plot(x_plot, 0.5*np.ones_like(x_plot), c='red')
plt.scatter(x, y, c='blue')
plt.show()

################# initialize hyperparams ######################
learning_rate = 5e-2
epochs = 10000
time_sleep = 0.2
W = 1
b = 1
epoch_interval = 200
#################################################################

loss_list = []
W_list = []
b_list = []
for epoch in range(epochs):
    z = W*x + b

    # sigmoid function
    p = np.exp(z)/(1+np.exp(z))

    # Negative log likelihood
    loss = -np.sum(y*np.log(p)+(1-y)*np.log(1-p))/N

    # compute Gradient
    p_prime = p*(1-p)
    p_prime_W = p_prime*x
    p_prime_b = p_prime*1
    W_grad = -np.sum(y/p*p_prime_W-(1-y)/(1-p)*p_prime_W)/N
    b_grad = -np.sum(y/p*p_prime_b-(1-y)/(1-p)*p_prime_b)/N

    # update
    W = W - learning_rate*W_grad
    b = b - learning_rate*b_grad

    if epoch%epoch_interval == 0 :
        plt.figure()
        x_plot = np.arange(-2,14,0.01)
        y_plot = np.exp(W*x_plot + b)/(1+np.exp(W*x_plot + b))
        plt.plot(x_plot, y_plot)
        plt.plot(x_plot, 0.5*np.ones_like(x_plot), c='red')
        plt.scatter(x, y, c= 'blue', label = f'b: {b:.4f}\nW: {W:.4f}\nepoch: {epoch}\nLoss: {loss:.4f}')
        plt.legend()
        plt.show()
        time.sleep(time_sleep)
        clear_output(wait=True)

        loss_list.append(loss)
        W_list.append(W)
        b_list.append(b)

for l, w, b in zip(loss_list, W_list, b_list):
    print(f"loss:{l:.4f}, W:{w:.4f}, b:{b:.4f}")
