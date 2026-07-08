from utils import *

advertising = pd.DataFrame(pd.read_csv("advertising.csv"))
advertising = advertising[['TV', 'Sales']]

X = advertising['TV'].to_numpy()
y = advertising['Sales'].to_numpy()

X.shape

X = (X-X.mean()+1)/np.std(X)

plt.scatter(X, y, c = 'navy')
plt.xlabel('TV')
plt.ylabel('Sales')
plt.title('TV-Sales Scatter')
plt.show()

W = 5
b = 15
y_pred = W*X + b

fig = plt.figure(figsize = (10,10))

ax = fig.add_subplot(111)
data_fitted_visualizer(X, y, y_pred)


# make sure X and y are already defined
def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

W_list = [5, 10, 15]
b_list = [5, 10, 15]

best_loss = float('inf')
best_W = None
best_b = None

for W in W_list:
    for b in b_list:
        y_pred = W * X + b
        loss = compute_loss(y, y_pred)
        print(f"W = {W}, b = {b}, Loss = {loss:.4f}")
        if loss < best_loss:
            best_loss = loss
            best_W = W
            best_b = b

print("\nBest combination:")
print(f"W = {best_W}, b = {best_b}, Lowest Loss = {best_loss:.4f}")

################# initialize hyperparams ######################

W = 10
b = -10
learning_rate = 0.1
epochs = 50
tol = 1e-3
time_sleep = 0.2

################################################################

W_mesh, b_mesh, loss_out = mesh_loss_out_generator(X,y, (-12,12), (-20,25), 100)

loss_list = []
W_list = []
b_list = []
for epoch in range(epochs):
    y_pred = W*X + b

    # MSE loss fucntion
    loss = ((y_pred - y)**2).mean()

    # compute gradient
    W_grad = ((y_pred - y)*X).mean()
    b_grad = (y_pred - y).mean()

    # parameter update
    W = W - learning_rate * W_grad
    b = b - learning_rate * b_grad

    # Plotting
    fig = plt.figure(figsize = (10,10))

    ax = fig.add_subplot(221)
    data_fitted_visualizer(X, y, y_pred, epoch)

    ax = fig.add_subplot(222)
    contour_2d(W, b, W_mesh, b_mesh, loss_out, ax, cmap_color=plt.cm.coolwarm, x_label = 'Weight', y_label = 'bias')

    ax = fig.add_subplot(223, projection='3d')
    loss_func_visualizer_3d(W, b, W_mesh, b_mesh, loss_out, X, y, ax, angle = (20,15))

    ax = fig.add_subplot(224, projection='3d')
    loss_func_visualizer_3d(W, b, W_mesh, b_mesh, loss_out, X, y, ax, angle = (20,165), clear =True)

    time.sleep(time_sleep)

    loss_list.append(loss)
    W_list.append(W)
    b_list.append(b)

    if loss < tol:
        break

for l, w, b in zip(loss_list, W_list, b_list):
    print(f"loss:{l:.4f}, W:{w:.4f}, b:{b:.4f}")

from numpy.linalg import inv

X_ = np.vstack([X, np.ones_like(X)]).T
W, b = np.dot(np.matmul(inv(np.matmul(X_.T, X_)), X_.T), y)
y_pred = W*X + b

fig = plt.figure(figsize = (10,10))

ax = fig.add_subplot(111)
data_fitted_visualizer(X, y, y_pred)
