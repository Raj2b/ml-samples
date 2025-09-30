# COMP 3105 Fall 2025 Assignment 1
# Carleton University
# NOTE: This is a sample script to show you how your functions will be called. 
#       You can use this script to visualize your models once you finish your codes. 
#       This script is not meant to be thorough (it does not call all your functions). 
#       We will use a different script to test your codes. 
import numpy as np
from matplotlib import pyplot as plt
import YourName.A1codes as A1codes


def _plotReg():

	# simple 2D example
	n = 30  # number of data points
	d = 1  # dimension
	noise = 0.2  # noise level
	X = np.random.randn(n, d)  # input matrix
	y = X + np.random.randn(n, 1) * noise + 2  # ground truth label

	plt.scatter(X, y, marker='x', color='k')  # plot data points

	# learning
	X = np.concatenate((np.ones((n, 1)), X),  axis=1)  # augment input
	w_L2 = A1codes.minimizeL2(X, y)
	y_hat_L2 = X @ w_L2
	w_Linf = A1codes.minimizeLinf(X, y)
	y_hat_Linf = X @ w_Linf

	# plot models
	plt.plot(X[:, 1], y_hat_L2, 'b', marker=None, label='$L_2$')
	plt.plot(X[:, 1], y_hat_Linf, 'r', marker=None, label='$L_\infty$')
	plt.legend()
	plt.show()


def _plotCls():

	# 2D classification example
	m = 100
	d = 2
	c0 = np.array([[1, 1]])  # cls 0 center
	c1 = np.array([[-1, -1]])  # cls 1 center
	X0 = np.random.randn(m, 2) + c0
	X1 = np.random.randn(m, 2) + c1

	# plot data points
	plt.scatter(X0[:, 0], X0[:, 1], marker='x', label='Negative')
	plt.scatter(X1[:, 0], X1[:, 1], marker='o', label='Positive')

	X = np.concatenate((X0, X1), axis=0)
	X = np.concatenate((np.ones((2*m, 1)), X),  axis=1)  # augment input
	y = np.concatenate([np.zeros([m, 1]), np.ones([m, 1])], axis=0)  # class labels

	# find optimal solution
	w_opt = A1codes.find_opt(A1codes.logisticRegObj, A1codes.logisticRegGrad, X, y)

	# plot models
	x_grid = np.arange(-4, 4, 0.01)
	plt.plot(x_grid, (-w_opt[0]-w_opt[1]*x_grid) / w_opt[2], '--k')
	plt.legend()
	plt.show()

# --------- EXTRA TESTS FOR PART (d) ---------

def _testPreprocess():
    """
    Sanity-check preprocessCCS:
      - loads Concrete_Data.xls from DATA_DIR
      - prints shapes, dtypes, a couple of values
    """
    # >>>>>> SET THIS FOLDER: it must contain Concrete_Data.xls <<<<<<
    DATA_DIR = "/Users/abdulmalik/Downloads/concrete+compressive+strength"

    X, y = A1codes.preprocessCCS(DATA_DIR)
    print("[preprocessCCS] X shape:", X.shape, "y shape:", y.shape)
    print("[preprocessCCS] X dtype:", X.dtype,  "y dtype:", y.dtype)

    # Basic assertions to catch common mistakes
    assert y.ndim == 2 and y.shape[1] == 1, "y must be (n,1)"
    assert X.shape[0] == y.shape[0], "X and y must have same row count"
    # UCI Concrete dataset typically has 8 feature columns (no bias here)
    assert X.shape[1] == 8, "Expected 8 features before bias is added in runCCS"

    # Peek at a few values
    print("[preprocessCCS] first row of X:", X[0])
    print("[preprocessCCS] first 5 y values:", y[:5].ravel())
    print("[preprocessCCS] ✅ looks good\n")


def _testRunCCS():
    """
    Runs runCCS once and prints the 2x2 train/test loss tables with labels.
    """
    # >>>>>> SET THIS FOLDER: it must contain Concrete_Data.xls <<<<<<
    DATA_DIR = "/Users/abdulmalik/Downloads/concrete+compressive+strength"

    train_avg, test_avg = A1codes.runCCS(DATA_DIR)

    # Raw arrays
    print("[runCCS] Train (2x2):\n", train_avg)
    print("[runCCS] Test  (2x2):\n",  test_avg)

    # Pretty print with labels
    models  = ["L2 model", "L_inf model"]
    metrics = ["L2 loss (MSE)", "L_inf loss (max |err|)"]

    def _fmt(mat, title):
        print(f"\n[runCCS] {title}")
        print("             " + " | ".join(f"{m:>18}" for m in metrics))
        for i, mname in enumerate(models):
            cells = " | ".join(f"{mat[i, j]:18.6f}" for j in range(2))
            print(f"{mname:>12} | {cells}")

    _fmt(train_avg, "Average TRAIN losses over runs")
    _fmt(test_avg,  "Average TEST  losses over runs")
    print("\n[runCCS] ✅ finished\n")

def _testPreprocessBCW():
    """
    Sanity-check preprocessBCW:
      - loads wdbc.data from DATA_DIR
      - prints shapes, dtypes, a couple of values
    """
    # >>>>>> SET THIS FOLDER: it must contain wdbc.data <<<<<<
    DATA_DIR = "/Users/abdulmalik/Downloads/breast+cancer+wisconsin+diagnostic"

    X, y = A1codes.preprocessBCW(DATA_DIR)
    print("[preprocessBCW] X shape:", X.shape, "y shape:", y.shape)
    print("[preprocessBCW] X dtype:", X.dtype,  "y dtype:", y.dtype)

    # Basic assertions to catch common mistakes
    assert y.ndim == 2 and y.shape[1] == 1, "y must be (n,1)"
    assert X.shape[0] == y.shape[0], "X and y must have same row count"
    # UCI BCW (diagnostic) has 30 features (no bias here)
    assert X.shape[1] == 30, "Expected 30 features before bias is added in runBCW"

    # Peek at a few values
    print("[preprocessBCW] first row of X:", X[0])
    print("[preprocessBCW] first 10 y values:", y[:10].ravel())
    print("[preprocessBCW] ✅ looks good\n")


def _testRunBCW():
    """
    Runs runBCW once and prints average train/test accuracy over runs.
    """
    # >>>>>> SET THIS FOLDER: it must contain wdbc.data <<<<<<
    DATA_DIR = "/Users/abdulmalik/Downloads/breast+cancer+wisconsin+diagnostic"

    train_avg, test_avg = A1codes.runBCW(DATA_DIR)

    print("[runBCW] Average TRAIN accuracy over runs:", f"{train_avg:.4f}")
    print("[runBCW] Average TEST  accuracy over runs:", f"{test_avg:.4f}")
    print("\n[runBCW] ✅ finished\n")

if __name__ == "__main__":

	#_plotReg()
	#_plotCls()
    _testPreprocess()
    _testRunCCS()
    
	# BCW tests
    _testPreprocessBCW()
    _testRunBCW()

