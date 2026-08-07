import numpy as np
from lightgbm import LGBMClassifier

X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)

model = LGBMClassifier(n_jobs=1, verbose=-1)
model.fit(X, y)
print("LightGBM basic fit succeeded")
