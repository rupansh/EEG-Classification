from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline

from utils.raw_loaders import load_training_sk
from sktime.transformations.panel.rocket import MiniRocketMultivariate

print("Loading Data..")
((ts, gt), (vts, vgt)) = load_training_sk()

pipeline = make_pipeline(
    MiniRocketMultivariate(),
    SGDClassifier(loss="log_loss")
)

print("Fitting Data...")
for x, y in zip(ts, enumerate(gt)):
    pipeline.fit(x, y)

print("Validation Score...")
score = pipeline.score(vts, vgt)
print(score)
