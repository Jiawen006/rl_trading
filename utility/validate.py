import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv

from env.environment_train import StockEnvTrain


def validate(model, data):
    """
    the function validate the trained model in a given period
    :return: sharpe ratio
    """
    env = DummyVecEnv([lambda: StockEnvTrain(data)])
    obj = env.reset()
    reward = 0
    for i in range(data["Index"].nunique() - 1):
        action, _states = model.predict(obj)
        obj, reward, done, info = env.step(action)
    return reward[0]


model = "trained_models/A2C.zip"
model = A2C.load(model)
data = pd.read_csv(
    "/Users/jiawentang/Documents/文件/工作/Preparation for Working/Project/FYP/Draft 1/DATA/train_processed.csv"
)
# trading_day = data['Index'].nunique()
# 2200 trading days
validate_data = data.tail(90 * 10)
validate_data.index = validate_data.Index.factorize()[0]
reward = validate(model, validate_data)

print("end")
