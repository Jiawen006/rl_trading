import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding

from config import config


class StockEnvTrain(gym.Env):
    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim=10,
        hmax=100,
        initial_amount=config.INITIAL_AMOUNT,
        transaction_cost_pct=0.2,
        profit_reward_scaling=1e-9,
        loss_reward_scaling=2e-9,
        state_space=91,
        action_space=10,
        tech_indicator_list=config.INDICATORS,
        day=1,
    ) -> None:
        """
        This is the constructor
        :param df: a large file with all information
        :param stock_dim: number of stocks, in this problem it should be 10
        :param hmax: maximum number of shares to trade
        :param initial_amount: initial balance
        :param transaction_cost_pct: transfer to slippage later
        :param reward_scaling: scaling factor for reward
        :param state_space: integer, number of state space
        :param tech_indicator_list:
        :param day:
        """
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.profit_reward_scaling = profit_reward_scaling
        self.loss_reward_scaling = loss_reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list

        # action_space normalization and shape is 5 * self.stock_dim (1 market order + 4 limit order)
        # for each stock it is arranged as:
        # market order
        # price of limit order 1
        # size of limit order 1
        # price of limit order 2
        # size of limit order 2
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.state_space,)
        )

        # load data from dataframe
        self.data = self.df.loc[self.day - 1, :]

        self.terminal = False

        hold_position = [0] * self.stock_dim

        # initialize state
        # Shape of observation space = 91
        # [Current Balance]
        # Open *10 + High *10 + Low * 10 + Close * 10
        # Owned shares
        # [macd 1-10]+ [rsi 1-10] + [cci 1-10] + [adx 1-10]
        self.state = (
            [self.initial_amount]
            + self.data.Open.values.tolist()
            + self.data.High.values.tolist()
            + self.data.Low.values.tolist()
            + self.data.Close.values.tolist()
            + hold_position
            + sum(
                [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
                [],
            )
        )

        self.reward = 0
        self.cost = 0
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.trades = 0
        self._seed()

    def market_order(self, actions) -> None:
        # firstly, check how much money that will spend
        money_spent = 0
        action_list = actions
        for index in range(len(action_list)):
            hold_index = 4 * self.stock_dim + index + 1
            cur_open = self.df.loc[self.day + 1, :].Open.values[index]
            pre_close = self.data.Close.values[index]
            this_action = action_list[index]
            if this_action > 0:
                # this is a buy market order
                money_spent += this_action * (
                    cur_open + abs(cur_open - pre_close) * self.transaction_cost_pct
                )
            elif this_action < 0:
                # # this is a sell market order
                # check whether short too much, make this order as zero if it short above threshold
                if (
                    self.state[hold_index] - abs(action_list[index])
                    > config.short_threshold[index]
                ):
                    # we can short at this series
                    money_spent += this_action * (
                        cur_open - abs(cur_open - pre_close) * self.transaction_cost_pct
                    )
                else:
                    # we can not short at this position
                    action_list[index] = 0

        # check whether all order can be processed
        if money_spent < self.state[0]:
            # we can process the order, update one by one
            self.state[0] = self.state[0] - money_spent
            self.state[
                (4 * self.stock_dim + 1) : (5 * self.stock_dim + 1)
            ] += action_list
            self.cost = self.cost + sum(
                (
                    abs(action_list)
                    * abs(
                        self.df.loc[self.day + 1, :].Open.values
                        - self.data.Close.values
                    )
                )
                * self.transaction_cost_pct
            )

            self.trades += 1
        else:
            # we can not process the order
            # print("attempt to spend {}, but only {} available. Stop executing on day {}.".format(money_spent, self.state[0],self.day))
            pass

    def limit_order(self, index, price, size) -> None:
        low = self.df.loc[self.day + 1, :].Low.values[index]
        high = self.df.loc[self.day + 1, :].High.values[index]
        # number of stocks hold at this index
        hold_index = 4 * self.stock_dim + index + 1
        if size > 0:
            # this is a buy limit order
            if low < price:
                # price is within the range
                execution_price = min(price, high)
                # check the maximum number of stock we can trade at this position
                max_trade = self.state[0] / execution_price
                if size <= max_trade:
                    # we can trade at this size
                    # update balance
                    self.state[0] = self.state[0] - size * execution_price
                    # update number of shares hold
                    self.state[hold_index] += size
                    self.trades += 1
                else:
                    # print("Atempt to buy {} stocks of series {}, but failed since the money only can buy {} stocks"
                    #       .format(size, index + 1, max_trade))
                    pass
            else:
                # price is not in the range
                pass
        elif size < 0:
            # this is a sell limit order
            if high > price:
                # price is within the range
                execution_price = max(price, low)
                # we can always operate sell, but make sure that it will cause bankrupt
                # update balance
                self.state[0] = self.state[0] - size * execution_price
                # update stock hold
                self.state[hold_index] = self.state[hold_index] + size
                self.trades += 1
            else:
                # price is not in the range
                pass
        else:
            # no limit order at this series
            pass

    def check_bankrupt(self) -> None:
        end_total_asset = self.state[0] + sum(
            np.array(self.state[1 : (self.stock_dim + 1)])
            * np.array(self.state[(4 * self.stock_dim + 1) : (self.stock_dim * 5 + 1)])
        )
        if end_total_asset < 0:
            self.terminal = True
            self.reward = (
                abs(
                    config.banrupt_penalty
                    * abs(end_total_asset)
                    * self.loss_reward_scaling
                )
                * -1
            )
            self.rewards_memory.append(self.reward)
            print(
                "Bankrupt at day {}! end_total_asset:{}".format(
                    self.day, end_total_asset
                )
            )
            print("=================================")

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            # the trading ends, calculate the networth : balance + shares * open
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(
                    self.state[(4 * self.stock_dim + 1) : (self.stock_dim * 5 + 1)]
                )
            )
            self.asset_memory.append(end_total_asset)
            print("===========================")
            print("begin_total_asset:{}".format(self.asset_memory[0]))
            print("end_total_asset:{}".format(end_total_asset))
            print(
                "final return: {}".format((end_total_asset / self.initial_amount) - 1)
            )
            print("{} days trade".format(self.trades))
            df_total_value = pd.DataFrame(self.asset_memory)
            # df_total_value.to_csv('results/account_value_train.csv')

            # print("total_trades: ", self.trades)
            df_total_value.columns = ["account_value"]
            df_total_value["daily_return"] = df_total_value.pct_change(1)
            sharpe = (
                (252**0.5)
                * df_total_value["daily_return"].mean()
                / df_total_value["daily_return"].std()
            )
            print("Sharpe: ", sharpe)
            print("=================================")
            self.reward = sharpe
        else:
            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(
                    self.state[(4 * self.stock_dim + 1) : (5 * self.stock_dim + 1)]
                )
            )
            # operate market order
            market_action = actions[0 : self.stock_dim]
            market_action[abs(market_action) < 0.05] = 0
            market_action = market_action * config.order_coefficient

            self.market_order(market_action)

            self.day += 1
            self.data = self.df.loc[self.day, :]
            # load next state
            self.state = (
                [self.state[0]]
                + self.data.Open.values.tolist()
                + self.data.High.values.tolist()
                + self.data.Low.values.tolist()
                + self.data.Close.values.tolist()
                + list(self.state[(4 * self.stock_dim + 1) : (5 * self.stock_dim + 1)])
                + sum(
                    [
                        self.data[tech].values.tolist()
                        for tech in self.tech_indicator_list
                    ],
                    [],
                )
            )

            # decide order at day t, execute at day t+1
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(
                    self.state[(4 * self.stock_dim + 1) : (self.stock_dim * 5 + 1)]
                )
            )
            self.asset_memory.append(end_total_asset)

            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.columns = ["account_value"]
            if self.day > 2:
                df_total_value["daily_return"] = df_total_value[
                    "account_value"
                ].pct_change(1)
                if df_total_value["daily_return"].std() != 0:
                    sharpe = (
                        (252**0.5)
                        * df_total_value["daily_return"].mean()
                        / df_total_value["daily_return"].std()
                    )
                    self.reward = sharpe
            else:
                self.reward = 0

            bankrupt = self.check_bankrupt()
            if bankrupt == False:
                self.rewards_memory.append(self.reward)

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.day = 1
        self.data = self.df.loc[self.day, :]
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        # initiate state
        self.state = (
            [self.initial_amount]
            + self.data.Open.values.tolist()
            + self.data.High.values.tolist()
            + self.data.Low.values.tolist()
            + self.data.Close.values.tolist()
            + [0] * self.stock_dim
            + sum(
                [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
                [],
            )
        )
        return self.state

    def render(self, mode="human"):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
