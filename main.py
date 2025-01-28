import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Step 1: Data Preprocessing with enhanced handling
def load_and_preprocess(filepath):
    df = pd.read_csv(filepath, parse_dates=["Report Date"])
    df.sort_values("Report Date", inplace=True)

    # Handle missing prices and conversions
    df["Product Price"] = df["Product Price"].replace(0, np.nan).ffill().bfill()
    df["Organic Conversion Percentage"] = df["Organic Conversion Percentage"].replace(
        0, np.nan
    )
    df["Ad Conversion Percentage"] = df["Ad Conversion Percentage"].replace(0, np.nan)

    # Separate historical and future data
    historical_data = df[df["Total Sales"] > 0].copy()
    future_data = df[df["Total Sales"] <= 0].copy()

    # Impute missing values using historical medians
    imputer = SimpleImputer(strategy="median")
    features = [
        "Product Price",
        "Organic Conversion Percentage",
        "Ad Conversion Percentage",
    ]
    historical_data[features] = imputer.fit_transform(historical_data[features])

    # Prepare training data
    X = historical_data[features].values
    y = historical_data["Total Sales"].values

    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (
        df,
        imputer,
        scaler,
        historical_data,
        future_data,
        X_train,
        X_test,
        y_train,
        y_test,
    )


# Step 2: Enhanced sales predictor
class SalesPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=200, random_state=42)
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")

    def train(self, X_train, y_train):
        X_train_imputed = self.imputer.fit_transform(X_train)
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        self.model.fit(X_train_scaled, y_train)

    def predict(self, price, organic_conv, ad_conv):
        features = np.array([[float(price), float(organic_conv), float(ad_conv)]])
        features_imputed = self.imputer.transform(features)
        features_scaled = self.scaler.transform(features_imputed)
        return float(self.model.predict(features_scaled)[0])


# Step 3: RL Environment with proper time series handling
class PricingEnv:
    def __init__(self, historical_data, future_data, sales_predictor, price_median):
        self.historical_data = historical_data
        self.future_data = future_data
        self.sales_predictor = sales_predictor
        self.price_median = price_median
        self.current_step = 0
        self.state_dim = 4  # [price, sales, organic_conv, ad_conv]

        # Store these for normalizing rewards
        self.max_price = historical_data["Product Price"].max()
        self.max_sales = historical_data["Total Sales"].max()
        self.max_conversion = max(
            historical_data["Organic Conversion Percentage"].max(),
            historical_data["Ad Conversion Percentage"].max(),
        )

    def reset(self):
        self.current_step = 0
        return self._get_state(0)

    def _get_state(self, idx):
        row = (
            self.historical_data.iloc[idx]
            if idx < len(self.historical_data)
            else self.future_data.iloc[idx - len(self.historical_data)]
        )
        # Handle NaN conversions
        return np.array(
            [
                float(row["Product Price"]),
                float(row["Total Sales"]),
                float(
                    row["Organic Conversion Percentage"]
                    if not pd.isna(row["Organic Conversion Percentage"])
                    else self.sales_predictor.imputer.statistics_[1]
                ),
                float(
                    row["Ad Conversion Percentage"]
                    if not pd.isna(row["Ad Conversion Percentage"])
                    else self.sales_predictor.imputer.statistics_[2]
                ),
            ]
        )

    def step(self, action_price):
        # Get current state components
        current_row = (
            self.historical_data.iloc[self.current_step]
            if self.current_step < len(self.historical_data)
            else self.future_data.iloc[self.current_step - len(self.historical_data)]
        )

        # Predict sales for new price
        predicted_sales = max(
            0.0,
            self.sales_predictor.predict(
                action_price,
                current_row["Organic Conversion Percentage"],
                current_row["Ad Conversion Percentage"],
            ),
        )

        # Get baseline predicted sales from CSV
        baseline_predicted = float(current_row["Predicted Sales"])
        if pd.isna(baseline_predicted) or baseline_predicted <= 0:
            baseline_predicted = predicted_sales

        # Normalize components for stable rewards
        norm_price = action_price / self.max_price
        norm_sales = predicted_sales / self.max_sales
        norm_conversion = (
            float(current_row["Organic Conversion Percentage"])
            + float(current_row["Ad Conversion Percentage"])
        ) / (2 * self.max_conversion)

        # Calculate reward components with safety checks
        price_reward = (
            max(0, (norm_price - 0.5)) * 0.5
        )  # Reward for pricing above median
        sales_reward = norm_sales * 0.3  # Reward for high sales
        conversion_reward = norm_conversion * 0.2  # Reward for good conversion

        # Calculate punishment with safety check
        sales_diff = max(0, baseline_predicted - predicted_sales)
        punishment = min(
            2.0, (sales_diff / self.max_sales) * 2
        )  # Cap punishment at 2.0

        # Combine rewards with safety check
        total_reward = max(
            -10.0,
            min(10.0, price_reward + sales_reward + conversion_reward - punishment),
        )

        # Update state
        self.current_step += 1
        next_state = self._get_state(self.current_step)
        done = (
            self.current_step >= len(self.historical_data) + len(self.future_data) - 1
        )

        return (
            next_state,
            float(total_reward),  # Ensure reward is float
            done,
            {
                "predicted_sales": predicted_sales,
                "baseline_sales": baseline_predicted,
                "price": action_price,
                "components": {
                    "price_reward": price_reward,
                    "sales_reward": sales_reward,
                    "conversion_reward": conversion_reward,
                    "punishment": punishment,
                },
            },
        )


# Step 4: Enhanced DDPG Agent with price exploration
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_price):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )
        self.max_price = max_price

    def forward(self, state):
        return (self.net(state) + 1) * (self.max_price / 2)  # Scale to [0, max_price]


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state, action):
        # Ensure action is 2D
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        return self.net(torch.cat([state, action], 1))


class DDPG:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_price,
        lr_actor=1e-4,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.005,
    ):
        self.actor = Actor(state_dim, action_dim, max_price)
        self.actor_target = Actor(state_dim, action_dim, max_price)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.gamma = gamma
        self.tau = tau
        self.max_price = max_price

        # Initialize target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def get_action(self, state, exploration_noise=0.3):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        noise = np.random.normal(0, exploration_noise) + 0.1
        action = float(action + noise)  # Add noise before converting to float
        return np.clip(action, 0, self.max_price)

    def update(self, replay_buffer, batch_size=64):
        if len(replay_buffer) < batch_size:
            return

        # Sample batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # Ensure actions have correct shape (batch_size, action_dim)
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)

        # Update critic
        with torch.no_grad():
            target_actions = self.actor_target(next_states)
            if target_actions.dim() == 1:
                target_actions = target_actions.unsqueeze(-1)
            target_q = self.critic_target(next_states, target_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_actions = self.actor(states)
        if actor_actions.dim() == 1:
            actor_actions = actor_actions.unsqueeze(-1)
        actor_loss = -self.critic(states, actor_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for param, target_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done),
        )

    def __len__(self):
        return len(self.buffer)


# Main execution flow
def train_and_predict(filepath):
    # Load and preprocess data
    (
        df,
        imputer,
        scaler,
        historical_data,
        future_data,
        X_train,
        X_test,
        y_train,
        y_test,
    ) = load_and_preprocess(filepath)

    # Train sales predictor
    sales_predictor = SalesPredictor()
    sales_predictor.train(X_train, y_train)

    # Initialize environment
    price_median = historical_data["Product Price"].median()
    env = PricingEnv(historical_data, future_data, sales_predictor, price_median)

    # Initialize DDPG agent
    max_price = (
        historical_data["Product Price"].max() * 1.5
    )  # Allow 50% above historical max
    ddpg = DDPG(state_dim=4, action_dim=1, max_price=max_price)
    replay_buffer = ReplayBuffer(100000)

    # Training parameters
    num_episodes = 200
    batch_size = 64

    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = ddpg.get_action(state)
            next_state, reward, done, info = env.step(action)

            replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward

            if len(replay_buffer) > batch_size:
                ddpg.update(replay_buffer, batch_size)

            state = next_state

        print(f"Episode {episode+1}/{num_episodes} | Total Reward: {episode_reward}")

    # Generate final price recommendation
    test_state = env.reset()
    with torch.no_grad():
        recommended_price = ddpg.actor(torch.FloatTensor(test_state)).item()

    return recommended_price


if __name__ == "__main__":
    products = {
        # "soapnut": "./dataset/soapnutshistory.csv",
        "woolball": "./dataset/woolballhistory.csv",
    }

    for product, filepath in products.items():
        print(f"Training model for {product}...")
        optimal_price = train_and_predict(filepath)
        print(f"Recommended price for {product}: ${optimal_price:.2f}\n")
