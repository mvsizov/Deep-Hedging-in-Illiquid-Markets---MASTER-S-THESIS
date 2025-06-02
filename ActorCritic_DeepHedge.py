import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pickle


class HedgeEnvironment:
    """
    Environment takes inputs: prices, volatility, greeks, strikes, time to maturiy, so on...
    Returns the reward = change in PnL of Hedging position + payoff from option - transaction costs
    """
    
    def __init__(self, price_paths, vol_paths, strikes, lambda_, transaction_cost):
        # price paths
        self.S = price_paths
        # volatility paths
        self.V = vol_paths
        # strike prices
        self.K = torch.tensor(strikes, dtype=torch.float32, device=price_paths.device)
        # tensor of length = # of paths
        self.lambda_ = lambda_  
        # transaction costs
        self.transaction_cost = transaction_cost
        # number of paths and time till maturity of option
        self.T, self.N_path = price_paths.shape
        # set number of days per year
        self.days_per_year = 252.
        self.dt = 1 / self.days_per_year

    
    '''
    Reset function - to get back environment to time 0
    '''
    def reset(self):
        # reset time to 0
        self.t = 0
        # we assume that we start with 0 hedge position
        self.previous_position = torch.zeros(self.N_path, device=self.S.device)
        # find current price and volatility at t = 0
        price_0 = self.S[0]
        volatility_0   = self.V[0]

        # set tau0 - time to maturity at the beginning
        tau0 = (self.T - 1) * self.dt
        price_bs0, delta0, gamma0, vega0, theta0, vanna0, vomma0, charm0 = \
            HedgeAgent._black_scholes_calculator(price_0, self.K, tau0, volatility_0)
        # scale tau0 for all paths
        tau0 = torch.full_like(price_0, fill_value=tau0)
        
        '''
        Book value at t = 0 (we shorted 1 call): -Black-Scholes price of call + 
            hedge postion value (previous position is currently zero)  = -Black-Scholes price of call
        '''
        Book_Value0 = -price_bs0 + self.previous_position * price_0

        # define the state at t0
        state = torch.stack([
            price_0, volatility_0,
            delta0, gamma0, vega0, theta0, vanna0, vomma0, charm0,
            self.previous_position,
            tau0,
            Book_Value0
        ], dim=1)
        
        return state

    
    def step(self, action):
        # define price at t and t1, and volatility at t1
        price_t   = self.S[self.t]
        price_t1  = self.S[self.t + 1]
        volatility_t1   = self.V[self.t + 1]
        '''
        COMPUTING THE PNL OF HEDGE POSITION
        '''
        # change in price of underlying
        delta_p   = price_t1 - price_t   
        # pnl of hedge position (change in value of hedge position due to change in underlying prices
        pnl_hedge_position = self.previous_position * delta_p         
        # proportional transaction costs
        trans_cost = self.transaction_cost * torch.abs(action - self.previous_position) * price_t1  
        # REWARD = pnl from hedge - transaction costs
        reward    = pnl_hedge_position - trans_cost   

        '''
        PAYOFF FROM OPTION (SHORT CALL)
        '''
        done      = (self.t + 1 == self.T - 1)
        if done:
            # payoff from call = max(S - K; 0) 
            payoff = torch.clamp(price_t1 - self.K, min=0.0)
            # payoff is substracted from reward as we have a short position
            reward = reward - payoff
        # time to maturity from next step
        tau1 = (self.T - self.t - 1) * self.dt
        # compute greeks for next step
        price_bs1, delta1, gamma1, vega1, theta1, vanna1, vomma1, charm1 = \
            HedgeAgent._black_scholes_calculator(price_t1, self.K, tau1, volatility_t1)
        
        tau1 = torch.full_like(price_t1, fill_value=tau1)
        
        # Book Value before action
        Book_Value1 = -price_bs1 + self.previous_position * price_t1
        
        next_state = torch.stack([
            price_t1, volatility_t1,
            delta1, gamma1, vega1, theta1, vanna1, vomma1, charm1,
            self.previous_position,
            tau1,
            Book_Value1
        ], dim=1)

        # update position to a current action
        self.previous_position = action.detach()

        self.t += 1
        return next_state, reward, done

class HedgeMLP(nn.Module):
    '''
    Defining MLP which will be then be used in actor and critic
    '''
    def __init__(self, input_dim):
        super().__init__()
        
        # 3 layer MLP with sigmoid activation functions
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Sigmoid(),
            nn.Linear(256, 256),
            nn.Sigmoid(),
            nn.Linear(256, 256),
            nn.Sigmoid(),
            nn.Linear(256, 1),
        )
        # initializing weights
        self.net.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # initializing weights from normal distribution with zero biases
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # features are inputted to MLP
        return self.net(x).squeeze(-1)  


class HedgeAgent:
    '''
    Implements Actor-Critic Deep Hedging
    '''
    def __init__(self, seq_length, batch_sz, num_features,
                 transaction_cost=0.002):
        # length of input (number of days)
        self.seq_length = seq_length
        # batch size
        self.batch_sz = batch_sz
        # propotional transaction costs
        self.tr_cocts = transaction_cost
        
        # dimension of input features (+1 as we also have lambda [risk aversion level])
        input_dim = num_features
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # actor and critic
        self.actor  = HedgeMLP(input_dim).to(device)
        self.critic = HedgeMLP(input_dim).to(device)
        
        # second copy of critic MLP that will serve as a targer
        self.critic_target = HedgeMLP(input_dim).to(device)
        # copy all the parameters from critic into critic_target
        self.critic_target.load_state_dict(self.critic.state_dict())
        # turn critic_target into eval mode
        self.critic_target.eval()
        # the soft update coefficient
        self.tau = 1e-3
        # initiating the optimizers
        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)

        # lists to save losses from actor and critic
        self.critic_loss_history = []
        self.actor_loss_history  = []
        self.ce_history = []
        # for normalization of Greeks
        self.greek_mean = None  
        self.greek_std  = None
        
    def _soft_update(self):
        # soft-target update for critic
        # we get target critic (target_cr) and critic (cr) and soft update params of target
        for target_cr, cr in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_cr.data.copy_(self.tau * cr.data + (1 - self.tau) * target_cr.data)

    def compute_normalization(self, S, V, strikes):
        # run BS‐delta policy to collect raw states
        device = self.device
        price_paths = torch.as_tensor(S,   dtype=torch.float32, device=device)
        vol_paths   = torch.as_tensor(V,   dtype=torch.float32, device=device)
        strikes   = torch.as_tensor(strikes, dtype=torch.float32, device=device)
        # lambda does not influence greek calculations
        lambda_bs   = torch.full_like(strikes, fill_value=0.01)

        # setting the environment
        env   = HedgeEnvironment(price_paths, vol_paths, strikes, lambda_bs, self.tr_cocts)
        state = env.reset()
        collected = [state.cpu()]
    
        done = False
        while not done:
            # looking at pure Black-Scholes hedge
            tau = (env.T - env.t - 1) * env.dt
            _, delta_bs = self._black_scholes_calculator(
                state[:,0], strikes, tau, state[:,1],
                only_price_and_delta=True)
            state, _, done = env.step(delta_bs)
            collected.append(state.cpu())
    
        all_states  = torch.cat(collected, dim=0).numpy()
        # Greeks are columns 2–8 
        greek_data      = all_states[:, 2:9]
        # finding mean and standard deviation of each greek
        self.greek_mean = greek_data.mean(axis=0)
        self.greek_std  = greek_data.std(axis=0)

    # function to save normalization parameters
    def save_normalization_params(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump({"mean": self.greek_mean, "std": self.greek_std}, f)

    # function to load the normalization parameters
    def load_normalization_params(self, filepath):
        with open(filepath, "rb") as f:
            norm_params = pickle.load(f)
        self.greek_mean, self.greek_std = norm_params["mean"], norm_params["std"]


    @staticmethod
    def _black_scholes_calculator( S, K, tau, sigma,only_price_and_delta = False):
        """
        Black-Sholes calculator 
        Inputs: price, strike, time to maturity, sigma
        Outputs: price of option + Greeks
        """
        # convert tau (tau is time to maturity in years) a tensor 
        tau = torch.as_tensor(tau, dtype=S.dtype, device=S.device)
        # ensuring that tau is not zero to avoid devision by zero
        tau = torch.clamp(tau, min=1e-8)

        # calculate d1
        d1 = (torch.log(S / K) + (sigma**2 / 2.) * tau) / (sigma * torch.sqrt(tau))
        # calculate d2
        d2 = d1 - sigma * torch.sqrt(tau)
        # standard normal CDF
        cdf = lambda x:  (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0, device=S.device)))) / 2.
        # compute sqrt(2*Pi)
        sq_2Pi = torch.sqrt(torch.tensor(2 * torch.pi,
                                           device=S.device,
                                           dtype=S.dtype))
        # define pdf of d1,d2 (n1, n2)
        n1 = torch.exp(-(d1**2)/ 2) / sq_2Pi
        n2 = torch.exp(-(d2**2)/ 2) / sq_2Pi
        # compute N1 and N2
        N1 = cdf(d1)
        N2 = cdf(d2)
        # delta is equal to N1 [ cdf(d1) ]
        delta = N1
        # calculate the option price
        price = S * N1 - K * N2

        # if I need only price and delta: return them 
        if only_price_and_delta:
            return price, delta

        # Gamma - measures delta change relative to changes in price of underlying
        gamma = n1 / (S * sigma * torch.sqrt(tau))
        # Vega - sensitivity of option price to volatility of underlying
        vega  = S * n1 * torch.sqrt(tau)
        # Theta - sensitivity of option price to time to maturity
        theta = - (S * n1 * sigma) / (2 * torch.sqrt(tau))
        # Vanna - change of delta relative to vega
        vanna = vega * (1 - d1/(sigma*torch.sqrt(tau))) / S
        # Vomma - measures change in vega with respect to volatility
        vomma = vega * d1 * d2 / sigma
        # Charm - change in delta with respect to time to maturity
        charm = n1 * ((sigma**2 * tau) / 2 - torch.log(S / K)) / (2 * sigma * tau * torch.sqrt(tau))

        # returning calculated Greeks
        return price, delta, gamma, vega, theta, vanna, vomma, charm

    

    def train_model(self, S, V, strikes, lambda_value,epochs,save_dir = None):
        # device from 1st layer of actor
        device = self.actor.net[0].weight.device
        # make directory to save the results of training
        os.makedirs(save_dir, exist_ok=True)

        # find latest epoch in save_dir
        ckpts = [f for f in os.listdir(save_dir) if f.startswith("actor_") and f.endswith(".pth")]
        if ckpts:
            last = max(int(fn.split("_")[1].split(".")[0]) for fn in ckpts)
            # load actor & critic
            self.actor.load_state_dict(torch.load(f"{save_dir}/actor_{last}.pth", map_location=device))
            self.critic.load_state_dict(torch.load(f"{save_dir}/critic_{last}.pth", map_location=device))
            # load loss histories
            with open(f"{save_dir}/loss_history_{last}.pkl","rb") as f:
                hist = pickle.load(f)
            self.actor_loss_history, self.critic_loss_history = hist
            start_epoch = last
            print(f"Resuming from epoch {last}")
        else:
            start_epoch = 0
            

        # converting prices, volatility and strikes to tensors
        price_paths = torch.as_tensor(S, dtype=torch.float32, device=device)
        vol_paths   = torch.as_tensor(V, dtype=torch.float32, device=device)

        def save_ckpt(ep):
            torch.save(self.actor.state_dict(),   f"{save_dir}/actor_{ep}.pth")
            torch.save(self.critic.state_dict(),  f"{save_dir}/critic_{ep}.pth")
            with open(f"{save_dir}/loss_history_{ep}.pkl","wb") as f:
                pickle.dump((self.actor_loss_history, self.critic_loss_history), f)
                

        n_paths_for_strikes = price_paths.shape[1]
        strikes_t   = torch.as_tensor(strikes, dtype=torch.float32, device=device)
        if strikes_t.dim() == 0:
            strikes_t = strikes_t.expand(n_paths_for_strikes)
            
        try:
            # Load the Greek‐normalization params (adjust path as needed)
            self.load_normalization_params(os.path.join(save_dir, "greek_norm.pkl"))
            print('Normalization params are loaded!')
        except:
            self.compute_normalization(S, V, strikes_t.cpu().numpy())
            self.save_normalization_params(os.path.join(save_dir, "greek_norm.pkl"))
            print('redefined normalization params')

        # looping over epochs
        for epoch_i in range(start_epoch,epochs):
            try:
                epoch_critic_losses = []
                epoch_actor_losses  = []
                epoch_rewards        = []
                epoch_residuals      = []
    
                # randomly sampling batch
                batch_idx = np.random.choice(n_paths_for_strikes, size=self.batch_sz, replace=False)
                # prices, volatility, strikes corresponding to this batch
                S_batch = price_paths[:, batch_idx]   
                V_batch = vol_paths[:, batch_idx]      
                K_batch = strikes_t[batch_idx]       

                lambda_batch = torch.as_tensor(lambda_value, dtype=torch.float32, device=device).expand(self.batch_sz)
    
                # create an environment for current batch
                env = HedgeEnvironment(S_batch, V_batch, K_batch, lambda_batch, self.tr_cocts)
                state = env.reset() 
                
                idx = slice(2,9)
                gm = torch.as_tensor(self.greek_mean, dtype=torch.float32, device=device)
                gs = torch.as_tensor(self.greek_std,  dtype=torch.float32, device=device)
                state[:, idx] = (state[:, idx] - gm) / (gs + 1e-8)
                

                
    
                # Go throught all time from begining to maturity
                batch_pnls = torch.zeros(self.batch_sz, device=device)
                for t in range(self.seq_length - 1):
                    # CALCULATE THE RESIDUAL HEDGE (RESIDUAL FROM Black-Scholes hedge)
                    residual_hedge = self.actor(state)    
                    epoch_residuals.append(residual_hedge.mean().item())
                    # CALCULATE THE RESIDUAL VALUE (RESIDUAL FROM Black-Sholes value)
                    residual_value = self.critic(state)    
    
                    # compute the remaining time-to-maturity
                    tau = (env.T - env.t - 1) * env.dt
                    # baseline price & delta
                    price_bs, delta_bs, gamma, vega, theta, vanna, vomma, charm = self._black_scholes_calculator(state[:, 0], K_batch, tau, state[:, 1])
    
                    # estimate hedge and value based on BS hedge and valie and residual estimates
                    action = delta_bs + residual_hedge    # residual actor + BS delta
                    value0    = -price_bs + residual_value     # BS value + residual critic
                    
                    # the action is taken
                    next_state, reward, done = env.step(action)
                    epoch_rewards.append(reward.mean().item())
                    next_state[:, idx] = (next_state[:, idx] - gm) / (gs + 1e-8)

                    batch_pnls += reward
    
                    # at expiry we don't have any future value, so we zero it 
                    if done:
                        value1 = torch.zeros_like(value0)
                    else:
                        # remaining time to maturity in years
                        tau1 = (env.T - env.t - 1) * env.dt
                        # calculate next steps' BS price
                        price_bs1, _ = self._black_scholes_calculator(next_state[:, 0], K_batch, tau1, next_state[:, 1],only_price_and_delta=True)
                        with torch.no_grad():
                            # estimate the residual value through target critic
                            residual_value1 = self.critic_target(next_state)
                        # the value of next state
                        value1 = -price_bs1 + residual_value1
    
                    # --- 1) compute y for critic (detaching reward so no actor grads leak) ---
                    r_c      = reward.detach()             # cut the actor→reward graph
                    value1_c = value1.detach()             # already computed under no_grad(), but just to be explicit
                    y_c      = r_c + value1_c              # y for critic
                    
                    exp_c    = torch.exp(-lambda_batch * (y_c - value0))
                    loss_c   = ((exp_c / lambda_batch) - value0).mean()

                    epoch_critic_losses.append(loss_c.item())
        
                    # critic update (only critic parameters get grads)
                    self.critic_optimizer.zero_grad()
                    loss_c.backward() 
                    torch.nn.utils.clip_grad_norm_(
                        self.actor.parameters(), max_norm=1.0 )
                    self.critic_optimizer.step()
                    
                    
                    # --- 2) compute actor loss (we want actor to optimize the utility of next payoff) ---
                    # actor sees the actual reward + value1 (value1 is detached, so no critic grads)
                    exp_a  = torch.exp(-lambda_batch * (reward + value1.detach()))
                    loss_a = (exp_a / lambda_batch).mean() 
                    epoch_actor_losses.append(loss_a.item())

                    # actor update (only actor parameters get grads)
                    self.actor_optimizer.zero_grad()
                    loss_a.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.critic.parameters(), max_norm=1.0)
                    self.actor_optimizer.step()
                    
                    
                    # --- 3) soft‐update your target network as before ---
                    self._soft_update()
    
                    if done:
                        break
                
                # save to history of average losses on actor and critic
                average_critic = sum(epoch_critic_losses) / len(epoch_critic_losses)
                average_actor = sum(epoch_actor_losses)  / len(epoch_actor_losses)
                average_reward   = sum(epoch_rewards)       / len(epoch_rewards)
                average_residual = sum(epoch_residuals)     / len(epoch_residuals)
                self.critic_loss_history.append(average_critic)
                self.actor_loss_history.append(average_actor)

                # compute vector of utilities
                utils = torch.exp(-lambda_batch[0]  * batch_pnls)       # shape [batch_sz]
                # mean utility
                mean_u = utils.mean()
                # certainty equivalent
                ce = - (1.0/lambda_batch[0] ) * torch.log(mean_u)
                self.ce_history.append(ce.item())

                # print results of every 100 epochs
                if (epoch_i + 1) % 100 == 0:
                    print(
                        f"epoch {epoch_i+1:5d} | "
                        f"critic_loss: {average_critic:.6f}, "
                        f"actor_loss: {average_actor:.6f}, "
                        f"avg_reward: {average_reward:.6f}, "
                        f"avg_residual: {average_residual:.6f}, "
                        f"ce:{ce:.6f}"
                            )
                    

                # save each 1000 checkpoint:
                if (epoch_i + 1) % 1000 == 0:
                    save_ckpt(epoch_i+1)
                    print(f"Saved checkpoint at epoch {epoch_i+1}")
                
            except Exception as e:
                # on crash, save what we have so far
                print(f"Exception at epoch {epoch_i}, saving checkpoint…")
                save_ckpt(epoch_i)
                raise
       

    
    def predict(self, S, V, strikes, lambda_,save_dir=None):
        """
        method to predict hedge position based on prices, volatilities, strikes and lambda (risk aversion level)
        returns an array of hedge positions 
        """
        
        # Load the Greek‐normalization params (adjust path as needed)
        self.load_normalization_params(os.path.join(save_dir, "greek_norm.pkl"))

        # put both actor and critic networks into evaluation mode
        self.actor.eval()
        if self.critic is not None:
            self.critic.eval()

        device = next(self.actor.parameters()).device

        idx = slice(2, 9)
        gm  = torch.as_tensor(self.greek_mean, dtype=torch.float32, device=device)
        gs  = torch.as_tensor(self.greek_std,  dtype=torch.float32, device=device)
        
        # convert inputs to tensor
        price_paths = torch.as_tensor(S, dtype=torch.float32, device=device)
        volatility_paths   = torch.as_tensor(V, dtype=torch.float32, device=device)
        strikes   = torch.as_tensor(strikes, dtype=torch.float32, device=device)
        lambda_   = torch.as_tensor(lambda_, dtype=torch.float32, device=device)
        # if lambda is passed as single value: expand it
        if lambda_.dim() == 0:
            lambda_ = lambda_.expand(strikes.shape[0])

        # creating environment with given inputs and resetting it
        env = HedgeEnvironment(price_paths, volatility_paths, strikes, lambda_, self.tr_cocts)
        state = env.reset()
        state[:, idx] = (state[:, idx] - gm) / (gs + 1e-8)
        
        actions = []
        residual_hedges = []
        # go through time periods and calculate actions (hedge positions)

        for t in range(self.seq_length - 1):
            # disabling gradient
            with torch.no_grad():
                # calculate residual hedge
                residual_hedge = self.actor(state)
                print(residual_hedge)
                tau = (env.T - env.t - 1) * env.dt
                # calculate BS delta
                _, delta_bs = self._black_scholes_calculator(state[:, 0], strikes, tau, state[:, 1],only_price_and_delta = True)
                # calculate action (hedge) by correcting BS delta hedge
                action = delta_bs + residual_hedge
            # saving actions
            actions.append(action.cpu().numpy())
            residual_hedges.append(residual_hedge.cpu().numpy())
            state, _, done = env.step(action)
            state[:, idx] = (state[:, idx] - gm) / (gs + 1e-8)
            if done:
                break
        return np.stack(actions, axis=0),np.stack(residual_hedges,axis=0)

    
    def load_trained_model(self,critic_path = None, actor_path = None):
        """
        method to load saved weights for critic and/or actor
        """
        # load critic
        if critic_path: 
            self.critic.load_state_dict(
                torch.load(critic_path, map_location=self.device)
            )
            self.critic.eval()
        # load actor
        if actor_path:
            self.actor.load_state_dict(
                torch.load(actor_path, map_location=self.device)
            )
            self.actor.eval()
        print(f"Models loaded! Critic: {critic_path}, Actor: {actor_path}")

        
         