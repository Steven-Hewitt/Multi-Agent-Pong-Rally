import gym
env = gym.make('Pong-2p-v0')
env.reset()
done=False
while (not done):
    env.render()
    _,r,done,_ = env.step(env.action_space.sample()) # take a random action
    print(r)