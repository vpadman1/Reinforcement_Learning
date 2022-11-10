
# import the library
import numpy as np

# setting the parameters gamma and alpha for Q-Learning
gamma = 0.75 # discount feature
alpha = 0.9 # learning rate

# Part 1 - defining the environment

# defining the states
location_to_state = {
    'A':0,
    'B':1,
    'C':2,
    'D':3,
    'E':4,
    'F':5,
    'G':6,
    'H':7,
    'I':8,
    'J':9,
    'K':10,
    'L':11
    }

# defining the actions
actions = [0,1,2,3,4,5,6,7,8,9,10,11]

# defining the rewards
R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0],
              [1,0,1,0,0,1,0,0,0,0,0,0],
              [0,1,0,0,0,0,1,0,0,0,0,0],
              [0,0,0,0,0,0,0,1,0,0,0,0],
              [0,0,0,0,0,0,0,0,1,0,0,0],
              [0,1,0,0,0,0,0,0,0,1,0,0],
              [0,0,1,0,0,0,1,1,0,0,0,0],
              [0,0,0,1,0,0,1,0,0,0,0,1],
              [0,0,0,0,1,0,0,0,0,1,0,0],
              [0,0,0,0,0,1,0,0,1,0,1,0],
              [0,0,0,0,0,0,0,0,0,1,0,1],
              [0,0,0,0,0,0,0,1,0,0,1,0]])

# making a mapping form the states to the locations
statetoLocation = {state: location for location, state in location_to_state.items()}

# Making the final function that will return the optimal route
def route(startingLocation, endingLocation):
    R_new = np.copy(R)
    endingState = location_to_state[endingLocation]
    R_new[endingState, endingState] = 1000
    # initializing the Q-Values
    Q = np.array(np.zeros([12,12]))
    # implementing the q-learning process
    for i in range(1000):
        currentState = np.random.randint(0,12)
        playableActions = []
        for j in range(12):
            if R_new[currentState, j] > 0:
                playableActions.append(j)
        nextState = np.random.choice(playableActions)
        temporalDifference = R_new[currentState, nextState] + gamma * Q[nextState, np.argmax(Q[nextState,])] - Q[currentState, nextState]
        Q[currentState, nextState] += alpha * temporalDifference
    optimalRoute = [startingLocation]
    nextLocation = startingLocation
    while (nextLocation != endingLocation):
        startingState =  location_to_state[startingLocation]
        nextState = np.argmax(Q[startingState,])
        nextLocation = statetoLocation[nextState]
        optimalRoute.append(nextLocation)
        startingLocation = nextLocation
    return optimalRoute

print("Route: ")
print(route('E', 'D'))



