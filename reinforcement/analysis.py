# analysis.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    answerDiscount = 0.9
    answerNoise = 0
    # because narrow bridge is abysmal, a zero noise would lead to the 'high reward' further terminal state (it would attempt to cross the bridge)
    # High discount = evaluation based on future rewards
    # High noise = high unpredictability
    return answerDiscount, answerNoise


###### Agent would perform expected behavior if followed optimal policy with no noise ######

# Prefer close exit (low discount), risk cliff, low living reward
def question3a():
    answerDiscount = 0.2
    answerNoise = 0
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward

# Prefer close exit (low discount), avoid cliff, low living reward
def question3b():
    answerDiscount = 0.3
    answerNoise = 0.2
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward

# Prefer distant exit (high discount), risk cliff, high living reward
def question3c():
    answerDiscount = 0.9
    answerNoise = 0
    answerLivingReward = 0.5
    return answerDiscount, answerNoise, answerLivingReward

# Prefer distant exit (high discount), avoid cliff, high living reward
def question3d():
    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0.5
    return answerDiscount, answerNoise, answerLivingReward

# Avoid both exits and cliff (really high living reward)
def question3e():
    answerDiscount = 0.5
    answerNoise = 0
    answerLivingReward = 100
    return answerDiscount, answerNoise, answerLivingReward

def question8():
    answerEpsilon = None
    answerLearningRate = None
    return answerEpsilon, answerLearningRate
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
