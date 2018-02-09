Feature: Solve smooth unimodal problems using EI and grid search
    In order to solve smooth unimodal optimization problems
    As a prefopt user
    I want to be able to run a binary preference experiment with an
    expected-improvement acquisition strategy and a grid-search optimizer

    Scenario: Solve quadratic problem
        Given a negative-quadratic utility function defined on [-2, 5] and a expected-improvement acquisition function with a probit preference model and a grid-search optimizer
        When the experiment is run 3 times and each run uses 20 iterations
        Then the result should be within 5.0e-1 of the optimum at least 3 times
