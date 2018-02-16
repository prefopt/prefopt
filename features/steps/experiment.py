from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing

import prefopt.experiment
import prefopt.presenter
import prefopt.utility

# pylint: disable=function-redefined,undefined-variable,unused-argument


@given('a {utility_name:S} utility function defined on [{lower:d}, {upper:d}]'
       ' and a {acquisition_strategy:S} acquisition function'
       ' with a {model_name:S} preference model'
       ' and a {optimizer_name:S} optimizer')
def step_impl(context, utility_name, lower, upper, acquisition_strategy,
              model_name, optimizer_name):
    context.bounds = prefopt.utility.BoundingBox([(lower, upper)])

    utility_classname = prefopt.experiment.to_classname(
        utility_name, 'UtilityFunction')
    utility_class = getattr(prefopt.utility, utility_classname)
    utility_function = utility_class(context.bounds)
    context.optimum = utility_function.argmax

    context.input_presenter = prefopt.presenter.FunctionInputPresenter(
        utility_function)
    x, y = context.bounds.sample(), context.bounds.sample()
    choice = context.input_presenter.get_choice(x, y)
    context.seed_data = (x, y, choice)

    context.acquisition_strategy = acquisition_strategy
    context.model_name = model_name
    context.optimizer_name = optimizer_name


@when('the experiment is run {number_of_runs:d} times'
      ' and each run uses {number_of_iterations:d} iterations')
def step_impl(context, number_of_runs, number_of_iterations):
    context.results = []
    for _ in range(number_of_runs):
        q = multiprocessing.Queue()

        def func(q):
            acquirer = prefopt.experiment.create_acquirer(
                context.acquisition_strategy,
                context.model_name,
                context.optimizer_name,
                context.bounds
            )
            output_presenter = prefopt.presenter.StdoutPresenter()
            experiment = prefopt.experiment.PreferenceExperiment(
                acquirer,
                context.input_presenter,
                output_presenter,
                context.seed_data
            )
            for _ in range(number_of_iterations):
                experiment.run()
            best, = experiment.acquirer.best
            q.put(best)

        p = multiprocessing.Process(target=func, args=(q,))
        p.start()
        context.results.append(q.get())
        p.join()


@then('the result should be within {epsilon:e} of the optimum'
      ' at least {minimum_number_of_successes:d} times')
def step_impl(context, epsilon, minimum_number_of_successes):
    number_of_successes = sum(abs(x - context.optimum) < epsilon
                              for x in context.results)
    assert number_of_successes >= minimum_number_of_successes, context.results
