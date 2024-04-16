package com.github.pohhnii

class Trainer(initModel: Model, val populationSize: Int, val tournamentSize: Int) {

    class Individual(val model: Model, var error: Double = Double.MAX_VALUE)

    private var population = Array(populationSize) {
        val model = initModel.copy()
        model.mutate()
        Individual(model)
    }

    var bestIndividual = Individual(initModel)

    fun errors(errorFunction: (Model) -> Double) {
        for (i in population.indices) {
            population[i].error = errorFunction(population[i].model)
        }
    }

    fun nextGen(mutationRate: Double = 0.1, sigma: Double = 0.1): Double {
        bestIndividual = population.minBy { it.error }

        population = Array(populationSize) {
            val parent1 = tournament()
            val parent2 = tournament()
            val child = parent1.copy()
            child.crossover(parent2)
            child.mutate(mutationRate, sigma)
            Individual(child)
        }

        return bestIndividual.error
    }

    private fun tournament(): Model {
        population.shuffle()
        val members = population.take(tournamentSize)
        return members.minBy { it.error }.model
    }
}