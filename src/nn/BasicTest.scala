package nn

import java.util.Random

object BasicTest {
  def main(args: Array[String]) = {
    val rnd = new Random()
    val network = NeuralNetwork.make()
    val a = network.input(0.0)
    val b = network.input(0.0)
    def iverson(b: Boolean) = if (b) 1.0 else 0.0
    val rate = network.input(-0.5)
    var weights = Vector[(network.Cell, network.Cell)]()
    def weight() = {
      val cell = network.input(rnd.nextGaussian)
      val grad = network.input(0.0)
      weights :+= ((cell, grad))
      network.sum(cell, network.product(rate, grad))
    }
    def neuron(input: Vector[network.Node]) = {
      val bias = weight()
      val terms = input.map(i => network.product(weight(), i))
      network.sigmoid(network.sum((bias +: terms): _*))
    }
    def layer(width: Int, input: Vector[network.Node]) = {
      Vector.fill(width)(neuron(input))
    }
    var l = Vector[network.Node](a, b)
    for (i <- 0 until 6) {
      l = layer(8, l)
    }
    val out = neuron(l)
    val c = network.input(0.0)
    val diff = network.sum(out, network.product(network.constant(-1.0), c))
    val err = network.product(diff, diff)
    var avg = 1.0
    for (i <- 0 until 1000000) {
      if (i % 1000 == 0) println(network(rate) + " " + avg)
      var batch = (node: network.Node) => 0.0
      for (j <- 0 until 10) {
        val inA = rnd.nextBoolean()
        val inB = rnd.nextBoolean()
        val inC = inA ^ inB
        network(a) = iverson(inA)
        network(b) = iverson(inB)
        network(c) = iverson(inC)
        network.reset()
        val grad = network.gradient(err)
        val q = batch
        avg = avg * 0.995 + network(err) * 0.005
        batch = (node: network.Node) => q(node) + grad(node)
      }
      for ((c, g) <- weights) {
        val g_ = network(g)
        network(g) = batch(c)
        network(c) = network(c) + network(rate) * g_
      }
      network(rate) = network(rate) - 0.001 * batch(rate)
    }
    for (inA <- Vector(0.0, 1.0)) {
      for (inB <- Vector(0.0, 1.0)) {
        network(a) = inA
        network(b) = inB
        network.reset()
        println(s"$inA $inB | ${network(out)}")
      }
    }
  }
}


// neuron = sigmoid(W v + bias)