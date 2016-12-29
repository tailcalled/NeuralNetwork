package nn

trait NeuralNetwork {
  type Node
  type Cell <: Node
  def product(nodes: Node*): Node
  def sum(nodes: Node*): Node
  def sigmoid(node: Node): Node
  def input(init: Double): Cell
  def constant(value: Double): Node = input(value)
  def update(cell: Cell, value: Double): Unit
  def divide(num: Node, den: Node): Node
  // gradient(x)(y) = dx/dy
  // gradient(x) = dx/d_
  def gradient(x: Node): (Node => Double)
  def apply(node: Node): Double
  def reset(): Unit
}

object NeuralNetwork {
  
  def make(): NeuralNetwork = {
    class BasicNeuralNetwork() extends NeuralNetwork {
      type Node = Int
      type Cell = Node
      var nodes = new Array[Double](0)
      var nodeInputs = new Array[Array[Int]](0)
      val TypeProduct = 0
      val TypeSum = 1
      val TypeSigmoid = 2
      val TypeInput = 3
      val TypeDivide = 4
      var nodeTypes = new Array[Int](0)
      def product(inputs: Node*) = {
        val node = nodes.length
        nodes :+= 0.0
        nodeInputs :+= inputs.toArray
        nodeTypes :+= TypeProduct
        node
      }
      def sum(inputs: Node*) = {
        val node = nodes.length
        nodes :+= 0.0
        nodeInputs :+= inputs.toArray
        nodeTypes :+= TypeSum
        node
      }
      def sigmoid(input: Node) = {
        val node = nodes.length
        nodes :+= 0.0
        nodeInputs :+= Array(input)
        nodeTypes :+= TypeSigmoid
        node
      }
      def divide(num: Node, den: Node) = {
        val node = nodes.length
        nodes :+= 0.0
        nodeInputs :+= Array(num, den)
        nodeTypes :+= TypeDivide
        node
      }
      def input(initial: Double) = {
        var node = nodes.length
        nodes :+= initial
        nodeInputs :+= Array[Int]()
        nodeTypes :+= TypeInput
        node
      }
      def update(cell: Cell, value: Double) = {
        nodes(cell) = value
      }
      def apply(node: Node) = nodes(node)
      def reset() = {
        var i = 0
        while (i < nodes.length) {
          nodeTypes(i) match {
            case TypeProduct =>
              var prod = 1.0
              val inputs = nodeInputs(i)
              var j = 0
              while (j < inputs.length) {
                prod *= nodes(inputs(j))
                j += 1
              }
              nodes(i) = prod
            case TypeSum =>
              var sum = 0.0
              val inputs = nodeInputs(i)
              var j = 0
              while (j < inputs.length) {
                sum += nodes(inputs(j))
                j += 1
              }
              nodes(i) = sum
            case TypeSigmoid =>
              val x = nodes(nodeInputs(i)(0))
              val s =
                if (x < 0) math.exp(x) / (1.0 + math.exp(x))
                else 1.0 / (1.0 + math.exp(-x))
              nodes(i) = s
            case TypeDivide =>
              val x = nodes(nodeInputs(i)(0))
              val y = nodes(nodeInputs(i)(1))
              nodes(i) = x/y
            case TypeInput =>
              // do nothing
          }
          i += 1
        }
      }
      def gradient(node: Node) = {
        val grad = new Array[Double](nodes.length)
        grad(node) = 1.0
        var i = node
        while (i >= 0) {
          nodeTypes(i) match {
            case TypeProduct =>
              val inputs = nodeInputs(i)
              var x = 0
              while (x < inputs.length) {
                var prod = 1.0
                var j = 0
                while (j < inputs.length) {
                  if (x != j) prod *= nodes(inputs(j))
                  j += 1
                }
                grad(inputs(x)) += prod * grad(i)
                x += 1
              }
            case TypeSum =>
              val inputs = nodeInputs(i)
              var x = 0
              while (x < inputs.length) {
                grad(inputs(x)) += grad(i)
                x += 1
              }
            case TypeSigmoid =>
              grad(nodeInputs(i)(0)) += nodes(i) * (1 - nodes(i)) * grad(i)
            case TypeDivide =>
              grad(nodeInputs(i)(0)) += grad(i) / nodes(nodeInputs(i)(1))
              grad(nodeInputs(i)(1)) -= grad(i) * nodes(nodeInputs(i)(0)) / (nodes(nodeInputs(i)(1)) * nodes(nodeInputs(i)(1)))
            case TypeInput =>
              // do nothing
          }
          i -= 1
        }
        grad(_)
      }
    }
    new BasicNeuralNetwork()
  }
  
}