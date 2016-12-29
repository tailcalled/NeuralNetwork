package nn

import javax.imageio.ImageIO
import java.io.File
import java.util.Random
import java.awt.image.BufferedImage

object LenaCode {
  def main(args: Array[String]) = {
    val lena = ImageIO.read(new File("lena512.png"))
    val rand = new Random(0)
    val sz = 6
    def at(x: Int, y: Int) = {
      def ch(n: Int) = n.toDouble / 255
      val clip = Array.tabulate(sz, sz) ((dx, dy) => {
        val rgb = lena.getRGB(x + dx, y + dy)
        ch(rgb & 0xFF)
      })
      clip
    }
    def clip() = {
      val x = rand.nextInt(512 - 2 * sz)
      val y = rand.nextInt(512 - sz)
      (at(x, y), at(x + sz, y))
    }
    val clips = Array.fill(1000)(clip())
    val network = NeuralNetwork.make()
    val makeWeightsEncode = {
      val s = Stream.continually(network.input(rand.nextGaussian()))
      () => s.iterator
    }
    val weightsDecide = Stream.continually(network.input(rand.nextGaussian())).iterator
    var weights = Vector[(network.Cell, network.Cell)]()
    val rate = network.input(-0.2)
    def weight()(implicit source: Iterator[network.Cell]) = {
      val w = source.next()
      val g = source.next()
      network(g) = 0.0
      weights :+= ((w, g))
      network.sum(w, network.product(rate, g))
    }
    def neuron(input: Vector[network.Node])(implicit source: Iterator[network.Cell]) = {
      val bias = weight()
      val terms = input.map(i => network.product(weight(), i))
      network.sigmoid(network.sum((bias +: terms): _*))
    }
    def layer(width: Int, input: Vector[network.Node])(implicit source: Iterator[network.Cell]) = {
      Vector.fill(width)(neuron(input))
    }
    def encode() = {
      implicit val source = makeWeightsEncode()
      val in = Vector.fill(sz, sz)(network.input(0.0))
      var l: Vector[network.Node] = in.flatten
      for (i <- 0 until 2) {
        l = layer(6, l)
      }
      (in, l)
    }
    def decide(left: Vector[network.Node], right: Vector[network.Node]) = {
      implicit val source = weightsDecide
      var l = left ++ right
      for (i <- 0 until 2) {
        l = layer(6, l)
      }
      neuron(l)
    }
    val (imgL, outL) = encode()
    val (imgR, outR) = encode()
    val out = decide(outL, outR)
    val expected = network.input(0.0)
    val diff = network.sum(out, network.product(network.constant(-1.0), expected))
    val err = network.product(diff, diff)
    var avg = 0.25
    def loadImg(q: Array[Array[Double]], slot: Vector[Vector[network.Cell]]) = {
      for (x <- 0 until sz) {
        for (y <- 0 until sz) {
          network(slot(x)(y)) = q(x)(y)
        }
      }
    }
    def load(i: Int, z: Boolean, slot: Vector[Vector[network.Cell]]) = {
      val (a, b) = clips(i)
      val q = if (z) b else a
      loadImg(q, slot)
    }
    for (i <- 0 until 500000) {
      var batch = (node: network.Node) => 0.0
      for (j <- 0 until 10) {
        val nice = rand.nextBoolean()
        val ix = rand.nextInt(clips.length - 1)
        load(ix, false, imgL)
        if (nice) load(ix, true, imgR)
        else load(ix + 1, false, imgR)
        network(expected) = if (nice) 1.0 else 0.0
        network.reset()
        val grad = network.gradient(err)
        val q = batch
        batch = (node: network.Node) => q(node) + grad(node)
      }
      avg = avg * 0.98 + network(err) * 0.02
      if (i % 100 == 0) {
        if (i % 1000 == 0) println(s"it $i : ${avg} (rate: ${network(rate)})")
      }
      for ((c, g) <- weights) {
        val g_ = network(g)
        network(g) = batch(c)
        network(c) = network(c) + network(rate) * batch(c)
      }
      network(rate) = network(rate) - 0.00001 * batch(rate)
      if (i % 25000 == 0) {
        val img = new BufferedImage(512 / sz - 1, 512 / sz, BufferedImage.TYPE_INT_ARGB)
        for (x <- 0 until 512 / sz - 1) {
          for (y <- 0 until 512 / sz) {
            def ch(c: Double) = (c * 255).floor.toInt max 0 min 255
            def col(c: Double) = (0xFF << 24) | (ch(c) << 16) | (ch(c) << 8) | (ch(c) << 0)
            loadImg(at(x * sz, y * sz), imgL)
            loadImg(at(x * sz + sz, y * sz), imgR)
            network.reset()
            img.setRGB(x, y, col(network(out)))
          }
        }
        ImageIO.write(img, "png", new File(s"$i probability lena.png"))
      }
    }
    for (factor <- 0 until outL.length) {
      val f = outL(factor)
      val img = new BufferedImage(512 / sz, 512 / sz, BufferedImage.TYPE_INT_ARGB)
      for (x <- 0 until 512 / sz) {
        for (y <- 0 until 512 / sz) {
          def ch(c: Double) = (c * 255).floor.toInt max 0 min 255
          def col(c: Double) = (0xFF << 24) | (ch(c) << 16) | (ch(c) << 8) | (ch(c) << 0)
          loadImg(at(x * sz, y * sz), imgL)
          network.reset()
          img.setRGB(x, y, col(network(f)))
        }
      }
      ImageIO.write(img, "png", new File(s"factor $factor lena.png"))
    }
    ()
/*    for (i <- 0 until clips.length) {
      for (clip <- Vector(false, true)) {
        load(i, clip, imgL)
        network.reset()
        val (a, b) = clips(i)
        val q = if (clip) b else a
        val img = new BufferedImage(sz, sz, BufferedImage.TYPE_INT_ARGB)
        def ch(c: Double) = (c * 255).floor.toInt max 0 min 255
        def col(c: Double) = (0xFF << 24) | (ch(c) << 16) | (ch(c) << 8) | (ch(c) << 0)
        for (x <- 0 until sz) {
          for (y <- 0 until sz) {
            img.setRGB(x, y, col(q(x)(y)))
          }
        }
        val values = outL.map(network.apply _).map(_ * 100).map(_.toInt)
        ImageIO.write(img, "png", new File(s"$i $clip ${values.mkString(",")}.png"))
        for (i <- 0 until values.length) {
          ImageIO.write(img, "png", new File(s"a $i ${values(i)}.png"))
        }
      }
    }*/
  }
}