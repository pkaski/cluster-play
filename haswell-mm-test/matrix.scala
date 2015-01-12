
/*

import matrix._

def mulPerf(n: Int, repeats: Int, seed: Int) {
  val g = new scala.util.Random(seed)
  val flops = n.toDouble*n*n*2.0
  val a = Matrix.rand(n, g.nextInt(100000))
  val b = Matrix.rand(n, g.nextInt(100000))
  val r = Matrix.zero(n)
  for(i <- 0 until repeats) {
    println("n = %7d".format(n))
    a*b
  }
}

mulPerf(24*128, 10, 12345)
mulPerf(2*24*128, 10, 12345)
mulPerf(4*24*128, 10, 12345)
mulPerf(8*24*128, 10, 12345)
mulPerf(32768, 1, 12345)

*/

package matrix

class Matrix(val n: Int) extends Serializable {
  require(n >= 0)
  val entries = new Array[Double](n*n)
  def apply(i: Int, j: Int) = {
    require(i >= 0 && i < n && j >= 0 && j < n)
    entries(i*n+j)
  }
  def apply(i: Int) = {
    require(i >= 0 && i < n)
    entries.slice(i*n,(i+1)*n)
  }
  def update(i: Int, j: Int, v: Double) {
    require(i >= 0 && i < n && j >= 0 && j < n)
    entries(i*n+j)=v
  }
  def transpose = {
    val r = new Matrix(n)
    var i = 0
    while(i < n) {
      var j = 0
      while(j < n) {
        r.entries(i+j*n)=entries(i*n+j)
        j = j+1
      }
      i = i+1
    }
    r
  }
  def unary_- = {
    val r = new Matrix(n)
    var i = 0
    while(i < n*n) {
      r.entries(i) = -entries(i)
      i = i+1
    }
    r
  }
  def +(that: Matrix) = {
    require(that.n == n)
    val r = new Matrix(n)
    var i = 0
    while(i < n*n) {
      r.entries(i) = entries(i)+that.entries(i)
      i = i+1
    }
    r
  }
  def -(that: Matrix) = this + (-that)
  def *(that: Matrix) = {
    val result = new Matrix(n)
    Matrix.nat.mulpar(n, result.entries, this.entries, that.entries)
    result
  }
  def view {
    for(i <- 0 until n) {
      for(j <- 0 until n) {
        print(" %9.3f".format(this(i,j)))
      }
      println("")
    }
  }
  def viewsupport {
    for(i <- 0 until n) {
      for(j <- 0 until n) {
        if(math.abs(this(i,j)) > 0.001)
          print("X")
        else
          print("-")
      }
      println("")
    }
  }
}

object Matrix {  
  val nat = new mynative

  def zero(n: Int) = {
    val r = new Matrix(n)
    var i = 0
    while(i < n*n) {
      r(i/n,i%n) = 0.0
      i = i+1
    }
    r
  }
  def eye(n: Int) = {
    val r = Matrix.zero(n)
    var i = 0
    while(i < n) {
      r(i,i) = 1.0
      i = i+1
    }
    r
  }
  def rand(n: Int, s: Int) = {
    val r = new Matrix(n)
    val g = new scala.util.Random(s)
    var i = 0
    while(i < n*n) {
      r(i/n,i%n) = g.nextDouble()
      i = i+1
    }
    r
  }
}


