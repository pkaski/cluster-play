
package matrix

class mynative extends Serializable
{
  System.loadLibrary("mynative") // remember to set LD_LIBRARY_PATH

  @native def ping()
  @native def mulpar(size: Int, result: Array[Double], left: Array[Double], right: Array[Double])
}

