// Databricks notebook source
// MAGIC %md
// MAGIC **Main Project (100 pts)** \
// MAGIC Implement closed form solution when m(number of examples is large) and n(number of features) is small:
// MAGIC \\[ \scriptsize \mathbf{\theta=(X^TX)}^{-1}\mathbf{X^Ty}\\]
// MAGIC Here, X is a distributed matrix.

// COMMAND ----------

import breeze.linalg._

// Step 1: Create an example RDD for matrix X and vector y
val matrix_X: Array[Array[Double]] = Array(
  Array(1.0, 5.0, 3.0), 
  Array(2.0, 6.0, 9.0),
  Array(7.0, 8.0, 11.0),
  Array(1.0, 5.0, 4.0)
)

val matrix_Y: Array[Double] = Array(10.0, 20.0, 30.0)

// Convert arrays to RDDs
val X: RDD[DenseVector[Double]] = sc.parallelize(matrix_X.map(arr => DenseVector(arr)))
val y: RDD[Double] = sc.parallelize(matrix_Y)

// Ensure all vectors in RDD X have the same length
val vectorLength = X.first().length
//println(vectorLength)
val XFiltered = X.filter(_.length == vectorLength)

// Check if all vectors have the same length
if (XFiltered.count() != X.count()) {
  println("Error: Vectors in RDD X have different lengths.")
} else {
  // Step 2: Compute (X^T X) using outer product method
  val XT_X: DenseMatrix[Double] = {
    // Convert RDD X to an Array of DenseVectors
    val XArray = XFiltered.collect()
    val outerProducts = for {
      i <- 0 until vectorLength
      j <- 0 until vectorLength
    } yield XArray.map(v => v(i) * v(j)).sum
    new DenseMatrix(vectorLength, vectorLength, outerProducts.toArray)
  }

  // Step 3: Convert the result matrix to a Breeze Dense Matrix and compute inverse
  val XT_X_inv: DenseMatrix[Double] = inv(XT_X)
  //println("XT_X_inv",XT_X_inv)

  // Step 4: Compute (X^T y)
  val XT_y: DenseVector[Double] = {
    val yt = y.collect()
    val result = XFiltered.map(x => x.toArray.zip(yt).map { case (xi, yi) => xi * yi }).reduce((a, b) => a.zip(b).map { case (ai, bi) => ai + bi }) 
    //println("XT_y",result)
    DenseVector(result) 
  }

  // Step 5: Multiply (X^T X)^-1 with (X^T y)
  val theta: DenseVector[Double] = XT_X_inv * XT_y

  // Print the result
  println("Closed-form solution for theta:")
  println("theta",theta)
}

// COMMAND ----------

// MAGIC %md
// MAGIC **Bonus(30 pts)** \
// MAGIC Implement the gradient descent update for linear regression is: \\[ \scriptsize \mathbf{\theta}_{j+1} = \mathbf{\theta}_j - \alpha \sum_i (\mathbf{\theta}_i^\top\mathbf{x}^i  - y^i) \mathbf{x}^i \\]

// COMMAND ----------

import org.apache.spark.mllib.linalg.{Vectors}

// Step 1: Initialize vectors θ = 0  and α = 0.001
val theta: DenseVector[Double] = DenseVector.zeros[Double](vectorLength)
val alpha: Double = 0.001
val numIterations: Int = 5

// Step 2: Implement a function to compute the summand (θᵀx - y)x 
def computeSummand(x: DenseVector[Double], y: Double, theta: DenseVector[Double]): DenseVector[Double] = {
  (theta.t * x - y) * x
  
}

// Test the computeSummand function on two examples
val exampleX1 = DenseVector(1.0, 5.0, 3.0)
val exampleY1 = 10.0
val summand1 = computeSummand(exampleX1, exampleY1, theta)

val exampleX2 = DenseVector(14.0, 25.0, 37.0)
val exampleY2 = 20.0
val summand2 = computeSummand(exampleX2, exampleY2, theta)

println("Summand 1:")
println(summand1)
println("Summand 2:")
println(summand2)

// Step 8: Implement a function to compute Root Mean Squared Error (RMSE)
def computeRMSE(predictions: RDD[(Double, Double)]): Double = {
  val squaredErrors = predictions.map { case (label, prediction) => math.pow(label - prediction, 2) }
  math.sqrt(squaredErrors.mean())
}

// Test the computeRMSE function on an example RDD
val examplePredictions = sc.parallelize(Seq((1.0, 11.0), (2.0, 19.0), (3.0, 28.0)))
val exampleRMSE = computeRMSE(examplePredictions)
println("Example RMSE:")
println(exampleRMSE)

// Step 9: Implement the gradient descent update function
def gradientDescentUpdate(data: RDD[(DenseVector[Double], Double)],
                          theta: DenseVector[Double],
                          alpha: Double): (DenseVector[Double], Array[Double]) = {
  val numExamples = data.count()

  val updatedTheta = (1 to numIterations).foldLeft(theta) { (currentTheta, _) =>
    val gradient = data.map { case (x, y) => computeSummand(x, y, currentTheta) }.reduce(_ + _)
    val deltaTheta = gradient * (alpha / numExamples.toDouble)
    currentTheta - deltaTheta
  }

  // Compute training errors (RMSE) for each iteration
  val errors = (1 to numIterations).map { _ =>
    val predictions = data.map { case (x, y) => (y, x.dot(updatedTheta)) }
    computeRMSE(predictions)
  }
  (updatedTheta, errors.toArray)
}

// Step 10: Test the gradientDescentUpdate function on an example RDD
val exampleData: RDD[(DenseVector[Double], Double)] = X.zip(y)
val (finalWeights, finalErrors) = gradientDescentUpdate(exampleData, theta, alpha)

println("Final Weights:")
println(finalWeights)
println("Training Errors (RMSE) for each iteration:")
finalErrors.foreach(println)

// Step 11: Run gradient descent for 5 iterations and print the results
val (trainedWeights, trainingErrors) = gradientDescentUpdate(exampleData, theta, alpha)

println("Trained Weights:")
println(trainedWeights)
println("Training Errors (RMSE) for each iteration:")
trainingErrors.foreach(println)
