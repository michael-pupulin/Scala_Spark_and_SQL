import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel, LinearRegressionTrainingSummary}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}



object Main {


  // setting up spark session
  val spark: SparkSession = SparkSession.builder().appName("NOAA Data").master("local[*]").getOrCreate()
  import spark.implicits._
  spark.sparkContext.setLogLevel("ERROR")

  // read in csv data
  val df: DataFrame = spark.read.option("header", "true").csv("twisters.csv")

  // filter out scenarios where tornado magnitude is unknown (webpage says this is why some are set to -9)
  val mydf: DataFrame = df.select("yr","mo","dy","st","mag","inj", "fat").filter(df("mag") =!= "-9")

  //mydf.show(5)
  mydf.printSchema()

  // Using describe to learn something about the magnitude of tornadoes, how many they injure or kill.
  println("Here is a quick summary for some of our numerical columns:")
  mydf.select("mag","inj","fat").describe().show()

  // Finding the event that killed the most people
  println("The tornado that resulted in the greatest number of casualties:")
  val newdf: Dataset[Row] = mydf.filter(mydf("fat") === mydf.agg(max("fat")).collect().head(0))
  newdf.show()

  // Now I do some analysis by the year and by the state ( filter, groupby, aggregate, sort, show) -

  // Which year saw the most tornadoes?
  println("The years that saw the most tornadoes: ")
  mydf.groupBy("yr").agg(count("yr")).sort(col("count(yr)").desc).show(5)

  // Which years saw the most F5 tornadoes?
  println("The years that saw the most F5 tornadoes are: ")
  mydf.filter(df("mag") === "5").groupBy("yr").agg(count("yr")).sort(col("count(yr)").desc).show(5)

  // Average tornado strength by state?
  println("The average tornado magnitude by state: ")
  mydf.groupBy("st").agg(avg("mag")).sort(col("avg(mag)").desc).show(5)

  // Which years saw the most F5 tornadoes?
  println("The states that have seen the most F5 tornadoes: ")
  mydf.filter(df("mag") === "5").groupBy("st").agg(count("st")).sort(col("count(st)").desc).show(5)

  // Doing some linear regression

  // I want to see if the relationship between time (in years) and number of tornadoes is a linear one.
  // I take a dataframe that has the year in one column and the number of tornadoes per year in the next.
  // Here, I am also using cast to change year from string type to integer type
  val lindata: Dataset[Row] =   mydf.groupBy("yr").agg(count("yr")).sort(col("yr").asc).withColumn("yr", $"yr".cast("Integer"))


  // Declaring a new linear regression model
  // We need to work with spark vectors to do regression (or any ML I believe), We use the assembler.
  val assembler: VectorAssembler = new VectorAssembler()
    .setInputCols(Array("yr"))
    .setOutputCol("year")

  // Using the assembler to transform our dataset
  val output: DataFrame = assembler.transform(lindata)
  println("A sample of our data to regress on:")
  output.select("year", "count(yr)").show(5)

  // Set feature and label column. Define parameters for regression.
  val lr: LinearRegression = new LinearRegression()
    .setMaxIter(10)
    .setRegParam(0.3)
    .setElasticNetParam(0.8)
    .setFeaturesCol("year")
    .setLabelCol("count(yr)")

  // Fit the model
  val lrModel: LinearRegressionModel = lr.fit(output)

  // Print the coefficients and intercept for linear regression
  println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

  // Summarize the model over the training set and print out some metrics
  val trainingSummary: LinearRegressionTrainingSummary = lrModel.summary
  println(s"numIterations: ${trainingSummary.totalIterations}")
  println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
  trainingSummary.residuals.show()
  println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
  println(s"r2: ${trainingSummary.r2}")



  spark.stop()
  def main(args: Array[String]): Unit = {
    println("Spark is now closed. End of program.")
  }

}