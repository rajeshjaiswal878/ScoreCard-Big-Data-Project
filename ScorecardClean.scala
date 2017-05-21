import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql._
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql.DataFrame
import org.apache.spark.mllib.recommendation._
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql._
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql.DataFrame
import org.apache.spark.mllib.recommendation._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.RFormula
import org.apache.spark.ml.regression.GeneralizedLinearRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.sql.functions.round
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils


/**
 * Scorecard Evalution for given student data such as ACT SAT score
 * Done by Rajesh Jaiswal and  Rajat Hatwar Spring 2017 Big Data Analytics
 */


object ScorecardClean{
  case class RegionState(region:Double,state:String,adr:Double)
  case class StateCollege(state:String,cname:String,adr:Double,totalcost:Double,instfee:Double,outsfee:Double)
  def main(args: Array[String]) {
    if (args.length<4) {
			      System.err.println("Usage: Mean elapsed time <input path> <output path>");
			      System.exit(-1);
			    } 
     //---------------- create Spark context with Spark configuration-----------------------------
    val spark = SparkSession.builder().appName("Ratings").config("spark.sql.warehouse.dir","/Desktop/raj").enableHiveSupport().master("local").getOrCreate()
    val sc=spark.sparkContext
    import spark.implicits._
    
    //In this part of code we are going to load data data and remove null and split each line----------------------------.
    val DataA = sc.textFile(args(0))
    val header=DataA.first()
    val updata=DataA.filter(line=>line!=header)
    val finalRdd=updata.map(line=>line.split(',')).map(line=>(line(3),line(4),line(5),line(17),line(18),line(21),line(22),line(36),line(55),line(59),line(376),line(378),line(379)))
    val reqData=finalRdd.map(line=>(line._1+","+line._2+","+line._3+","+line._4+","+line._5+","+line._6+","+line._7+","+line._8+","+line._9+","+line._10+","+line._11+","+line._12+","+line._13)).filter(line => !line.contains("NULL"))
    val printData=reqData.take(1000)
    val check1=sc.parallelize(printData)   
    check1.repartition(1).saveAsTextFile(args(1))
    
   //------------------------Creation of Table to use under Rank function capability----------------------------- 
   val regionToState=finalRdd.map(line=>(line._5+","+line._3+","+line._8)).distinct().filter(line => !line.contains("NULL")).map(line=>(line.split(',')(0),line.split(',')(1),line.split(',')(2))).map{case(region,state,admrate)=>RegionState(region.toDouble,state,admrate.toDouble)}.toDF
   regionToState.createOrReplaceTempView("RSTable")
   //------------------------------Region to State Combination-----------------------------
   val stateToCollege=finalRdd.map(line=>(line._3+","+line._1+","+line._8+","+line._11+","+line._12+","+line._13)).distinct().filter(line => !line.contains("NULL")).map(line=>(line.split(',')(0),line.split(',')(1),line.split(',')(2),line.split(',')(3),line.split(',')(4),line.split(',')(5))).map{case(state,cname,admrate,totalcost,instfee,outsfee)=>StateCollege(state,cname,admrate.toDouble,totalcost.toDouble,instfee.toDouble,outsfee.toDouble)}.toDF
   
   stateToCollege.createOrReplaceTempView("SCTable") 
   
   val rtsResult=spark.sql("select region,state,adr from(select region,state,adr,rank() over (order by adr desc) as rank from RSTable) where rank<2")
   //------------------------------State to College Combination-----------------------------
   val stcResult=spark.sql("select state,cname,adr,totalcost,instfee,outsfee,rank() over (order by adr desc) as rank from SCTable")
   //------------------------------Combination of region to state and state to college-----------------------------
   val finalResult=spark.sql("select RS.region,RS.state,SC.cname,RS.adr,SC.totalcost,SC.instfee,SC.outsfee,rank() over(order by RS.adr desc) as rank from RSTable RS,SCTable SC where RS.state=SC.state").distinct()
   finalResult.createOrReplaceTempView("FNTable")
   
    
   //-----------------------------Random forest Model generation -------------------------------------
    val DataD = sc.textFile(args(1)).map(line=>line.split(',')).map(splits=>(splits(3).toInt,splits(5).toDouble,splits(6).toDouble,splits(7).toDouble,splits(8).toDouble,splits(9).toDouble,splits(4).toDouble)).toDF
    val DataB = DataD.selectExpr("_1 as SZIP","_2 as LATT","_3 as LONG", "_4 as ADR","_5 as ACT","_6 as SAT","_7 as REG")    
    val formula = new RFormula().setFormula("REG ~ SZIP + LATT + LONG + ADR + ACT + SAT").setFeaturesCol("features").setLabelCol("label")
    val output = formula.fit(DataB).transform(DataB)
    
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(output)
    val Array(trainingData, testData) = output.randomSplit(Array(0.7, 0.3))
        // Train a RandomForest model.
    val rf = new RandomForestRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures")
        // Chain indexer and forest in a Pipeline.
    val pipeline = new Pipeline().setStages(Array(featureIndexer, rf))
        // Train model. This also runs the indexer.
    val model = pipeline.fit(trainingData)
      // Make predictions.
      val predictions = model.transform(testData)
      predictions.createOrReplaceTempView("PRDTable")
      
     val requiredPRD=spark.sql("Select ADR,ACT,SAT,round(prediction) as predn from PRDTable")
      requiredPRD.createOrReplaceTempView("PDTable")
   val CollegeChoiceResult=spark.sql("select PD.ACT,PD.SAT,PD.predn,FN.cname,FN.totalcost,FN.instfee,FN.outsfee from PDTable PD,FNTable FN where PD.predn=FN.region")
      
      //-------------------------------------------------Final Output----------------------------------------------
     CollegeChoiceResult.show(5)
     CollegeChoiceResult.rdd.repartition(1).saveAsTextFile(args(3))  
      
      
      //------------------------------------------Performance Parameters for model--------------------------------
      // RMSE value for Random Forest      
     val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
     val rmse = evaluator.evaluate(predictions)
     println("Root Mean Squared Error (RMSE) on test data = " + rmse)
      
      //----------------------------------Accuracy precsion and recall----------------------------------------------
   
      val finalmetricCheck=spark.sql("select round(prediction) as prediction,label from PRDTable")
      finalmetricCheck.rdd.repartition(1).saveAsTextFile(args(2))
      
      //val checkabd=finalmetricRDD.map(line=>line.replaceALL
      val finalmetricRDD=sc.textFile(args(2)).map(line=>line.replace("[","")).map(line=>line.replace("]","")).map(line=>line.split(',')).map{line=>(line(0).toDouble,line(1).toDouble)}
      val metrics = new MulticlassMetrics(finalmetricRDD)

          // Overall Statistics
        val accuracy = metrics.accuracy
        println("Summary Statistics")
        println(s"Accuracy = $accuracy")
        
        // Precision by label
        val labels = metrics.labels
        labels.foreach { l =>
          println(s"Precision($l) = " + metrics.precision(l))
        }
        
        // Recall by label
        labels.foreach { l =>
          println(s"Recall($l) = " + metrics.recall(l))
        }

}}
      
    
    
    
    
    
    
    
    
    
    
    
    
   