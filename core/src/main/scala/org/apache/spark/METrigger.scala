package org.apache.spark

/**
 * Trait model evaluation trigger, which will be implemented by SparkContext to trigger dynamic model evaluation
 */
trait METrigger {
  var RMRequest: Boolean = false

  def setMETrigger(state: Boolean): Unit ={
    RMRequest = state
  }

  def getMETrigger(): Boolean ={
    return RMRequest
  }
}
