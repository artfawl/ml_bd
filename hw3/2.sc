import LinReg._

val m = new LinearRegression(100, 1e-4, 64)
m.readData("/home/anaxagoras/MADE/ml_bd/hw3/files/train.csv")
println("done")