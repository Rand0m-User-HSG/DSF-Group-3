# The code is based on the script of JPO, Day 1, Polynomial regression splines Ex_2_2_polynomial_regression_parallel.txt


# install.packages("doParallel")
library(doParallel)

# Parallel computation
cl = makeCluster(4)
#registerDoSNOW(cl)
registerDoParallel(cl)
