# install.packages("doParallel")
library(doParallel)

# Parallel computation
cl = makeCluster(4)
#registerDoSNOW(cl)
registerDoParallel(cl)
