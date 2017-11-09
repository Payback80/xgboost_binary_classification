# XGBOOST binary prediction examples

I have created examples on how to perform binary classification using xgboost and the standarnd R datasets, i've uploaded 3 examples and maybe i will add more.
These examples's code are almost identical. I have uploaded because in one case (breast_cancer) i had to transform data.frame object to matrix because xgb.Dmatrix gave me error with the dimensions of the datasets. 
I have read of many people with the same problem on StackOverflow and i've figured out what i think a very easy solution:
First extract the outcome column as a vector eg: train.label  <- train$class
Second transform the input values X1....Xn to a matrix  eg: train<- as.matrix(train[1:9])
In such way the creation of xgboost.Dmatrix is flawless 

The prediction accuracy is very high and xgboost is very fast so i hope you could use my code as a tutorial for your projects/learning 



## Author

* **Andrea Mariani** - *Initial work* - [Payback80](https://github.com/Payback80)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

