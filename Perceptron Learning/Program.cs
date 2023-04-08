namespace Perceptron_Learning
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // define training data
            double[][] trainingData = new double[][] {
                new double[] {1, 1, 1},
                new double[] {1, -1, -1},
                new double[] {-1, 1, -1},
                new double[] {-1, -1, -1},
            };
            //trainingData = MinMaxScale(trainingData);
            //    // define training data
            //    double[][] trainingData = new double[][] {
            //    new double[] {0.22, 0.14, -1},
            //    new double[] {0.67, 1, 1},
            //    new double[] {1, 0.97, 1},
            //    new double[] {0.33, 0.33, -1},
            //    new double[] {0, 0, -1},
            //    new double[] {0.78, 0.69, 1},
            //};

            // initialize weights to small random values
            double[] weights = new double[] { 0, 0 };

            // set learning rate
            double alpha = 0.1;
            double b = 0;

            // initialize variables for convergence check
            bool converged = false;
            int epoch = 0;

            System.Diagnostics.Stopwatch sw = System.Diagnostics.Stopwatch.StartNew();

            // train the perceptron
            while (!converged) // maximum 1000 epochs
            {
                int errors = 0; // misclassification error counter

                for (int i = 0; i < trainingData.Length; i++)
                {
                    double[] x = new double[] { trainingData[i][0], trainingData[i][1] };
                    double y = trainingData[i][2];

                    double y_hat = Sign(DotProduct(weights, x) + b);

                    if (y_hat != y)
                    {
                        for (int j = 0; j < weights.Length; j++)
                        {
                            weights[j] = Math.Round(weights[j] + alpha * (y - y_hat) * x[j], 2);
                        }
                        b = Math.Round(b + alpha * (y - y_hat), 2);
                        errors++;
                    }
                }

                if (errors == 0) // if there are no errors, the perceptron has converged
                {
                    converged = true;
                }

                epoch++; // increment epoch counter
                //alpha += alpha + 0.1;
            }

            //// iterate over training data and update weights
            //for (int i = 0; i < trainingData.Length; i++)
            //{
            //    double[] x = new double[] { trainingData[i][0], trainingData[i][1] };
            //    double y = trainingData[i][2];

            //    double y_hat = Sign(DotProduct(weights, x));

            //    if (y_hat != y)
            //    {
            //        for (int j = 0; j < weights.Length; j++)
            //        {
            //            weights[j] = weights[j] + alpha * (y - y_hat) * x[j];
            //        }
            //        b += b + alpha * (y - y_hat);
            //    }
            //}
            sw.Stop();
            // print final weights
            Console.WriteLine("Final weights: [{0}, {1}]", weights[0], weights[1]);
            Console.WriteLine($"Bias: {b}");
            Console.WriteLine($"Epoch: {epoch}");
            Console.WriteLine($"Alpha: {alpha}");
            Console.WriteLine($"Time: {sw.Elapsed.TotalMilliseconds}");
            Console.ReadLine();
        }

        static double Sign(double x)
        {
            if (x > 0) return 1;
            else if (x == 0) return 0;
            else return -1;
        }

        static double DotProduct(double[] a, double[] b)
        {
            double dotProduct = 0;

            for (int i = 0; i < a.Length; i++)
            {
                dotProduct += a[i] * b[i];
            }

            return dotProduct;
        }

        static double[][] MinMaxScale(double[][] data)
        {
            double[] mins = new double[data[0].Length];
            double[] maxs = new double[data[0].Length];
            for (int i = 0; i < data[0].Length; i++)
            {
                mins[i] = double.MaxValue;
                maxs[i] = double.MinValue;
            }

            // calculate minimum and maximum values for each feature
            for (int i = 0; i < data.Length; i++)
            {
                for (int j = 0; j < data[i].Length; j++)
                {
                    if (data[i][j] < mins[j]) mins[j] = data[i][j];
                    if (data[i][j] > maxs[j]) maxs[j] = data[i][j];
                }
            }

            // perform min-max scaling
            for (int i = 0; i < data.Length; i++)
            {
                for (int j = 0; j < data[i].Length-1; j++)
                {
                    data[i][j] = Math.Round((data[i][j] - mins[j]) / (maxs[j] - mins[j]), 2);
                }
            }

            return data;
        }
    }
}