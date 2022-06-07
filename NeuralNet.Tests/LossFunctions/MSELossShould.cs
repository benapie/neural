namespace NeuralNet.Tests.LossFunctions;

using MathNet.Numerics.LinearAlgebra;
using NeuralNet;

public class MSELossShould
{
    private static readonly VectorBuilder<double> Builder = Vector<double>.Build;

    [Theory]
    [InlineData(new double[] { 1, 1, 1 }, new double[] { 2, 2, 2 }, 1)]
    [InlineData(new double[] { 1, 1, 1 }, new double[] { 1, 1, 1 }, 0)]
    [InlineData(new double[] { 1, 2, 3 }, new double[] { 4, 5, 6 }, 9)]
    [InlineData(new double[] { 0, 0, 0 }, new double[] { 1, 2, 3 }, 14 / 3.0)]
    [InlineData(new double[] { 3, 2, 1 }, new double[] { 4, 5, 6 }, 35 / 3.0)]
    public void MSELoss_SimpleInputs_ReturnAnswer(double[] value1, double[] value2, double expected)
    {
        var vec1 = Builder.DenseOfArray(value1);
        var vec2 = Builder.DenseOfArray(value2);

        var actual = LossFunctions.MSE.Loss(vec1, vec2);

        Assert.Equal(expected, actual);
    }

    [Fact]
    public void MSELoss_EmptyInputs_AssertArgumentError()
    {
        var vec = Builder.DenseOfArray(Array.Empty<double>());

        Assert.Throws<ArgumentException>(() => LossFunctions.MSE.Loss(vec, vec));
    }

    [Fact]
    public void MSELoss_UnequalSize_AssertArgumentError()
    {
        var vec1 = Builder.DenseOfArray(new double[] { 1, 2 });
        var vec2 = Builder.DenseOfArray(new double[] { 1, 2, 3, 4 });

        Assert.Throws<ArgumentException>(() => LossFunctions.MSE.Loss(vec1, vec2));
    }
}