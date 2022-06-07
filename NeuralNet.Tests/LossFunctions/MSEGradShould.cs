namespace NeuralNet.Tests.LossFunctions;

using MathNet.Numerics.LinearAlgebra;

public class MSEGradShould
{
    private static readonly VectorBuilder<double> Builder = Vector<double>.Build;

    [Theory]
    [InlineData(new[] { 1.0, 1.0, 1.0 }, new[] { 2.0, 2.0, 2.0 }, new[] { -2.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0 })]
    [InlineData(new[] { 1.0, 1.0, 1.0 }, new[] { 1.0, 1.0, 1.0 }, new[] { 0.0, 0.0, 0.0 })]
    [InlineData(new[] { 1.0, 2.0, 3.0 }, new[] { 4.0, 5.0, 6.0 }, new[] { -2.0, -2.0, -2.0 })]
    [InlineData(new[] { 0.0, 0.0, 0.0 }, new[] { 1.0, 2.0, 3.0 }, new[] { -2.0 / 3.0, -4.0 / 3.0, -2.0 })]
    [InlineData(new[] { 3.0, 2.0, 1.0 }, new[] { 4.0, 5.0, 6.0 }, new[] { -2.0 / 3.0, -2.0, -10.0 / 3.0 })]
    public void MSELoss_SimpleInputs_ReturnAnswer(double[] value1, double[] value2, double[] expected)
    {
        var vec1 = Builder.DenseOfArray(value1);
        var vec2 = Builder.DenseOfArray(value2);
        var vecExpected = Builder.DenseOfArray(expected);

        var actual = NeuralNet.LossFunctions.MSE.Grad(vec1, vec2);

        Assert.Equal(vecExpected, actual);
    }

    [Fact]
    public void MSELoss_EmptyInputs_ThrowArgumentError()
    {
        var vec = Builder.DenseOfArray(Array.Empty<double>());

        Assert.Throws<ArgumentException>(() => NeuralNet.LossFunctions.MSE.Grad(vec, vec));
    }

    [Fact]
    public void MSELoss_UnequalSize_ThrowArgumentError()
    {
        var vec1 = Builder.DenseOfArray(new[] { 1.0, 2.0 });
        var vec2 = Builder.DenseOfArray(new[] { 1.0, 2.0, 3.0, 4.0 });

        Assert.Throws<ArgumentException>(() => NeuralNet.LossFunctions.MSE.Grad(vec1, vec2));
    }
}