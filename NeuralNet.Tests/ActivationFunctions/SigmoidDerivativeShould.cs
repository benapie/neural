using MathNet.Numerics.LinearAlgebra;

namespace NeuralNet.Tests.ActivationFunctions;

public class SigmoidDerivativeShould
{
    private static readonly VectorBuilder<double> Builder = Vector<double>.Build;

    [Fact]
    public void SigmoidActivation_EmptyInputs_AssertArgumentError()
    {
        var vec = Builder.DenseOfArray(Array.Empty<double>());

        Assert.Throws<ArgumentException>(() => NeuralNet.ActivationFunctions.Sigmoid.Derivative(vec));
    }
}