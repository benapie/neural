using MathNet.Numerics.LinearAlgebra;

namespace NeuralNet;

public struct ActivationFunction
{
    public readonly Func<Vector<double>, Vector<double>> Activation;
    public readonly Func<Vector<double>, Vector<double>> Derivative;

    public ActivationFunction(Func<Vector<double>, Vector<double>> activation,
        Func<Vector<double>, Vector<double>> derivative)
    {
        Activation = activation;
        Derivative = derivative;
    }
}

public static class ActivationFunctions
{
    public static readonly ActivationFunction Sigmoid = new ActivationFunction(SigmoidVec, SigmoidDerVec);

    private static Vector<double> MapToVec(IList<double> x, Func<double, double> func)
    {
        return Vector<double>.Build.Dense(x.Count, i => func(x[i]));
    }

    private static double SigmoidBase(double x)
    {
        /*
         *                1
         * sig(x) = -------------
         *           1 + exp(-x)
         */
        var output = 1 + Math.Exp(-x);
        output = 1 / (output);
        return output;
    }

    private static double SigmoidDerBase(double x)
    {
        /*
         *                exp(x)
         * sig'(x) = ----------------
         *            (1 + exp(x))^2
         */
        var output = 1 + Math.Exp(x);
        output = Math.Pow(output, 2);
        output = 1 / output;
        output = output * Math.Exp(x);
        return output;
    }

    private static Vector<double> SigmoidVec(IList<double> x)
    {
        return MapToVec(x, SigmoidBase);
    }

    private static Vector<double> SigmoidDerVec(IList<double> x)
    {
        return MapToVec(x, SigmoidDerBase);
    }
}