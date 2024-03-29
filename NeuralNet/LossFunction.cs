﻿using MathNet.Numerics.LinearAlgebra;

namespace NeuralNet;

public struct LossFunction
{
    public readonly Func<Vector<double>, Vector<double>, double> Loss;
    public readonly Func<Vector<double>, Vector<double>, Vector<double>> Grad;

    public LossFunction(Func<Vector<double>, Vector<double>, double> loss,
        Func<Vector<double>, Vector<double>, Vector<double>> grad)
    {
        Loss = loss;
        Grad = grad;
    }
}

public static class LossFunctions
{
    public static readonly LossFunction MSE = new LossFunction(MSEBase, MSEGradBase);

    private static bool IsEmpty(ICollection<double> x)
    {
        return x.Count == 0;
    }

    private static double MSEBase(Vector<double> actual, Vector<double> predicted)
    {
        if (actual.Count != predicted.Count)
        {
            throw new ArgumentException(
                $"Inputs have length {actual.Count} and {predicted.Count}, but they should match.");
        }

        if (IsEmpty(actual))
        {
            throw new ArgumentException($"Inputs are empty.");
        }

        var n = actual.Count;

        double diffSum = 0;
        for (var i = 0; i < n; i++)
        {
            var diff = actual[i] - predicted[i];
            diffSum += Math.Pow(diff, 2);
        }

        var output = diffSum / n;
        return output;
    }

    private static Vector<double> MSEGradBase(Vector<double> actual, Vector<double> predicted)
    {
        if (actual.Count != predicted.Count)
        {
            throw new ArgumentException(
                $"Inputs have length {actual.Count} and {predicted.Count}, but they should match.");
        }
        
        
        if (IsEmpty(actual))
        {
            throw new ArgumentException($"Inputs are empty.");
        }

        var n = actual.Count;
        var diff = actual - predicted;
        var output = (2 * diff) / n;
        return output;
    }
}