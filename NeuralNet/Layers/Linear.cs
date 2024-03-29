﻿using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;

// unnecessary warning on uninitialized fields
#pragma warning disable CS8618


namespace NeuralNet.Layers;

public class Linear
{
    private Matrix<double> _weights;
    private Vector<double> _biases;
    private ActivationFunction _activationFunction;
    private double _learningRate;

    public Linear(int inputCount, int outputCount)
    {
        InitializeWeightsAndBiases(inputCount, outputCount);
        _activationFunction = ActivationFunctions.Sigmoid;
        _learningRate = 0.1;
    }

    public Linear(int inputCount, int outputCount, ActivationFunction activationFunction)
    {
        InitializeWeightsAndBiases(inputCount, outputCount);
        _activationFunction = activationFunction;
        _learningRate = 0.1;
    }

    public Linear(int inputCount, int outputCount, float learningRate)
    {
        InitializeWeightsAndBiases(inputCount, outputCount);
        _activationFunction = ActivationFunctions.Sigmoid;
        _learningRate = learningRate;
    }

    public Linear(int inputCount, int outputCount, ActivationFunction activationFunction, float learningRate)
    {
        InitializeWeightsAndBiases(inputCount, outputCount);
        _activationFunction = activationFunction;
        _learningRate = learningRate;
    }


    public int InputCount => _weights.ColumnCount;
    public int OutputCount => _weights.RowCount;

    public double LearningRate
    {
        get => _learningRate;
        set
        {
            if (value is < 0 or > 1)
            {
                throw new ArgumentOutOfRangeException(nameof(value), "Must be between 0 and 1.");
            }

            _learningRate = value;
        }
    }

    private void InitializeWeightsAndBiases(int inputCount, int outputCount)
    {
        var matrixBuilder = Matrix<double>.Build;
        var uniform = new ContinuousUniform(-1, 1);
        _weights = matrixBuilder.Random(outputCount, inputCount, uniform);

        var vectorBuilder = Vector<double>.Build;
        _biases = vectorBuilder.Dense(outputCount, 0);
    }
}