namespace NeuralNet.Tests.Layers.Linear;


public class LearningRateShould
{
    [Theory]
    [InlineData(-1)]
    [InlineData(2)]
    public void Set_OutOfRange_ThrowOutOfRange(float value)
    {
        var layer = new NeuralNet.Layers.Linear(5, 5);
        
        Assert.Throws<ArgumentOutOfRangeException>(() => layer.LearningRate = value);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void Set_AtBoundary_NoThrow(float value)
    {
        var layer = new NeuralNet.Layers.Linear(5, 5);

        var exception = Record.Exception(() => layer.LearningRate = value);

        Assert.Null(exception);
    }
    
    [Theory]
    [InlineData(0.2)]
    [InlineData(0.66)]
    public void Set_WithinRange_ValueUpdates(float value)
    {
        var layer = new NeuralNet.Layers.Linear(5, 5);

        layer.LearningRate = value;
        
        Assert.Equal(value, layer.LearningRate);
    }
}