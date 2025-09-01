using System;

public enum Act { Tanh, ReLU, Sigmoid }

public static class Activations
{
    public static (Func<float, float> f, Func<float, float> df) Get(Act a)
    {
        switch (a)
        {
            case Act.ReLU:
                return (x => x > 0 ? x : 0f,
                        x => x > 0 ? 1f : 0f);
            case Act.Sigmoid:
                return (x => 1f / (1f + (float)Math.Exp(-x)),
                        x => { float s = 1f / (1f + (float)Math.Exp(-x)); return s * (1f - s); }
                );
            default:
                return (x => (float)Math.Tanh(x),
                        x => { float t = (float)Math.Tanh(x); return 1f - t * t; }
                );
        }
    }
}
